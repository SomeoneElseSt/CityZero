"""Image downloader with progress tracking and resume capability."""

import json
from pathlib import Path
from typing import Dict, List, Set

from tqdm import tqdm

from cityzero.config import BoundingBox, RAW_DATA_DIR
from cityzero.mapillary_client import MapillaryClient


class ImageDownloader:
    """Downloads Mapillary images with progress tracking."""
    
    # Maximum resolution available from Mapillary API (thumb_2048_url)
    # Options: 256, 1024, 2048 (higher = better quality but larger files)
    MAX_RESOLUTION = 2048
    
    # Grid cell size in degrees (approximately 1km x 1km)
    # Smaller cells = more API calls but avoids hitting result limits
    GRID_CELL_SIZE = 0.01
    
    # Save metadata every N images to prevent data loss on interruption
    SAVE_INTERVAL = 10
    
    def __init__(self, client: MapillaryClient, output_dir: Path = RAW_DATA_DIR):
        """Initialize image downloader.
        
        Args:
            client: Mapillary API client
            output_dir: Directory to save downloaded images
        """
        self.client = client
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file to track downloads
        self.metadata_file = self.output_dir / "download_metadata.json"
    
    def get_downloaded_image_ids(self) -> Set[str]:
        """Get set of already downloaded image IDs.
        
        Returns:
            Set of image IDs that have been downloaded
        """
        downloaded_ids = set()
        
        # Check metadata file
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                downloaded_ids.update(metadata.get('downloaded_ids', []))
        
        # Also check actual files on disk
        for file_path in self.output_dir.glob("*.jpg"):
            image_id = file_path.stem
            downloaded_ids.add(image_id)
        
        return downloaded_ids
    
    def save_metadata(self, downloaded_ids: Set[str], total_found: int):
        """Save download metadata to track progress.
        
        Args:
            downloaded_ids: Set of successfully downloaded image IDs
            total_found: Total number of images discovered
        """
        metadata = {
            'total_found': total_found,
            'total_downloaded': len(downloaded_ids),
            'downloaded_ids': list(downloaded_ids)
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def split_bbox_into_grid(self, bbox: BoundingBox) -> List[BoundingBox]:
        """Split large bounding box into smaller grid cells.
        
        Args:
            bbox: Large bounding box to split
            
        Returns:
            List of smaller bounding boxes
        """
        cells = []
        
        # Calculate number of cells needed
        lon_cells = int((bbox.east - bbox.west) / self.GRID_CELL_SIZE) + 1
        lat_cells = int((bbox.north - bbox.south) / self.GRID_CELL_SIZE) + 1
        
        for i in range(lon_cells):
            for j in range(lat_cells):
                cell_west = bbox.west + (i * self.GRID_CELL_SIZE)
                cell_east = min(cell_west + self.GRID_CELL_SIZE, bbox.east)
                cell_south = bbox.south + (j * self.GRID_CELL_SIZE)
                cell_north = min(cell_south + self.GRID_CELL_SIZE, bbox.north)
                
                cells.append(BoundingBox(
                    west=cell_west,
                    south=cell_south,
                    east=cell_east,
                    north=cell_north
                ))
        
        return cells
    
    def discover_images(self, bbox: BoundingBox) -> List[Dict]:
        """Discover all available images in bounding box.
        
        Args:
            bbox: Geographic bounding box to search
            
        Returns:
            List of image metadata dictionaries
        """
        print(f"\n🔍 Discovering images in area...")
        print(f"   Bbox: {bbox.to_tuple()}")
        
        # Split into grid cells to avoid API limits
        cells = self.split_bbox_into_grid(bbox)
        print(f"   Searching {len(cells)} grid cells...")
        
        all_images = []
        seen_ids = set()
        
        # Query each cell with progress bar
        for cell in tqdm(cells, desc="Discovering", unit="cell"):
            images = self.client.get_images_in_bbox(cell, limit=10000)
            
            # Deduplicate images (same image might appear in adjacent cells)
            for img in images:
                img_id = img.get('id')
                if img_id and img_id not in seen_ids:
                    all_images.append(img)
                    seen_ids.add(img_id)
        
        print(f"\n✓ Found {len(all_images)} unique images")
        return all_images
    
    def download_images(
        self,
        bbox: BoundingBox,
        max_images: int = None
    ) -> Dict[str, int]:
        """Download all images in bounding box.
        
        Args:
            bbox: Geographic bounding box
            max_images: Maximum number of images to download (None = all)
            
        Returns:
            Dictionary with download statistics
        """
        print("\n" + "="*70)
        print("CityZero Image Downloader")
        print("="*70)
        
        # Check for already downloaded images
        downloaded_ids = self.get_downloaded_image_ids()
        if downloaded_ids:
            print(f"\n📂 Found {len(downloaded_ids)} already downloaded images")
            print("   (Will skip these to resume download)")
        
        # Discover all available images
        all_images = self.discover_images(bbox)
        
        if not all_images:
            print("\n❌ No images found in this area")
            return {'total_found': 0, 'downloaded': 0, 'skipped': 0, 'failed': 0}
        
        # Filter out already downloaded
        images_to_download = [
            img for img in all_images 
            if img.get('id') not in downloaded_ids
        ]
        
        skipped_count = len(all_images) - len(images_to_download)
        
        # Apply max_images limit
        if max_images and len(images_to_download) > max_images:
            images_to_download = images_to_download[:max_images]
            print(f"\n⚠️  Limiting download to {max_images} images")
        
        if not images_to_download:
            print(f"\n✓ All {len(all_images)} images already downloaded!")
            return {
                'total_found': len(all_images),
                'downloaded': 0,
                'skipped': skipped_count,
                'failed': 0
            }
        
        print(f"\n📥 Downloading {len(images_to_download)} images...")
        print(f"   Resolution: {self.MAX_RESOLUTION}px")
        print(f"   Output: {self.output_dir}")
        
        # Download with progress bar
        failed_count = 0
        success_count = 0
        
        with tqdm(total=len(images_to_download), desc="Downloading", unit="img") as pbar:
            for img in images_to_download:
                img_id = img.get('id')
                if not img_id:
                    continue
                
                output_path = self.output_dir / f"{img_id}.jpg"
                
                success = self.client.download_image(
                    image_id=img_id,
                    output_path=output_path,
                    resolution=self.MAX_RESOLUTION
                )
                
                if success:
                    downloaded_ids.add(img_id)
                    success_count += 1
                else:
                    failed_count += 1
                
                pbar.update(1)
                
                # Save metadata periodically to prevent data loss
                if success_count % self.SAVE_INTERVAL == 0:
                    self.save_metadata(downloaded_ids, len(all_images))
        
        # Final metadata save
        self.save_metadata(downloaded_ids, len(all_images))
        
        # Print summary
        print("\n" + "="*70)
        print("Download Complete")
        print("="*70)
        print(f"Total found:       {len(all_images):,}")
        print(f"Already had:       {skipped_count:,}")
        print(f"Downloaded:        {success_count:,}")
        print(f"Failed:            {failed_count:,}")
        print(f"Total on disk:     {len(downloaded_ids):,}")
        print("="*70)
        
        return {
            'total_found': len(all_images),
            'downloaded': success_count,
            'skipped': skipped_count,
            'failed': failed_count
        }
