"""Mapillary API client and image downloader for street view imagery."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import mapillary.interface as mly
import requests
from tqdm import tqdm

from config import BoundingBox, MapillaryConfig, DATA_DIR
from db import DiscoveryDB


MAX_RESOLUTION = 2048
API_IMAGE_LIMIT = 2000
# Important: lower to increase initial cell count
GRID_CELL_SIZE = 0.2
# Important: lower to increase resolution
MIN_CELL_SIZE = 0.1
DISCOVERY_WORKERS = 30

OPTIONAL_FIELDS = {
    'altitude': 'altitude',
    'camera_type': 'camera_type',
    'creator': 'creator',
    'height': 'image_height',
    'width': 'image_width',
}


class MapillaryClient:
    """Client for interacting with Mapillary API."""

    BASE_URL = "https://graph.mapillary.com"

    def __init__(self, config: MapillaryConfig):
        self.config = config
        mly.set_access_token(config.client_token)
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"OAuth {config.client_token}"})

    def get_images_in_bbox(
        self,
        bbox: BoundingBox,
        limit: int = 1000,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict]:
        """Get images within a bounding box."""
        url = f"{self.BASE_URL}/images"
        params = {
            "bbox": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}",
            "limit": limit,
            "fields": "id,geometry,captured_at,compass_angle,sequence,is_pano,altitude,camera_type,creator,height,width"
        }

        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        response = self.session.get(url, params=params)
        if response.status_code != 200:
            return []

        return response.json().get("data", [])

    def get_image_metadata(self, image_id: str) -> Optional[Dict]:
        """Get detailed metadata for a specific image."""
        url = f"{self.BASE_URL}/{image_id}"
        params = {
            "fields": "id,geometry,captured_at,compass_angle,sequence,is_pano,altitude,camera_type,creator,height,width,thumb_256_url,thumb_1024_url,thumb_2048_url"
        }

        response = self.session.get(url, params=params)
        if response.status_code != 200:
            return None

        return response.json()

    def download_image(self, image_id: str, output_path: Path, resolution: int = MAX_RESOLUTION) -> bool:
        """Download an image at specified resolution (256, 1024, or 2048)."""
        if resolution not in [256, 1024, 2048]:
            return False

        metadata = self.get_image_metadata(image_id)
        if not metadata:
            return False

        thumb_url = metadata.get(f"thumb_{resolution}_url")
        if not thumb_url:
            return False

        response = self.session.get(thumb_url, stream=True)
        if response.status_code != 200:
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True

    def get_coverage_stats(self, bbox: BoundingBox) -> Dict:
        """Get statistics about image coverage in a bounding box."""
        images = self.get_images_in_bbox(bbox, limit=10000)
        total_images = len(images)
        pano_count = sum(1 for img in images if img.get("is_pano"))
        sequences = set(img.get("sequence") for img in images if img.get("sequence"))

        return {
            "total_images": total_images,
            "panoramic_images": pano_count,
            "regular_images": total_images - pano_count,
            "unique_sequences": len(sequences),
            "bbox": bbox.to_tuple()
        }


class ImageDownloader:
    """Downloads Mapillary images with progress tracking."""

    def __init__(self, client: MapillaryClient, output_dir: Path = DATA_DIR):
        self.client = client
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _split_cell(self, cell: BoundingBox) -> List[BoundingBox]:
        """Split a cell into 4 equal quadrants."""
        mid_lon = (cell.west + cell.east) / 2
        mid_lat = (cell.south + cell.north) / 2
        return [
            BoundingBox(cell.west, cell.south, mid_lon, mid_lat),
            BoundingBox(mid_lon, cell.south, cell.east, mid_lat),
            BoundingBox(cell.west, mid_lat, mid_lon, cell.north),
            BoundingBox(mid_lon, mid_lat, cell.east, cell.north),
        ]

    def _fetch_cell_images(self, cell: BoundingBox) -> List[Dict]:
        """Fetch images for a cell, recursively splitting if the API limit is hit.

        Stops recursing at MIN_CELL_SIZE
        """
        images = self.client.get_images_in_bbox(cell, limit=API_IMAGE_LIMIT)
        cell_size = min(cell.east - cell.west, cell.north - cell.south)
        if len(images) < API_IMAGE_LIMIT or cell_size <= MIN_CELL_SIZE:
            return images
        all_images = []
        for sub_cell in self._split_cell(cell):
            all_images.extend(self._fetch_cell_images(sub_cell))
        return all_images

    def split_bbox_into_grid(self, bbox: BoundingBox) -> List[BoundingBox]:
        """Split large bounding box into smaller grid cells."""
        cells = []
        lon_cells = int((bbox.east - bbox.west) / GRID_CELL_SIZE) + 1
        lat_cells = int((bbox.north - bbox.south) / GRID_CELL_SIZE) + 1

        for i in range(lon_cells):
            for j in range(lat_cells):
                cell_west = bbox.west + (i * GRID_CELL_SIZE)
                cell_east = min(cell_west + GRID_CELL_SIZE, bbox.east)
                cell_south = bbox.south + (j * GRID_CELL_SIZE)
                cell_north = min(cell_south + GRID_CELL_SIZE, bbox.north)

                cells.append(BoundingBox(
                    west=cell_west,
                    south=cell_south,
                    east=cell_east,
                    north=cell_north
                ))

        return cells

    def discover_images(self, bbox: BoundingBox, db: Optional["DiscoveryDB"] = None) -> List[Dict]:
        """Discover all available images in bounding box.

        If db is provided, inserts images into the DB as each cell completes so
        progress is preserved on Ctrl+C.
        """
        print(f"\n🔍 Discovering images in area...")
        print(f"   Bbox: {bbox.to_tuple()}")

        cells = self.split_bbox_into_grid(bbox)
        update_interval = max(1, len(cells) // 100)
        print(f"   Searching {len(cells)} grid cells...")
        print(f"   Time estimates refresh every {update_interval} cells")

        all_images = []
        seen_ids = set()
        completed = 0

        with ThreadPoolExecutor(max_workers=DISCOVERY_WORKERS) as executor:
            futures = {executor.submit(self._fetch_cell_images, cell): cell for cell in cells}
            with tqdm(total=len(cells), desc="Discovering", unit="cell") as pbar:
                for future in as_completed(futures):
                    cell_images = future.result() or []
                    new_images = []
                    for img in cell_images:
                        img_id = img.get('id')
                        if img_id and img_id not in seen_ids:
                            all_images.append(img)
                            new_images.append(img)
                            seen_ids.add(img_id)
                    if db and new_images:
                        db.insert_images(new_images)
                    completed += 1
                    pbar.set_postfix({"found": f"{len(all_images):,}"})
                    if completed % update_interval == 0:
                        pbar.update(update_interval)

        print(f"\n✓ Found {len(all_images)} unique images")
        return all_images

    def download_images(
        self,
        bbox: BoundingBox,
        db: DiscoveryDB,
        max_images: int = None,
        images: List[Dict] = None,
    ) -> Dict[str, int]:
        """Download images. Pass `images` to skip rediscovery. Uses db for tracking."""
        downloaded_ids = db.get_downloaded_ids()
        if downloaded_ids:
            print(f"\n📂 Found {len(downloaded_ids)} already downloaded images")
            print("   (Will skip these to resume download)")

        all_images = images if images is not None else self.discover_images(bbox)

        if not all_images:
            print("\n❌ No images found in this area")
            return {'total_found': 0, 'downloaded': 0, 'skipped': 0, 'failed': 0}

        images_to_download = [img for img in all_images if img.get('id') not in downloaded_ids]
        skipped_count = len(all_images) - len(images_to_download)

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
        print(f"   Resolution: {MAX_RESOLUTION}px")
        print(f"   Output: {self.output_dir}")

        failed_count = 0
        success_count = 0
        completed = 0
        update_interval = max(1, len(images_to_download) // 100)
        print(f"   Time estimates refresh every {update_interval} images")

        with tqdm(total=len(images_to_download), desc="Downloading", unit="img") as pbar:
            for img in images_to_download:
                img_id = img.get('id')
                if not img_id:
                    continue

                output_path = self.output_dir / f"{img_id}.jpg"
                success = self.client.download_image(
                    image_id=img_id,
                    output_path=output_path,
                    resolution=MAX_RESOLUTION
                )

                if success:
                    db.mark_downloaded(img_id)
                    success_count += 1
                else:
                    failed_count += 1

                completed += 1
                if completed % update_interval == 0:
                    pbar.update(update_interval)

        print("\n" + "="*70)
        print("Download Complete")
        print("="*70)
        print(f"Total found:       {len(all_images):,}")
        print(f"Already had:       {skipped_count:,}")
        print(f"Downloaded:        {success_count:,}")
        print(f"Failed:            {failed_count:,}")
        print("="*70)

        return {
            'total_found': len(all_images),
            'downloaded': success_count,
            'skipped': skipped_count,
            'failed': failed_count
        }
