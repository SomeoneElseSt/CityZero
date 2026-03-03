"""Mapillary API client and image downloader for street view imagery."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import mapillary.interface as mly
import requests
from tqdm import tqdm

from config import BoundingBox, MapillaryConfig, DATA_DIR


MAX_RESOLUTION = 2048
GRID_CELL_SIZE = 0.01
SAVE_INTERVAL = 10

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
        self.metadata_file = self.output_dir / "download_metadata.json"
        self.images_metadata_file = self.output_dir / "images_metadata.json"

    def get_downloaded_image_ids(self) -> Set[str]:
        """Get set of already downloaded image IDs."""
        downloaded_ids = set()

        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                downloaded_ids.update(metadata.get('downloaded_ids', []))

        for file_path in self.output_dir.glob("*.jpg"):
            downloaded_ids.add(file_path.stem)

        return downloaded_ids

    def save_metadata(self, downloaded_ids: Set[str], total_found: int):
        """Save download metadata to track progress."""
        metadata = {
            'total_found': total_found,
            'total_downloaded': len(downloaded_ids),
            'downloaded_ids': list(downloaded_ids)
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_images_metadata(self, images: List[Dict]):
        """Save full metadata for downloaded images (GPS coords, timestamps, etc)."""
        metadata_list = []
        for img in images:
            if not img.get('id'):
                continue

            geometry = img.get('geometry', {})
            coords = geometry.get('coordinates', []) if geometry else []

            entry = {
                'id': img.get('id'),
                'longitude': coords[0] if len(coords) > 0 else None,
                'latitude': coords[1] if len(coords) > 1 else None,
                'captured_at': img.get('captured_at'),
                'compass_angle': img.get('compass_angle'),
                'sequence': img.get('sequence'),
                'is_pano': img.get('is_pano'),
            }

            entry.update({dst: img[src] for src, dst in OPTIONAL_FIELDS.items() if src in img})

            metadata_list.append(entry)

        with open(self.images_metadata_file, 'w') as f:
            json.dump(metadata_list, f, indent=2)

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

    def discover_images(self, bbox: BoundingBox) -> List[Dict]:
        """Discover all available images in bounding box."""
        print(f"\n🔍 Discovering images in area...")
        print(f"   Bbox: {bbox.to_tuple()}")

        cells = self.split_bbox_into_grid(bbox)
        print(f"   Searching {len(cells)} grid cells...")

        all_images = []
        seen_ids = set()

        for cell in tqdm(cells, desc="Discovering", unit="cell"):
            images = self.client.get_images_in_bbox(cell, limit=5000)
            for img in images:
                img_id = img.get('id')
                if img_id and img_id not in seen_ids:
                    all_images.append(img)
                    seen_ids.add(img_id)

        print(f"\n✓ Found {len(all_images)} unique images")
        return all_images

    def download_images(self, bbox: BoundingBox, max_images: int = None, images: List[Dict] = None) -> Dict[str, int]:
        """Download all images in bounding box. Pass `images` to skip rediscovery."""
        print("\n" + "="*70)
        print("CityZero Image Downloader")
        print("="*70)

        downloaded_ids = self.get_downloaded_image_ids()
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
                    downloaded_ids.add(img_id)
                    success_count += 1
                else:
                    failed_count += 1

                pbar.update(1)

                if success_count % SAVE_INTERVAL == 0:
                    self.save_metadata(downloaded_ids, len(all_images))

        self.save_metadata(downloaded_ids, len(all_images))
        self.save_images_metadata(all_images)

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
