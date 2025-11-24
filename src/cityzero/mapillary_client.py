"""Mapillary API client for fetching street view imagery."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import requests
from mapillary import Mapillary

from cityzero.config import BoundingBox, MapillaryConfig


class MapillaryClient:
    """Client for interacting with Mapillary API."""
    
    BASE_URL = "https://graph.mapillary.com"
    
    def __init__(self, config: MapillaryConfig):
        """Initialize Mapillary client.
        
        Args:
            config: Mapillary API configuration
        """
        self.config = config
        self.client = Mapillary(access_token=config.client_token)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"OAuth {config.client_token}"
        })
    
    def get_images_in_bbox(
        self,
        bbox: BoundingBox,
        limit: int = 1000,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[Dict]:
        """Get images within a bounding box.
        
        Args:
            bbox: Geographic bounding box
            limit: Maximum number of images to retrieve
            start_time: Start time in ISO format (e.g., '2020-01-01T00:00:00')
            end_time: End time in ISO format
            
        Returns:
            List of image metadata dictionaries
        """
        url = f"{self.BASE_URL}/images"
        
        params = {
            "bbox": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}",
            "limit": limit,
            "fields": "id,geometry,captured_at,compass_angle,sequence,is_pano"
        }
        
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        response = self.session.get(url, params=params)
        
        if response.status_code != 200:
            return []
        
        data = response.json()
        return data.get("data", [])
    
    def get_image_metadata(self, image_id: str) -> Optional[Dict]:
        """Get detailed metadata for a specific image.
        
        Args:
            image_id: Mapillary image ID
            
        Returns:
            Image metadata dictionary or None if not found
        """
        url = f"{self.BASE_URL}/{image_id}"
        
        params = {
            "fields": "id,geometry,captured_at,compass_angle,sequence,is_pano,altitude,camera_type,creator,height,width,thumb_256_url,thumb_1024_url,thumb_2048_url"
        }
        
        response = self.session.get(url, params=params)
        
        if response.status_code != 200:
            return None
        
        return response.json()
    
    def download_image(
        self,
        image_id: str,
        output_path: Path,
        resolution: int = 2048
    ) -> bool:
        """Download an image at specified resolution.
        
        Args:
            image_id: Mapillary image ID
            output_path: Path to save the image
            resolution: Desired resolution (256, 1024, or 2048)
            
        Returns:
            True if successful, False otherwise
        """
        valid_resolutions = [256, 1024, 2048]
        if resolution not in valid_resolutions:
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
    
    def save_image_metadata(
        self,
        images: List[Dict],
        output_path: Path
    ) -> bool:
        """Save image metadata to JSON file.
        
        Args:
            images: List of image metadata dictionaries
            output_path: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, "w") as f:
                json.dump(images, f, indent=2)
            return True
        except Exception:
            return False
    
    def get_coverage_stats(self, bbox: BoundingBox) -> Dict:
        """Get statistics about image coverage in a bounding box.
        
        Args:
            bbox: Geographic bounding box
            
        Returns:
            Dictionary with coverage statistics
        """
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
