"""Configuration management for CityZero."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MAPS_DATA_DIR = DATA_DIR / "maps"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_OUTPUT_DIR = OUTPUT_DIR / "models"
RENDERS_OUTPUT_DIR = OUTPUT_DIR / "renders"


@dataclass
class MapillaryConfig:
    """Mapillary API configuration."""
    
    client_token: str
    
    def __post_init__(self):
        if not self.client_token or self.client_token == "your_mapillary_token_here":
            raise ValueError(
                "MAPILLARY_CLIENT_TOKEN not set. "
                "Please copy .env.example to .env and add your token."
            )


@dataclass
class BoundingBox:
    """Geographic bounding box (west, south, east, north)."""
    
    west: float
    south: float
    east: float
    north: float
    
    @classmethod
    def from_string(cls, bbox_string: str) -> "BoundingBox":
        """Parse bounding box from comma-separated string."""
        parts = bbox_string.split(",")
        if len(parts) != 4:
            raise ValueError(f"Invalid bbox format: {bbox_string}")
        
        return cls(
            west=float(parts[0]),
            south=float(parts[1]),
            east=float(parts[2]),
            north=float(parts[3])
        )
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return as tuple (west, south, east, north)."""
        return (self.west, self.south, self.east, self.north)


def get_mapillary_config() -> MapillaryConfig:
    """Get Mapillary configuration from environment."""
    token = os.getenv("MAPILLARY_CLIENT_TOKEN", "")
    return MapillaryConfig(client_token=token)


def get_sf_bbox() -> BoundingBox:
    """Get San Francisco bounding box from environment or use default."""
    bbox_string = os.getenv("SF_BBOX", "-122.5147,37.7034,-122.3549,37.8324")
    return BoundingBox.from_string(bbox_string)
