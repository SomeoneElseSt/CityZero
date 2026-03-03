"""Configuration management for Mapillary client and CLI downloader."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class MapillaryConfig:
    """Mapillary API configuration."""

    client_token: str


@dataclass
class BoundingBox:
    """Geographic bounding box (west, south, east, north)."""
    
    west: float
    south: float
    east: float
    north: float
    
    @classmethod
    def from_string(cls, bbox_string: str) -> "BoundingBox | None":
        """Parse bounding box from comma-separated string, or None if invalid."""
        parts = bbox_string.split(",")
        if len(parts) != 4:
            return None
        try:
            return cls(west=float(parts[0]), south=float(parts[1]), east=float(parts[2]), north=float(parts[3]))
        except ValueError:
            return None
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return as tuple (west, south, east, north)."""
        return (self.west, self.south, self.east, self.north)


def get_mapillary_config() -> MapillaryConfig | None:
    """Get Mapillary configuration from environment, or None if token is missing."""
    token = os.getenv("MAPILLARY_CLIENT_TOKEN", "")
    if not token:
        return None
    return MapillaryConfig(client_token=token)


# Predefined city bounding boxes (can be extended)
CITY_BBOXES: dict[str, BoundingBox] = {
    "san francisco": BoundingBox(
        west=-122.5147,
        south=37.7034,
        east=-122.3549,
        north=37.8324
    ),
    "new york": BoundingBox(
        west=-74.0479,
        south=40.6829,
        east=-73.9067,
        north=40.8820
    ),
    "los angeles": BoundingBox(
        west=-118.6682,
        south=33.7037,
        east=-118.1553,
        north=34.3373
    ),
    "chicago": BoundingBox(
        west=-87.9401,
        south=41.6444,
        east=-87.5241,
        north=42.0230
    ),
    "miami": BoundingBox(
        west=-80.3203,
        south=25.7090,
        east=-80.1300,
        north=25.8554
    ),
}
