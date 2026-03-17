"""Configuration management for Mapillary client and CLI downloader."""

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory (default for programmatic use)
DATA_DIR = Path.cwd()

# Normalizes DB and EXIF coords to the same precision so == comparisons work exactly.
# 10^7 = 7 decimal places (~1cm); fits in uint32 for both lat and lon.
GPS_COORD_PRECISION = 10_000_000

# Downloader / API constants
MAX_RESOLUTION = 2048
API_IMAGE_LIMIT = 2000
# Note: 40 and 40 workers were found to be empirically fastest without running into I/O limits
# Feel free to increment until whatever your system can handle  
DISCOVERY_WORKERS = 40
DOWNLOAD_WORKERS = 40
DB_COMMIT_BATCH = 50
DISCOVERY_STALENESS_DAYS = 21
GRANULARITY_MIN = 1
GRANULARITY_MAX = 100
GRANULARITY_DEFAULT = 25

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

@dataclass
class GridParams:
    """Grid cell sizes derived from a granularity level."""
    grid_cell_size: float
    min_cell_size: float


def granularity_to_grid_params(level: int) -> GridParams:
    """Convert a 1–100 granularity level to grid and min cell sizes (log scale)."""
    t = (level - GRANULARITY_MIN) / (GRANULARITY_MAX - GRANULARITY_MIN)
    grid = 0.5 * math.pow(0.0004, t)
    min_cell = 0.25 * math.pow(0.0008, t)
    return GridParams(grid_cell_size=round(grid, 6), min_cell_size=round(min_cell, 6))

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
