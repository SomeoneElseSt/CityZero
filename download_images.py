#!/usr/bin/env python3
"""CLI tool to download street-level imagery from Mapillary for any city.

Usage:
    # San Francisco (default)
    python download_images.py
    
    # Custom city by name
    python download_images.py --city "New York"
    
    # Custom bounding box
    python download_images.py --bbox "-122.52,37.70,-122.35,37.83"
    
    # With image limit (for testing)
    python download_images.py --city "San Francisco" --limit 100
    
    # Resume interrupted download
    python download_images.py  # Just run again, it auto-resumes
"""

import argparse
import sys
from pathlib import Path

from cityzero.config import get_mapillary_config, BoundingBox, RAW_DATA_DIR
from cityzero.downloader import ImageDownloader
from cityzero.mapillary_client import MapillaryClient


# Predefined city bounding boxes (can be extended)
CITY_BBOXES = {
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


def parse_bbox_string(bbox_str: str) -> BoundingBox:
    """Parse bounding box from comma-separated string.
    
    Args:
        bbox_str: String in format "west,south,east,north"
        
    Returns:
        BoundingBox object
    """
    try:
        parts = [float(x.strip()) for x in bbox_str.split(',')]
        if len(parts) != 4:
            raise ValueError("Bbox must have exactly 4 values")
        
        return BoundingBox(
            west=parts[0],
            south=parts[1],
            east=parts[2],
            north=parts[3]
        )
    except Exception as e:
        print(f"❌ Error parsing bbox: {e}")
        print("   Format should be: west,south,east,north")
        print("   Example: -122.52,37.70,-122.35,37.83")
        sys.exit(1)


def get_bbox_for_city(city_name: str) -> BoundingBox:
    """Get bounding box for a known city.
    
    Args:
        city_name: Name of the city
        
    Returns:
        BoundingBox object
    """
    city_lower = city_name.lower()
    
    if city_lower in CITY_BBOXES:
        return CITY_BBOXES[city_lower]
    
    print(f"\n⚠️  City '{city_name}' not found in predefined list.")
    print("\nAvailable cities:")
    for city in sorted(CITY_BBOXES.keys()):
        print(f"  - {city.title()}")
    print("\nPlease use --bbox to specify custom coordinates.")
    sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download street-level imagery from Mapillary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download San Francisco (default):
    python download_images.py
    
  Download a specific city:
    python download_images.py --city "New York"
    
  Download with custom bounding box:
    python download_images.py --bbox "-74.05,40.68,-73.91,40.88"
    
  Limit download for testing:
    python download_images.py --city "San Francisco" --limit 50
    
  Specify output directory:
    python download_images.py --output-dir data/sf_images
        """
    )
    
    parser.add_argument(
        '--city',
        type=str,
        default='San Francisco',
        help='City name (default: San Francisco)'
    )
    
    parser.add_argument(
        '--bbox',
        type=str,
        help='Custom bounding box as "west,south,east,north" (overrides --city)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of images to download (useful for testing)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=RAW_DATA_DIR,
        help=f'Output directory for images (default: {RAW_DATA_DIR})'
    )
    
    parser.add_argument(
        '--list-cities',
        action='store_true',
        help='List available predefined cities and exit'
    )
    
    args = parser.parse_args()
    
    # Handle --list-cities
    if args.list_cities:
        print("\n📍 Available cities:")
        print("="*50)
        for city in sorted(CITY_BBOXES.keys()):
            bbox = CITY_BBOXES[city]
            print(f"  {city.title():20} {bbox.to_tuple()}")
        print("\nUse --city to select a city, or --bbox for custom coordinates")
        return
    
    # Determine bounding box
    if args.bbox:
        print(f"\n📍 Using custom bounding box")
        bbox = parse_bbox_string(args.bbox)
        location_name = "Custom Area"
    else:
        print(f"\n📍 Location: {args.city}")
        bbox = get_bbox_for_city(args.city)
        location_name = args.city
    
    # Initialize Mapillary client
    try:
        config = get_mapillary_config()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\nPlease ensure:")
        print("1. .env file exists in project root")
        print("2. MAPILLARY_CLIENT_TOKEN is set correctly")
        print("3. Token format: MLY|numeric_id|hex_string")
        sys.exit(1)
    
    client = MapillaryClient(config)
    
    # Initialize downloader
    downloader = ImageDownloader(client, output_dir=args.output_dir)
    
    # Download images
    try:
        stats = downloader.download_images(
            bbox=bbox,
            max_images=args.limit
        )
        
        # Exit with appropriate code
        if stats['failed'] > 0:
            sys.exit(1)  # Some failures occurred
        else:
            sys.exit(0)  # Success
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        print("Run the same command again to resume from where you left off.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
