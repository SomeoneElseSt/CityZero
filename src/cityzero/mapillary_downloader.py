#!/usr/bin/env python3
"""CLI tool to download street-level imagery from Mapillary for any city.

Usage:
    # Interactive mode (no arguments)
    uv run python -m cityzero.mapillary_downloader

    # Non-interactive: specify city by name
    uv run python -m cityzero.mapillary_downloader --city "New York"

    # Non-interactive: custom bounding box
    uv run python -m cityzero.mapillary_downloader --bbox "-122.52,37.70,-122.35,37.83"

    # With image limit (for testing)
    uv run python -m cityzero.mapillary_downloader --city "San Francisco" --limit 100

    # Resume interrupted download
    uv run python -m cityzero.mapillary_downloader --city "San Francisco"  # auto-resumes

    # Show available cities
    uv run python -m cityzero.mapillary_downloader --list-cities
"""

import argparse
import sys
import tempfile
import webbrowser
from pathlib import Path

import folium
import questionary

from .config import get_mapillary_config, BoundingBox, RAW_DATA_DIR, CITY_BBOXES
from .downloader import ImageDownloader
from .mapillary_client import MapillaryClient


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


def generate_map_preview(bbox: BoundingBox, location_name: str) -> str:
    """Generate an interactive folium map showing the bounding box.

    Args:
        bbox: Bounding box to visualize
        location_name: Name of the location for the map title

    Returns:
        Path to the generated HTML file
    """
    # Calculate center of bbox
    center_lat = (bbox.south + bbox.north) / 2
    center_lon = (bbox.west + bbox.east) / 2

    # Create map centered on bbox
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    # Draw bbox as a red rectangle
    bbox_coords = [
        [bbox.south, bbox.west],
        [bbox.south, bbox.east],
        [bbox.north, bbox.east],
        [bbox.north, bbox.west],
        [bbox.south, bbox.west],  # Close the rectangle
    ]

    folium.PolyLine(
        bbox_coords,
        color="red",
        weight=3,
        opacity=0.8,
        popup=f"Download Area: {location_name}"
    ).add_to(m)

    # Add a marker at the center
    folium.Marker(
        location=[center_lat, center_lon],
        popup=f"Center of {location_name}",
        tooltip="Download area center"
    ).add_to(m)

    # Save to temp file
    temp_file = Path(tempfile.gettempdir()) / "cityzero_preview.html"
    m.save(str(temp_file))

    return str(temp_file)


def show_download_summary(
    downloader: ImageDownloader,
    bbox: BoundingBox,
    location_name: str,
    max_images: int = None
) -> bool:
    """Discover images and show summary before download.

    Args:
        downloader: ImageDownloader instance
        bbox: Bounding box to search
        location_name: Name of the location
        max_images: Optional limit on images to download

    Returns:
        True if user confirms, False if cancelled
    """
    print(f"\n📊 Analyzing {location_name}...")

    # Get cached images
    cached_ids = downloader.get_downloaded_image_ids()

    # Discover available images
    all_images = downloader.discover_images(bbox)

    if not all_images:
        print("❌ No images found in this area")
        return False

    # Calculate new images to download
    images_to_download = [
        img for img in all_images
        if img.get('id') not in cached_ids
    ]

    # Apply max_images limit
    if max_images and len(images_to_download) > max_images:
        images_to_download = images_to_download[:max_images]

    # Show summary
    print("\n📋 Summary:")
    print(f"  Location:        {location_name}")
    print(f"  Total found:     {len(all_images):,}")
    print(f"  Already cached:  {len(cached_ids):,}")
    print(f"  New to download: {len(images_to_download):,}")

    if len(images_to_download) == 0:
        print("\n✓ All images already downloaded!")
        return False

    # Final confirmation
    proceed = questionary.confirm(
        f"Download {len(images_to_download):,} new images?",
        default=True
    ).ask()

    return proceed if proceed is not None else False


def interactive_mode() -> tuple[BoundingBox, str]:
    """Run interactive mode: prompt user to select city and show map preview.

    Returns:
        Tuple of (BoundingBox, location_name)
    """
    print("\n" + "="*70)
    print("🗺️ CityZero Image Downloader")
    print("="*70)

    # Prepare city choices (capitalize city names nicely)
    city_choices = [city.title() for city in sorted(CITY_BBOXES.keys())]
    city_choices.append("Custom bounding box...")

    # Interactive selection
    selected = questionary.select(
        "Select a city or custom area:",
        choices=city_choices
    ).ask()

    if selected is None:
        print("\n⚠️  No selection made. Exiting.")
        sys.exit(0)

    if selected == "Custom bounding box...":
        # Prompt for custom bbox
        bbox_str = questionary.text(
            "Enter bounding box (west,south,east,north):",
            default="-122.52,37.70,-122.35,37.83"
        ).ask()

        if bbox_str is None:
            print("\n⚠️  No input provided. Exiting.")
            sys.exit(0)

        bbox = parse_bbox_string(bbox_str)
        location_name = "Custom Area"
    else:
        # Selected a predefined city
        location_name = selected
        bbox = get_bbox_for_city(selected)

    # Show map preview
    print(f"\n📍 Generating map preview for {location_name}...")
    map_file = generate_map_preview(bbox, location_name)
    print(f"   Opening in browser: {map_file}")
    webbrowser.open(f"file://{map_file}")

    # Confirmation prompt
    proceed = questionary.confirm(
        "Proceed with preview?",
        default=True
    ).ask()

    if not proceed:
        print("\n⚠️  Download cancelled by user.")
        sys.exit(0)

    return bbox, location_name


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download street-level imagery from Mapillary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode (recommended):
    uv run python -m cityzero.mapillary_downloader

  Non-interactive: specify city by name:
    uv run python -m cityzero.mapillary_downloader --city "New York"

  Non-interactive: custom bounding box:
    uv run python -m cityzero.mapillary_downloader --bbox "-74.05,40.68,-73.91,40.88"

  Limit download for testing:
    uv run python -m cityzero.mapillary_downloader --city "San Francisco" --limit 50

  Specify output directory:
    uv run python -m cityzero.mapillary_downloader --output-dir data/sf_images

  Show available cities:
    uv run python -m cityzero.mapillary_downloader --list-cities
        """
    )

    parser.add_argument(
        '--city',
        type=str,
        help='City name (enables non-interactive mode)'
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
        default=None,
        help=f'Output directory for images (default: {RAW_DATA_DIR}/<city>)'
    )

    parser.add_argument(
        '--list-cities',
        action='store_true',
        help='List available predefined cities and exit'
    )

    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show map preview before downloading (non-interactive mode only)'
    )

    args = parser.parse_args()

    # Handle --list-cities
    if args.list_cities:
        print("\n📍 Available cities:")
        for city in sorted(CITY_BBOXES.keys()):
            bbox = CITY_BBOXES[city]
            print(f"  {city.title():20} {bbox.to_tuple()}")
        return

    # Determine if interactive or non-interactive mode
    if args.city or args.bbox:
        # Non-interactive mode (flag-based)
        if args.bbox:
            print(f"\n📍 Using custom bounding box")
            bbox = parse_bbox_string(args.bbox)
            location_name = "Custom Area"
        else:
            print(f"\n📍 Location: {args.city}")
            bbox = get_bbox_for_city(args.city)
            location_name = args.city

        # Show preview if requested
        if args.preview:
            print(f"\n📍 Generating map preview...")
            map_file = generate_map_preview(bbox, location_name)
            print(f"   Opening in browser: {map_file}")
            webbrowser.open(f"file://{map_file}")
            input("\nPress Enter to continue...")
    else:
        # Interactive mode (default when no flags)
        bbox, location_name = interactive_mode()

    # Determine output directory
    if args.output_dir is None:
        # Use default with city subdir (unless custom area)
        if location_name == "Custom Area":
            args.output_dir = RAW_DATA_DIR
        else:
            args.output_dir = RAW_DATA_DIR / location_name
    # else: user explicitly specified --output-dir, use it as-is

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {args.output_dir}")

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

    # Show summary and get final confirmation
    if not show_download_summary(downloader, bbox, location_name, args.limit):
        print("\n⚠️  Download cancelled by user.")
        sys.exit(0)

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
