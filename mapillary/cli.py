#!/usr/bin/env python3
"""CLI tool to download street-level imagery from Mapillary for any city.

Usage:
    # Interactive mode (no arguments)
    uv run python3 cli.py

    # Non-interactive: specify city by name
    uv run python3 cli.py --city "New York"

    # Non-interactive: custom bounding box
    uv run python3 cli.py --bbox "-122.52,37.70,-122.35,37.83"

    # With image limit (for testing)
    uv run python3 cli.py --city "San Francisco" --limit 100

    # Resume without re-hitting API (default when images.db exists)
    uv run python3 cli.py --city "San Francisco" --state maintain

    # Re-discover and merge new images into existing DB
    uv run python3 cli.py --city "San Francisco" --state merge

    # Wipe DB and discover fresh
    uv run python3 cli.py --city "San Francisco" --state rediscover

    # Show available cities
    uv run python3 cli.py --list-cities
"""

import argparse
import atexit
import sys
import tempfile
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import folium
import folium.plugins
import questionary

from config import get_mapillary_config, BoundingBox, DATA_DIR, CITY_BBOXES
from client import MapillaryClient, ImageDownloader
from db import DiscoveryDB


DISCOVERY_STALENESS_DAYS = 21


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


def generate_map_preview(
    bbox: BoundingBox,
    location_name: str,
    heat_coords: list[list[float]] | None = None,
) -> str:
    """Generate an interactive folium map showing the bounding box and optional heat map.

    Args:
        bbox: Bounding box to visualize
        location_name: Name of the location for the map title
        heat_coords: Optional list of [lat, lon] pairs to render as heat map

    Returns:
        Path to the generated HTML file
    """
    center_lat = (bbox.south + bbox.north) / 2
    center_lon = (bbox.west + bbox.east) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap"
    )

    bbox_coords = [
        [bbox.south, bbox.west],
        [bbox.south, bbox.east],
        [bbox.north, bbox.east],
        [bbox.north, bbox.west],
        [bbox.south, bbox.west],
    ]

    folium.PolyLine(
        bbox_coords,
        color="red",
        weight=3,
        opacity=0.8,
        popup=f"Download Area: {location_name}"
    ).add_to(m)

    folium.Marker(
        location=[center_lat, center_lon],
        popup=f"Center of {location_name}",
        tooltip="Download area center"
    ).add_to(m)

    if heat_coords:
        folium.plugins.HeatMap(heat_coords, radius=8, blur=10, min_opacity=0.3).add_to(m)

    temp_file = Path(tempfile.gettempdir()) / "cityzero_preview.html"
    m.save(str(temp_file))
    atexit.register(lambda: temp_file.unlink(missing_ok=True))

    return str(temp_file)


def warn_if_stale(db: DiscoveryDB) -> None:
    last = db.get_last_discovered_at()
    if last is None:
        return
    age = datetime.now(timezone.utc) - last
    if age.days >= DISCOVERY_STALENESS_DAYS:
        print(f"\n⚠️  Discovery data is {age.days} days old (>{DISCOVERY_STALENESS_DAYS} days).")
        print("   Consider --state merge or --state rediscover to refresh.")


def prompt_discovery_state() -> str:
    print()
    state = questionary.select(
        "An existing database for this city was found. Discovery state?",
        choices=[
            questionary.Choice(title="Maintain: load from DB, skip API discovery", value="maintain"),
            questionary.Choice(title="Merge: re-discover and add new images to existing DB", value="merge"),
            questionary.Choice(title="Rediscover: wipe DB and run a full fresh discovery", value="rediscover"),
        ],
    ).ask()
    return state or "maintain"


def show_download_summary(
    downloader: ImageDownloader,
    bbox: BoundingBox,
    location_name: str,
    db: DiscoveryDB,
    state: str,
    save_to_db: bool,
    max_images: int = None,
) -> tuple[bool, list[dict]]:
    """Determine images to download and show summary before download.

    Args:
        state: 'maintain' | 'merge' | 'rediscover'
        save_to_db: Whether to persist discovered images to DB.

    Returns:
        (confirmed, pending_images)
    """
    print(f"\n📊 Analyzing {location_name}...")

    if state == "rediscover":
        db.wipe_images()

    if state in ("merge", "rediscover"):
        if save_to_db:
            db.set_meta("city", location_name)
            db.set_meta("bbox_west", str(bbox.west))
            db.set_meta("bbox_south", str(bbox.south))
            db.set_meta("bbox_east", str(bbox.east))
            db.set_meta("bbox_north", str(bbox.north))

        discovery_db = db if save_to_db else None
        downloader.discover_images(bbox, db=discovery_db)

        if save_to_db:
            db.set_meta("last_discovered_at", str(int(datetime.now(timezone.utc).timestamp())))

    pending = db.get_pending_images_metadata()

    if not pending:
        if db.get_image_count() > 0:
            print("✓ All images already downloaded!")
        else:
            print("❌ No images found in existing database. Consider running with --state rediscover.")
        return False, []

    if max_images and len(pending) > max_images:
        pending = pending[:max_images]

    heat_coords = [[img["lat"], img["lon"]] for img in pending]
    print(f"\n📍 Generating coverage map...")
    coverage_map = generate_map_preview(bbox, location_name, heat_coords)
    print(f"   Opening in browser: {coverage_map}")
    webbrowser.open(f"file://{coverage_map}")

    total = db.get_image_count()
    downloaded_count = total - db.get_pending_count()
    print("\n📋 Summary:")
    print(f"  {'Location:':<22} {location_name}")
    print(f"  {'Total found:':<22} {total:,}")
    print(f"  {'Already downloaded:':<22} {downloaded_count:,}")
    print(f"  {'New to download:':<22} {len(pending):,}")

    proceed = questionary.confirm(
        f"Download {len(pending):,} new images?",
        default=True,
    ).ask()

    return bool(proceed), pending


def interactive_mode() -> tuple[BoundingBox, str]:
    """Run interactive mode: prompt user to select city and show map preview.

    Returns:
        Tuple of (BoundingBox, location_name)
    """
    print("\n" + "="*70)
    print("🗺️ CityZero Image Downloader")
    print("="*70)

    city_choices = [city.title() for city in sorted(CITY_BBOXES.keys())]
    city_choices.append("Custom bounding box...")

    selected = questionary.select(
        "Select a city or custom area:",
        choices=city_choices
    ).ask()

    if selected is None:
        print("\n⚠️  No selection made. Exiting.")
        sys.exit(0)

    if selected == "Custom bounding box...":
        bbox_str = questionary.text(
            "Enter bounding box (west,south,east,north):",
            default="-122.52,37.70,-122.35,37.83"
        ).ask()

        if bbox_str is None:
            print("\n⚠️  No input provided. Exiting.")
            sys.exit(0)

        bbox = BoundingBox.from_string(bbox_str)
        if bbox is None:
            print(f"Invalid bbox format: '{bbox_str}'. Expected: west,south,east,north")
            sys.exit(1)
        location_name = "Custom Area"
    else:
        location_name = selected
        bbox = get_bbox_for_city(selected)

    print(f"\n📍 Generating map preview for {location_name}...")
    map_file = generate_map_preview(bbox, location_name)
    print(f"   Opening in browser: {map_file}")
    webbrowser.open(f"file://{map_file}")

    return bbox, location_name


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download street-level imagery from Mapillary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode (recommended):
    uv run python3 cli.py

  Non-interactive: specify city by name:
    uv run python3 cli.py --city "New York"

  Non-interactive: custom bounding box:
    uv run python3 cli.py --bbox "-74.05,40.68,-73.91,40.88"

  Limit download for testing:
    uv run python3 cli.py --city "San Francisco" --limit 50

  Specify output directory:
    uv run python3 cli.py --output-dir data/sf_images

  Show available cities:
    uv run python3 cli.py --list-cities
        """
    )

    parser.add_argument('--city', type=str, help='City name (enables non-interactive mode)')
    parser.add_argument('--bbox', type=str, help='Custom bounding box as "west,south,east,north" (overrides --city)')
    parser.add_argument('--limit', type=int, help='Maximum number of images to download (useful for testing)')
    parser.add_argument('--output-dir', type=Path, default=None, help=f'Output directory for images (default: {DATA_DIR}/<city>)')
    parser.add_argument('--list-cities', action='store_true', help='List available predefined cities and exit')
    parser.add_argument('--preview', action='store_true', help='Show map preview before downloading (non-interactive mode only)')
    parser.add_argument(
        '--state',
        choices=['maintain', 'merge', 'rediscover'],
        default=None,
        help='Discovery state when images.db exists: maintain (default) | merge | rediscover',
    )
    parser.add_argument(
        '--no-save-discovery',
        action='store_true',
        help='Skip saving discovered image IDs to images.db (headless only)',
    )

    args = parser.parse_args()

    if args.list_cities:
        print("\n📍 Available cities:")
        for city in sorted(CITY_BBOXES.keys()):
            bbox = CITY_BBOXES[city]
            print(f"  {city.title():20} {bbox.to_tuple()}")
        return

    is_interactive = not (args.city or args.bbox)

    if is_interactive:
        bbox, location_name = interactive_mode()
    elif args.bbox:
        print(f"\n📍 Using custom bounding box")
        bbox = BoundingBox.from_string(args.bbox)
        if bbox is None:
            print(f"Invalid bbox format: '{args.bbox}'. Expected: west,south,east,north")
            print("   Example: -122.52,37.70,-122.35,37.83")
            sys.exit(1)
        location_name = "Custom Area"
    else:
        print(f"\n📍 Location: {args.city}")
        bbox = get_bbox_for_city(args.city)
        location_name = args.city

    if not is_interactive and args.preview:
        print(f"\n📍 Generating map preview...")
        map_file = generate_map_preview(bbox, location_name)
        print(f"   Opening in browser: {map_file}")
        webbrowser.open(f"file://{map_file}")
        input("\nPress Enter to continue...")

    if args.output_dir is None:
        if location_name == "Custom Area":
            args.output_dir = DATA_DIR
        else:
            normalized = location_name.lower().replace(" ", "_")
            args.output_dir = DATA_DIR / normalized

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {args.output_dir}")

    config = get_mapillary_config()
    if config is None:
        print("\n❌ MAPILLARY_CLIENT_TOKEN not set.")
        print("\nPlease ensure:")
        print("1. .env file exists in project root")
        print("2. MAPILLARY_CLIENT_TOKEN is set correctly")
        print("3. Token format: MLY|numeric_id|hex_string")
        sys.exit(1)

    client = MapillaryClient(config)
    downloader = ImageDownloader(client, output_dir=args.output_dir / "images")
    db = DiscoveryDB.get(args.output_dir / "images.db")

    db_has_data = db.get_image_count() > 0
    if db_has_data:
        warn_if_stale(db)
        if is_interactive:
            state = prompt_discovery_state()
        else:
            state = args.state or "maintain"
        save_to_db = True
    else:
        state = "rediscover"
        save_to_db = not args.no_save_discovery
        if is_interactive:
            proceed = questionary.confirm("Proceed with discovery?", default=True).ask()
            if not proceed:
                print("\n⚠️  Discovery cancelled by user..")
                sys.exit(0)

    confirmed, pending_images = show_download_summary(
        downloader, bbox, location_name, db, state, save_to_db, args.limit
    )
    if not confirmed:
        print("\n⚠️  Download cancelled by user.")
        sys.exit(0)

    try:
        stats = downloader.download_images(
            bbox=bbox, db=db, max_images=args.limit, images=pending_images
        )

        sys.exit(1 if stats['failed'] > 0 else 0)

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
