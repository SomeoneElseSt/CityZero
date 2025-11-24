"""Test script to verify Mapillary API setup."""

from cityzero.config import get_mapillary_config, get_sf_bbox
from cityzero.mapillary_client import MapillaryClient


def main():
    """Test Mapillary API connection."""
    print("=" * 60)
    print("CityZero - Mapillary API Test")
    print("=" * 60)
    
    # Get configuration
    print("\n1. Loading configuration...")
    try:
        config = get_mapillary_config()
        print("   ✓ Mapillary token loaded")
    except ValueError as e:
        print(f"   ✗ Error: {e}")
        print("\nPlease:")
        print("1. Copy .env.example to .env")
        print("2. Get your token from: https://www.mapillary.com/dashboard/developers")
        print("3. Add it to .env as MAPILLARY_CLIENT_TOKEN")
        return
    
    bbox = get_sf_bbox()
    print(f"   ✓ San Francisco bbox: {bbox.to_tuple()}")
    
    # Initialize client
    print("\n2. Initializing Mapillary client...")
    client = MapillaryClient(config)
    print("   ✓ Client initialized")
    
    # Get coverage statistics
    print("\n3. Fetching coverage statistics for San Francisco...")
    print("   (This may take a moment...)")
    stats = client.get_coverage_stats(bbox)
    
    print("\n" + "=" * 60)
    print("Coverage Statistics")
    print("=" * 60)
    print(f"Total images:      {stats['total_images']:,}")
    print(f"Panoramic images:  {stats['panoramic_images']:,}")
    print(f"Regular images:    {stats['regular_images']:,}")
    print(f"Unique sequences:  {stats['unique_sequences']:,}")
    print("\n✓ API connection successful!")
    print("\nNext steps:")
    print("- Review the coverage statistics")
    print("- Adjust bbox if needed in .env")
    print("- Ready to start downloading images")


if __name__ == "__main__":
    main()
