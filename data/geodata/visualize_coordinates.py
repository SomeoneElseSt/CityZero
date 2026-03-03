"""
Visualize coordinates from downloads_coordinates.json
Creates a scatter plot
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Constants
INPUT_FILE = "downloads_coordinates.json"
SCATTER_OUTPUT = "coordinates_scatter.png"

def load_coordinates(filepath: str) -> tuple[list[float], list[float]]:
    """Load coordinates from JSON file"""
    print(f"Loading coordinates from {filepath}...")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    downloaded_ids = data.get('downloaded_ids', {})
    total = len(downloaded_ids)
    print(f"Found {total:,} coordinate points")
    
    lats = []
    lons = []
    
    for coord_data in downloaded_ids.values():
        lat = float(coord_data['lat'])
        lon = float(coord_data['lon'])
        lats.append(lat)
        lons.append(lon)
    
    return lats, lons

def create_scatter_plot(lats: list[float], lons: list[float], output_file: str):
    """Create a scatter plot of all coordinates"""
    print(f"\nCreating scatter plot with {len(lats):,} points...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    plt.figure(figsize=(12, 10))
    plt.scatter(lons, lats, alpha=0.3, s=1, c='blue')
    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.title(f'Image Coordinates Distribution ({len(lats):,} points)\n{timestamp}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to: {output_file}")
    plt.close()

def main():
    """Main execution function"""
    print("=" * 60)
    print("Coordinate Visualization Tool")
    print("=" * 60)
    
    # Load coordinates
    lats, lons = load_coordinates(INPUT_FILE)
    
    if not lats:
        print("Error: No coordinates found!")
        return
    
    # Create visualization
    create_scatter_plot(lats, lons, SCATTER_OUTPUT)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Scatter plot: {SCATTER_OUTPUT}")
    print("=" * 60)

if __name__ == "__main__":
    main()
