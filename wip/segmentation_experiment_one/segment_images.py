import json
import os
import shutil
import random
from pathlib import Path

# Constants
GEODATA_DIR = Path("data/geodata")
RAW_IMAGES_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/geodata/image_boxes")
METADATA_FILE = GEODATA_DIR / "download_metadata.json"
SEGMENTS_FILE = GEODATA_DIR / "findi_segments.json"

BOX_NAME_MAPPING = {
    "northwest": "nw",
    "northeast": "ne",
    "southwest": "sw",
    "southeast": "se"
}

CENTER_TOLERANCE = 0.0001  # Tolerance for considering a point as "at center"


def load_json_file(file_path):
    """Load and return JSON data from file."""
    with open(file_path, "r") as f:
        return json.load(f)


def get_box_corners(box_data):
    """Extract corner coordinates from box data as list of [lat, lon] tuples."""
    corners = []
    order = ["nw", "ne", "se", "sw"]
    for corner_key in order:
        corner = box_data.get(corner_key, {})
        lat = float(corner.get("lat", 0))
        lon = float(corner.get("lon", 0))
        corners.append([lat, lon])
    return corners


def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm.
    Returns True if point is inside polygon, False otherwise.
    """
    lat, lon = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def is_at_center(point, center_point, tolerance=CENTER_TOLERANCE):
    """Check if a point is at the center (within tolerance)."""
    lat, lon = point
    center_lat, center_lon = center_point
    lat_diff = abs(lat - center_lat)
    lon_diff = abs(lon - center_lon)
    return lat_diff < tolerance and lon_diff < tolerance


def get_center_point(segments_data):
    """Calculate the center point where all boxes meet."""
    # The center is the se corner of northwest, ne corner of southwest,
    # sw corner of northeast, and nw corner of southeast
    # They should all be the same point (Lotta's Fountain)
    northwest_se = segments_data["boxes"]["northwest"]["se"]
    center_lat = float(northwest_se["lat"])
    center_lon = float(northwest_se["lon"])
    return [center_lat, center_lon]


def assign_image_to_box(image_lat, image_lon, boxes_data, center_point, box_names):
    """
    Determine which box an image belongs to.
    Returns box name (nw, ne, sw, se) or None if not in any box.
    """
    point = [float(image_lat), float(image_lon)]
    
    # Check if point is at center - randomize assignment
    if is_at_center(point, center_point):
        return random.choice(box_names)
    
    # Check each box
    for box_name, box_data in boxes_data.items():
        corners = get_box_corners(box_data)
        if point_in_polygon(point, corners):
            return BOX_NAME_MAPPING[box_name]
    
    return None


def create_box_folder(box_code):
    """Create folder for a box and return its path."""
    box_folder = OUTPUT_DIR / box_code
    box_folder.mkdir(parents=True, exist_ok=True)
    return box_folder


def create_readme(box_folder, box_code, full_box_name, image_count):
    """Create README.md file in box folder."""
    readme_path = box_folder / "README.md"
    content = f"""# {box_code.upper()} Box

This folder contains images that belong to the {full_box_name} segment.

Images in this folder were filtered based on their GPS coordinates falling within
the bounding box defined for the {full_box_name} region.

**Total images in this box: {image_count}**
"""
    with open(readme_path, "w") as f:
        f.write(content)


def copy_image_to_box(image_id, box_folder):
    """Copy image file from raw directory to box folder."""
    source_file = RAW_IMAGES_DIR / f"{image_id}.jpg"
    if not source_file.exists():
        print(f"Warning: Image {image_id} not found in {RAW_IMAGES_DIR}")
        return False
    
    dest_file = box_folder / f"{image_id}.jpg"
    shutil.copy2(source_file, dest_file)
    return True


def main():
    """Main function to organize images into boxes."""
    print("Loading metadata and segment data...")
    metadata = load_json_file(METADATA_FILE)
    segments_data = load_json_file(SEGMENTS_FILE)
    
    print(f"Found {metadata.get('total_downloaded', 0)} downloaded images in metadata")
    
    # Get center point
    center_point = get_center_point(segments_data)
    print(f"Center point: {center_point[0]}, {center_point[1]}")
    
    # Get box names
    boxes_data = segments_data["boxes"]
    box_names = list(BOX_NAME_MAPPING.values())
    
    # Initialize counters
    box_counts = {box: 0 for box in box_names}
    center_count = 0
    unassigned_count = 0
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create box folders
    print("\nCreating box folders...")
    for full_name, box_code in BOX_NAME_MAPPING.items():
        box_folder = create_box_folder(box_code)
        print(f"Created folder: {box_folder}")
    
    # Process each image
    print("\nProcessing images...")
    downloaded_ids = metadata.get("downloaded_ids", {})
    total_images = len(downloaded_ids)
    
    for idx, (image_id, image_data) in enumerate(downloaded_ids.items(), 1):
        if idx % 2000 == 0:
            print(f"Processed {idx}/{total_images} images...")
        
        lat = image_data.get("lat")
        lon = image_data.get("lon")
        
        if not lat or not lon:
            unassigned_count += 1
            continue
        
        # Assign to box
        box_code = assign_image_to_box(lat, lon, boxes_data, center_point, box_names)
        
        if not box_code:
            unassigned_count += 1
            continue
        
        # Check if at center for logging
        point = [float(lat), float(lon)]
        if is_at_center(point, center_point):
            center_count += 1
        
        # Copy image to box folder
        box_folder = OUTPUT_DIR / box_code
        if copy_image_to_box(image_id, box_folder):
            box_counts[box_code] += 1
    
    # Create/update READMEs with final counts
    print("\nCreating README files with final counts...")
    for full_name, box_code in BOX_NAME_MAPPING.items():
        box_folder = OUTPUT_DIR / box_code
        create_readme(box_folder, box_code, full_name, box_counts[box_code])
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total images processed: {total_images}")
    print(f"\nImages assigned to boxes:")
    for box_code in box_names:
        print(f"  {box_code.upper()}: {box_counts[box_code]}")
    print(f"\nImages at center (randomly assigned): {center_count}")
    print(f"Unassigned images: {unassigned_count}")
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
