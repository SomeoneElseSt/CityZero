"""
Custom spatial matcher. Outputs list of images to match as pairs. 

Ingests:
1. .txt file with images to match Spatially. 
2. .json file with lat/lon coords for each image ID in the above. 

Will then, for each image, find the n closest images (default n=50) within max_distance (default 100m), 
and write them as pairs to an output .txt file in COLMAP format.

"""

import argparse
import json
import math
import sys
from collections import defaultdict

EARTH_RADIUS_METERS = 6371000


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two GPS points."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * EARTH_RADIUS_METERS * math.atan2(math.sqrt(a), math.sqrt(1-a))


def generate_spatial_pairs(image_list_path, gps_json_path, output_path, max_distance, max_neighbors):
    """Generate spatial pairs and write to output file."""
    
    if image_list_path is None or not image_list_path:
        print("Image list file path is None. Please provide a valid image list path")
        sys.exit(1)
    
    if ".txt" not in str(image_list_path):
        print("You must provide a path + .txt file for the image list path -- you did not provide a file at the end of the path -> image_list.txt")
        sys.exit(1)
    
    if gps_json_path is None or not gps_json_path:
        print("GPS JSON file path is None. Please provide a valid GPS JSON path")
        sys.exit(1)
    
    if ".json" not in str(gps_json_path):
        print("You must provide a path + .json file for the GPS JSON path -- you did not provide a file at the end of the path -> gps_data.json")
        sys.exit(1)
    
    if output_path is None or not output_path:
        print("Output file path is None. Please provide a valid output path")
        sys.exit(1)
    
    if ".txt" not in str(output_path):
        print("You must provide a path + .txt file for the output path -- you did not provide a file at the end of the path -> pairs.txt")
        sys.exit(1)
    
    # Read image IDs from file
    try:
        with open(image_list_path, 'r') as f:
            image_ids = [line.strip().replace('.jpg', '') for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Image list file not found: {image_list_path}")
        sys.exit(1)
    
    if not image_ids:
        print("No image IDs found in image list file")
        sys.exit(1)
    
    # Load GPS data
    try:
        with open(gps_json_path, 'r') as f:
            gps_data = json.load(f)['downloaded_ids']
    except FileNotFoundError:
        print(f"GPS JSON file not found: {gps_json_path}")
        sys.exit(1)
    except KeyError:
        print("GPS JSON file missing 'downloaded_ids' key")
        sys.exit(1)
    
    # Filter GPS for images in the list
    box_gps = {}
    missing_gps = []
    for img_id in image_ids:
        if img_id in gps_data:
            box_gps[img_id] = gps_data[img_id]
        else:
            missing_gps.append(img_id)
    
    if missing_gps:
        print(f"Warning: {len(missing_gps)} images missing GPS data, skipping them")
    
    if len(box_gps) < 2:
        print("Not enough images with GPS data to generate pairs (need at least 2)")
        sys.exit(1)
    
    # For each image, find all neighbors within max_distance
    image_neighbors = defaultdict(list)
    image_list = list(box_gps.keys())
    
    for i in range(len(image_list)):
        img1 = image_list[i]
        coords1 = box_gps[img1]
        lat1 = float(coords1['lat'])
        lon1 = float(coords1['lon'])
        
        for j in range(i + 1, len(image_list)):
            img2 = image_list[j]
            coords2 = box_gps[img2]
            lat2 = float(coords2['lat'])
            lon2 = float(coords2['lon'])
            
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            
            if dist <= max_distance:
                image_neighbors[img1].append((img2, dist))
                image_neighbors[img2].append((img1, dist))
    
    # Limit to max_neighbors per image (keep closest ones)
    pairs = []
    for img_id, neighbors in image_neighbors.items():
        neighbors.sort(key=lambda x: x[1])
        limited_neighbors = neighbors[:max_neighbors]
        
        for neighbor_id, _ in limited_neighbors:
            # Ensure consistent ordering to avoid duplicates
            if img_id < neighbor_id:
                pairs.append((f"{img_id}.jpg", f"{neighbor_id}.jpg"))
            else:
                pairs.append((f"{neighbor_id}.jpg", f"{img_id}.jpg"))
    
    # Remove duplicate pairs
    pairs = list(set(pairs))
    
    # Write pairs to output file
    try:
        with open(output_path, 'w') as f:
            for img1, img2 in pairs:
                f.write(f"{img1} {img2}\n")
    except IOError as e:
        print(f"Error writing to output file: {e}")
        sys.exit(1)
    
    print(f"Generated {len(pairs)} image pairs and wrote to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image pairs based on GPS proximity for spatial matching")
    parser.add_argument(
        "--image_list_path",
        type=str,
        required=True,
        help="Path to .txt file containing image IDs (one per line)"
    )
    parser.add_argument(
        "--gps_json_path",
        type=str,
        required=True,
        help="Path to .json file containing GPS coordinates with 'downloaded_ids' key"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output .txt file for image pairs (COLMAP format)"
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=100.0,
        help="Maximum distance in meters between images to consider as pairs (default: 100.0)"
    )
    parser.add_argument(
        "--max_neighbors",
        type=int,
        default=50,
        help="Maximum number of neighbors per image (default: 50)"
    )
    
    args = parser.parse_args()
    
    generate_spatial_pairs(
        args.image_list_path,
        args.gps_json_path,
        args.output_path,
        args.max_distance,
        args.max_neighbors
    )
