"""
Generate fringe boundary zones between adjacent spatial boxes and create
exhaustive image pair lists for COLMAP matches_importer.

This script:
1. Ingests box definitions (N boxes with lat/lon corners)
2. Ingests GPS coordinates for all images
3. Detects shared borders between adjacent boxes
4. Creates fringe buffer zones (default 50m wide) along each border
5. Selects images within each fringe zone
6. Generates exhaustive pair lists with canonical ordering
7. Outputs pair files ready for COLMAP matches_importer
8. Exports fringe geometries and visualization

Flags:
    --box_coords
        Path to JSON file with box corner coordinates

    --gps_coords
        Path to JSON file mapping image_id → {lat, lon}

    --fringe_length
        Total width of fringe zone in meters (default: 50.0)

    --output_path
        Directory path for output pair files

"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

from shapely.geometry import Polygon, Point, LineString
from shapely.ops import transform
import pyproj
from pyproj import Transformer
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

# Constants
DEFAULT_FRINGE_LENGTH_METERS = 50
METERS_PER_SIDE = DEFAULT_FRINGE_LENGTH_METERS / 2


@dataclass
class Box:
    name: str
    polygon: Polygon
    
    
@dataclass  
class Border:
    box_a: str
    box_b: str
    line: LineString


@dataclass
class FringeZone:
    border_name: str
    polygon: Polygon
    images: List[str]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fringe boundary pair lists for COLMAP matching"
    )
    parser.add_argument(
        "--box_coords",
        type=str,
        required=True,
        help="Path to JSON file with box corner coordinates"
    )
    parser.add_argument(
        "--gps_coords", 
        type=str,
        required=True,
        help="Path to JSON file mapping image_id → {lat, lon}"
    )
    parser.add_argument(
        "--fringe_length",
        type=float,
        default=DEFAULT_FRINGE_LENGTH_METERS,
        help=f"Total width of fringe zone in meters (default: {DEFAULT_FRINGE_LENGTH_METERS})"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory path for output pair files"
    )
    
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> bool:
    """Validate input arguments and file existence."""
    
    box_coords_path = Path(args.box_coords)
    if not box_coords_path.exists():
        print(f"Error: Box coordinates file not found: {args.box_coords}")
        return False
        
    if not str(box_coords_path).endswith('.json'):
        print("Error: --box_coords must be a .json file")
        return False
    
    gps_coords_path = Path(args.gps_coords)
    if not gps_coords_path.exists():
        print(f"Error: GPS coordinates file not found: {args.gps_coords}")
        return False
        
    if not str(gps_coords_path).endswith('.json'):
        print("Error: --gps_coords must be a .json file")
        return False
    
    if args.fringe_length <= 0:
        print(f"Error: --fringe_length must be positive, got {args.fringe_length}")
        return False
    
    output_path = Path(args.output_path)
    if not output_path.exists():
        print(f"Creating output directory: {args.output_path}")
        output_path.mkdir(parents=True, exist_ok=True)
    
    return True


def load_box_definitions(box_coords_path: str) -> List[Box]:
    """Load and parse box definitions from JSON."""
    
    with open(box_coords_path, 'r') as f:
        data = json.load(f)
    
    if 'boxes' not in data:
        print("Error: JSON must contain 'boxes' key")
        sys.exit(1)
    
    boxes = []
    for box_name, corners in data['boxes'].items():
        
        if not all(k in corners for k in ['nw', 'ne', 'sw', 'se']):
            print(f"Error: Box '{box_name}' missing corner definitions")
            sys.exit(1)
        
        coords = [
            (float(corners['nw']['lon']), float(corners['nw']['lat'])),
            (float(corners['ne']['lon']), float(corners['ne']['lat'])),
            (float(corners['se']['lon']), float(corners['se']['lat'])),
            (float(corners['sw']['lon']), float(corners['sw']['lat'])),
        ]
        
        polygon = Polygon(coords)
        boxes.append(Box(name=box_name, polygon=polygon))
    
    print(f"Loaded {len(boxes)} boxes: {[b.name for b in boxes]}")
    return boxes


def load_gps_coordinates(gps_coords_path: str) -> Dict[str, Tuple[float, float]]:
    """Load GPS coordinates mapping image_id → (lat, lon)."""
    
    with open(gps_coords_path, 'r') as f:
        data = json.load(f)
    
    if 'downloaded_ids' not in data:
        print("Error: JSON must contain 'downloaded_ids' key")
        sys.exit(1)
    
    gps_map = {}
    for image_id, coords in data['downloaded_ids'].items():
        
        if 'lat' not in coords or 'lon' not in coords:
            continue
            
        image_name = f"{image_id}.jpg"
        gps_map[image_name] = (float(coords['lat']), float(coords['lon']))
    
    print(f"Loaded GPS coordinates for {len(gps_map)} images")
    return gps_map


def detect_shared_borders(boxes: List[Box]) -> List[Border]:
    """Detect which boxes share continuous border edges."""
    
    borders = []
    
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            box_a = boxes[i]
            box_b = boxes[j]
            
            shared_edge = find_shared_edge(box_a.polygon, box_b.polygon)
            
            if shared_edge is None:
                continue
            
            borders.append(Border(
                box_a=box_a.name,
                box_b=box_b.name,
                line=shared_edge
            ))
    
    print(f"Detected {len(borders)} shared borders:")
    for border in borders:
        print(f"  {border.box_a} ↔ {border.box_b}")
    
    return borders


def find_shared_edge(poly_a: Polygon, poly_b: Polygon) -> LineString:
    """Find shared edge between two polygons, return None if only corner touch."""
    
    intersection = poly_a.intersection(poly_b)
    
    if intersection.is_empty:
        return None
    
    if intersection.geom_type == 'Point':
        return None
    
    if intersection.geom_type == 'LineString':
        if intersection.length < 1e-6:
            return None
        return intersection
    
    return None


def create_fringe_polygon(border: Border, buffer_distance_meters: float) -> Polygon:
    """Create geodesic buffer around border line."""
    
    centroid = border.line.centroid
    center_lat = centroid.y
    center_lon = centroid.x
    
    utm_zone = int((center_lon + 180) / 6) + 1
    utm_crs = f"+proj=utm +zone={utm_zone} +ellps=WGS84"
    
    wgs84_to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    utm_to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    
    line_utm = transform(wgs84_to_utm.transform, border.line)
    buffered_utm = line_utm.buffer(buffer_distance_meters)
    buffered_wgs84 = transform(utm_to_wgs84.transform, buffered_utm)
    
    return buffered_wgs84


def select_images_in_fringe(
    fringe_polygon: Polygon, 
    gps_map: Dict[str, Tuple[float, float]]
) -> List[str]:
    """Select all images whose GPS coordinates fall within fringe polygon."""
    
    images = []
    
    for image_name, (lat, lon) in gps_map.items():
        point = Point(lon, lat)
        
        if fringe_polygon.contains(point):
            images.append(image_name)
    
    return images


def generate_exhaustive_pairs(images: List[str]) -> List[Tuple[str, str]]:
    """Generate all unique pairs with canonical ordering."""
    
    images_sorted = sorted(images)
    pairs = []
    
    for i in range(len(images_sorted)):
        for j in range(i + 1, len(images_sorted)):
            pairs.append((images_sorted[i], images_sorted[j]))
    
    return pairs


def write_pair_file(pairs: List[Tuple[str, str]], output_path: Path) -> None:
    """Write pairs to text file, one pair per line."""
    
    with open(output_path, 'w') as f:
        for img_a, img_b in pairs:
            f.write(f"{img_a} {img_b}\n")


def extract_polygon_coordinates(polygon: Polygon) -> List[Dict[str, float]]:
    """Extract exterior coordinates from polygon as list of {lat, lon} dicts."""
    
    coords = []
    exterior_coords = list(polygon.exterior.coords)
    
    for lon, lat in exterior_coords:
        coords.append({"lat": lat, "lon": lon})
    
    return coords


def write_fringe_geometries(fringes: List[FringeZone], output_path: Path) -> None:
    """Write fringe polygon coordinates to JSON for visualization."""
    
    fringe_data = {
        "type": "FeatureCollection",
        "features": []
    }
    
    for fringe in fringes:
        coords = extract_polygon_coordinates(fringe.polygon)
        
        feature = {
            "type": "Feature",
            "properties": {
                "border": fringe.border_name,
                "num_images": len(fringe.images)
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": coords
            }
        }
        
        fringe_data["features"].append(feature)
    
    fringes_json_path = output_path / "fringes.json"
    with open(fringes_json_path, 'w') as f:
        json.dump(fringe_data, f, indent=2)
    
    print(f"Wrote fringe geometries to: {fringes_json_path}")


def visualize_boxes_and_fringes(
    boxes: List[Box],
    fringes: List[FringeZone],
    output_path: Path
) -> None:
    """Create visualization of boxes and fringe zones."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for box in boxes:
        coords = list(box.polygon.exterior.coords)
        polygon_patch = MplPolygon(
            coords,
            fill=False,
            edgecolor='blue',
            linewidth=2,
            label=f'Box: {box.name}' if boxes.index(box) == 0 else None
        )
        ax.add_patch(polygon_patch)
        
        centroid = box.polygon.centroid
        ax.text(
            centroid.x,
            centroid.y,
            box.name,
            fontsize=10,
            ha='center',
            va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    
    for fringe in fringes:
        coords = list(fringe.polygon.exterior.coords)
        polygon_patch = MplPolygon(
            coords,
            fill=True,
            facecolor='red',
            alpha=0.3,
            edgecolor='red',
            linewidth=1,
            label='Fringe zones' if fringes.index(fringe) == 0 else None
        )
        ax.add_patch(polygon_patch)
    
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Boxes and Fringe Boundary Zones')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    viz_path = output_path / "boxes_and_fringes_.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {viz_path}")


def write_metadata(
    fringes: List[FringeZone],
    output_path: Path,
    fringe_length: float
) -> None:
    """Write fringe metadata to JSON."""
    
    metadata = {
        "fringe_length_meters": fringe_length,
        "total_fringes": len(fringes),
        "fringes": []
    }
    
    for fringe in fringes:
        num_images = len(fringe.images)
        num_pairs = (num_images * (num_images - 1)) // 2
        
        metadata["fringes"].append({
            "border": fringe.border_name,
            "num_images": num_images,
            "num_pairs": num_pairs,
            "sample_images": fringe.images[:5]
        })
    
    metadata_path = output_path / "fringe_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Wrote metadata to: {metadata_path}")


def main():
    args = parse_arguments()
    
    if not validate_inputs(args):
        sys.exit(1)
    
    boxes = load_box_definitions(args.box_coords)
    gps_map = load_gps_coordinates(args.gps_coords)
    
    borders = detect_shared_borders(boxes)
    
    if len(borders) == 0:
        print("Error: No shared borders detected between boxes")
        sys.exit(1)
    
    buffer_distance = args.fringe_length / 2
    output_path = Path(args.output_path)
    fringes = []
    
    print(f"\nGenerating fringe zones (buffer: {buffer_distance}m per side)...")
    
    for border in borders:
        fringe_polygon = create_fringe_polygon(border, buffer_distance)
        images = select_images_in_fringe(fringe_polygon, gps_map)
        
        if len(images) == 0:
            print(f"Warning: No images found in fringe {border.box_a}_{border.box_b}")
            continue
        
        border_name = f"{border.box_a}_{border.box_b}"
        fringes.append(FringeZone(
            border_name=border_name,
            polygon=fringe_polygon,
            images=images
        ))
        
        pairs = generate_exhaustive_pairs(images)
        pair_file = output_path / f"fringe_{border_name}_pairs.txt"
        write_pair_file(pairs, pair_file)
        
        print(f"  {border_name}: {len(images)} images → {len(pairs)} pairs")
        print(f"    Written to: {pair_file}")
    
    write_metadata(fringes, output_path, args.fringe_length)
    write_fringe_geometries(fringes, output_path)
    visualize_boxes_and_fringes(boxes, fringes, output_path)
    
    total_pairs = sum((len(f.images) * (len(f.images) - 1)) // 2 for f in fringes)
    print(f"\n✓ Generated {len(fringes)} fringe zones with {total_pairs} total pairs")
    print(f"✓ Outputs: pair files, metadata, fringes.json, and visualization PNG")


if __name__ == "__main__":
    main()
