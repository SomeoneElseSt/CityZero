"""
Filter COLMAP database to contain only specified images.

This script:
1. Reads a list of image names from a text file
2. Creates a new database containing only those images
3. Remaps all related tables (keypoints, descriptors, matches, geometries, etc.)
4. Preserves all relationships and data integrity

The filtered database is useful for working with subsets of large reconstructions
without needing to recompute features and matches from scratch.
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Set, Tuple


def load_image_list(image_list_path: str) -> Set[str]:
    """
    Load image names from text file.
    
    Args:
        image_list_path: Path to text file with one image name per line
    
    Returns:
        Set of image names (filenames)
    """
    try:
        with open(image_list_path, 'r') as f:
            image_names = set(line.strip() for line in f if line.strip())
    except OSError as e:
        print(f"Failed to read image list: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(image_names)} images from list")
    return image_names


def copy_database_schema(source_cur: sqlite3.Cursor, output_cur: sqlite3.Cursor) -> None:
    """
    Copy database schema from source to output database.
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
    """
    print("Copying database schema...")
    
    try:
        schema = source_cur.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    except sqlite3.Error as e:
        print(f"Failed to read database schema: {e}")
        sys.exit(1)
    
    for sql, in schema:
        try:
            output_cur.execute(sql)
        except sqlite3.Error as e:
            print(f"Failed to create table: {e}")
            sys.exit(1)


def filter_images(source_cur: sqlite3.Cursor, 
                 output_cur: sqlite3.Cursor,
                 image_names: Set[str]) -> Tuple[Dict[int, int], Set[int]]:
    """
    Filter images table and build ID mapping.
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        image_names: Set of image names to keep
    
    Returns:
        Tuple of (image_id_map, camera_ids) where:
        - image_id_map: Dict mapping old image_id to new image_id
        - camera_ids: Set of camera IDs used by filtered images
    """
    print("Filtering images...")
    
    try:
        source_cur.execute("SELECT image_id, name, camera_id FROM images")
    except sqlite3.Error as e:
        print(f"Failed to query images table: {e}")
        sys.exit(1)
    
    image_id_map = {}
    camera_ids = set()
    new_image_id = 1
    
    for row in source_cur:
        image_id, name, camera_id = row
        
        if name not in image_names:
            continue
        
        camera_ids.add(camera_id)
        image_id_map[image_id] = new_image_id
        
        try:
            output_cur.execute(
                "INSERT INTO images VALUES (?, ?, ?)",
                (new_image_id, name, camera_id)
            )
        except sqlite3.Error as e:
            print(f"Failed to insert image {name}: {e}")
            sys.exit(1)
        
        new_image_id += 1
    
    print(f"Filtered to {len(image_id_map)} images using {len(camera_ids)} cameras")
    return image_id_map, camera_ids


def copy_cameras(source_cur: sqlite3.Cursor,
                output_cur: sqlite3.Cursor,
                camera_ids: Set[int]) -> None:
    """
    Copy camera entries for filtered images.
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        camera_ids: Set of camera IDs to copy
    """
    print("Copying cameras...")
    
    if not camera_ids:
        return
    
    placeholders = ','.join('?' * len(camera_ids))
    
    try:
        source_cur.execute(
            f"SELECT * FROM cameras WHERE camera_id IN ({placeholders})",
            list(camera_ids)
        )
    except sqlite3.Error as e:
        print(f"Failed to query cameras: {e}")
        sys.exit(1)
    
    for row in source_cur:
        try:
            output_cur.execute("INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)", row)
        except sqlite3.Error as e:
            print(f"Failed to insert camera: {e}")
            sys.exit(1)


def filter_keypoints(source_cur: sqlite3.Cursor,
                    output_cur: sqlite3.Cursor,
                    image_id_map: Dict[int, int]) -> None:
    """
    Filter keypoints for specified images.
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        image_id_map: Mapping from old image_id to new image_id
    """
    print("Filtering keypoints...")
    
    for old_id, new_id in image_id_map.items():
        try:
            source_cur.execute(
                "SELECT rows, cols, data FROM keypoints WHERE image_id = ?",
                (old_id,)
            )
            row = source_cur.fetchone()
        except sqlite3.Error as e:
            print(f"Failed to query keypoints for image {old_id}: {e}")
            sys.exit(1)
        
        if not row:
            continue
        
        try:
            output_cur.execute(
                "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
                (new_id,) + row
            )
        except sqlite3.Error as e:
            print(f"Failed to insert keypoints for image {new_id}: {e}")
            sys.exit(1)


def filter_descriptors(source_cur: sqlite3.Cursor,
                      output_cur: sqlite3.Cursor,
                      image_id_map: Dict[int, int]) -> None:
    """
    Filter descriptors for specified images.
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        image_id_map: Mapping from old image_id to new image_id
    """
    print("Filtering descriptors...")
    
    for old_id, new_id in image_id_map.items():
        try:
            source_cur.execute(
                "SELECT rows, cols, data FROM descriptors WHERE image_id = ?",
                (old_id,)
            )
            row = source_cur.fetchone()
        except sqlite3.Error as e:
            print(f"Failed to query descriptors for image {old_id}: {e}")
            sys.exit(1)
        
        if not row:
            continue
        
        try:
            output_cur.execute(
                "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
                (new_id,) + row
            )
        except sqlite3.Error as e:
            print(f"Failed to insert descriptors for image {new_id}: {e}")
            sys.exit(1)


def filter_frames(source_cur: sqlite3.Cursor,
                 output_cur: sqlite3.Cursor,
                 image_id_map: Dict[int, int]) -> None:
    """
    Filter frames (frame_id corresponds to image_id).
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        image_id_map: Mapping from old image_id to new image_id
    """
    print("Filtering frames...")
    
    for old_id, new_id in image_id_map.items():
        try:
            source_cur.execute(
                "SELECT rig_id FROM frames WHERE frame_id = ?",
                (old_id,)
            )
            row = source_cur.fetchone()
        except sqlite3.Error as e:
            print(f"Failed to query frames for image {old_id}: {e}")
            sys.exit(1)
        
        if not row:
            continue
        
        old_rig_id = row[0]
        new_rig_id = image_id_map.get(old_rig_id, new_id)
        
        try:
            output_cur.execute(
                "INSERT INTO frames VALUES (?, ?)",
                (new_id, new_rig_id)
            )
        except sqlite3.Error as e:
            print(f"Failed to insert frame for image {new_id}: {e}")
            sys.exit(1)


def filter_frame_data(source_cur: sqlite3.Cursor,
                     output_cur: sqlite3.Cursor,
                     image_id_map: Dict[int, int]) -> None:
    """
    Filter frame_data (frame_id and data_id both correspond to image_id).
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        image_id_map: Mapping from old image_id to new image_id
    """
    print("Filtering frame_data...")
    
    for old_id, new_id in image_id_map.items():
        try:
            source_cur.execute(
                "SELECT data_id, sensor_id, sensor_type FROM frame_data WHERE frame_id = ?",
                (old_id,)
            )
            row = source_cur.fetchone()
        except sqlite3.Error as e:
            print(f"Failed to query frame_data for image {old_id}: {e}")
            sys.exit(1)
        
        if not row:
            continue
        
        old_data_id, sensor_id, sensor_type = row
        new_data_id = image_id_map.get(old_data_id, new_id)
        
        try:
            output_cur.execute(
                "INSERT INTO frame_data VALUES (?, ?, ?, ?)",
                (new_id, new_data_id, sensor_id, sensor_type)
            )
        except sqlite3.Error as e:
            print(f"Failed to insert frame_data for image {new_id}: {e}")
            sys.exit(1)


def filter_rigs(source_cur: sqlite3.Cursor,
               output_cur: sqlite3.Cursor,
               image_id_map: Dict[int, int]) -> None:
    """
    Filter rigs (rig_id corresponds to image_id).
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        image_id_map: Mapping from old image_id to new image_id
    """
    print("Filtering rigs...")
    
    for old_id, new_id in image_id_map.items():
        try:
            source_cur.execute(
                "SELECT ref_sensor_id, ref_sensor_type FROM rigs WHERE rig_id = ?",
                (old_id,)
            )
            row = source_cur.fetchone()
        except sqlite3.Error as e:
            print(f"Failed to query rigs for image {old_id}: {e}")
            sys.exit(1)
        
        if not row:
            continue
        
        try:
            output_cur.execute(
                "INSERT INTO rigs VALUES (?, ?, ?)",
                (new_id,) + row
            )
        except sqlite3.Error as e:
            print(f"Failed to insert rig for image {new_id}: {e}")
            sys.exit(1)


def copy_rig_sensors(source_cur: sqlite3.Cursor,
                    output_cur: sqlite3.Cursor,
                    image_id_map: Dict[int, int]) -> None:
    """
    Copy rig_sensors if any exist.
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        image_id_map: Mapping from old image_id to new image_id
    """
    print("Copying rig_sensors...")
    
    try:
        rig_sensor_count = source_cur.execute(
            "SELECT COUNT(*) FROM rig_sensors"
        ).fetchone()[0]
    except sqlite3.Error as e:
        print(f"Failed to count rig_sensors: {e}")
        sys.exit(1)
    
    if rig_sensor_count == 0:
        return
    
    for old_id, new_id in image_id_map.items():
        try:
            source_cur.execute(
                "SELECT sensor_id, sensor_type, sensor_from_rig FROM rig_sensors WHERE rig_id = ?",
                (old_id,)
            )
        except sqlite3.Error as e:
            print(f"Failed to query rig_sensors for image {old_id}: {e}")
            sys.exit(1)
        
        for row in source_cur:
            try:
                output_cur.execute(
                    "INSERT INTO rig_sensors VALUES (?, ?, ?, ?)",
                    (new_id,) + row
                )
            except sqlite3.Error as e:
                print(f"Failed to insert rig_sensor for image {new_id}: {e}")
                sys.exit(1)


def filter_matches(source_cur: sqlite3.Cursor,
                  output_cur: sqlite3.Cursor,
                  image_id_map: Dict[int, int]) -> int:
    """
    Filter matches for specified image pairs.
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        image_id_map: Mapping from old image_id to new image_id
    
    Returns:
        Number of filtered match pairs
    """
    print("Filtering matches...")
    
    try:
        source_cur.execute("SELECT pair_id, rows, cols, data FROM matches")
    except sqlite3.Error as e:
        print(f"Failed to query matches: {e}")
        sys.exit(1)
    
    match_count = 0
    
    for row in source_cur:
        pair_id = row[0]
        image_id1 = pair_id >> 32
        image_id2 = pair_id & 0xFFFFFFFF
        
        if image_id1 not in image_id_map:
            continue
        if image_id2 not in image_id_map:
            continue
        
        new_id1 = image_id_map[image_id1]
        new_id2 = image_id_map[image_id2]
        new_pair_id = (new_id1 << 32) | new_id2
        
        try:
            output_cur.execute(
                "INSERT INTO matches VALUES (?, ?, ?, ?)",
                (new_pair_id,) + row[1:]
            )
            match_count += 1
        except sqlite3.Error as e:
            print(f"Failed to insert match for pair {new_id1}-{new_id2}: {e}")
            sys.exit(1)
    
    print(f"Filtered to {match_count} match pairs")
    return match_count


def filter_two_view_geometries(source_cur: sqlite3.Cursor,
                              output_cur: sqlite3.Cursor,
                              image_id_map: Dict[int, int]) -> int:
    """
    Filter two_view_geometries for specified image pairs.
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        image_id_map: Mapping from old image_id to new image_id
    
    Returns:
        Number of filtered geometries
    """
    print("Filtering two_view_geometries...")
    
    try:
        source_cur.execute("SELECT * FROM two_view_geometries")
    except sqlite3.Error as e:
        print(f"Failed to query two_view_geometries: {e}")
        sys.exit(1)
    
    geom_count = 0
    
    for row in source_cur:
        pair_id = row[0]
        image_id1 = pair_id >> 32
        image_id2 = pair_id & 0xFFFFFFFF
        
        if image_id1 not in image_id_map:
            continue
        if image_id2 not in image_id_map:
            continue
        
        new_id1 = image_id_map[image_id1]
        new_id2 = image_id_map[image_id2]
        new_pair_id = (new_id1 << 32) | new_id2
        
        try:
            output_cur.execute(
                "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (new_pair_id,) + row[1:]
            )
            geom_count += 1
        except sqlite3.Error as e:
            print(f"Failed to insert geometry for pair {new_id1}-{new_id2}: {e}")
            sys.exit(1)
    
    print(f"Filtered to {geom_count} two_view_geometries")
    return geom_count


def filter_pose_priors(source_cur: sqlite3.Cursor,
                      output_cur: sqlite3.Cursor,
                      image_id_map: Dict[int, int]) -> int:
    """
    Filter pose_priors for specified images.
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        image_id_map: Mapping from old image_id to new image_id
    
    Returns:
        Number of filtered pose priors
    """
    print("Filtering pose_priors...")
    
    try:
        source_cur.execute("SELECT * FROM pose_priors")
    except sqlite3.Error as e:
        print(f"Failed to query pose_priors: {e}")
        sys.exit(1)
    
    prior_count = 0
    new_prior_id = 1
    
    for row in source_cur:
        pose_prior_id, corr_data_id = row[0], row[1]
        
        if corr_data_id not in image_id_map:
            continue
        
        new_data_id = image_id_map[corr_data_id]
        
        try:
            output_cur.execute(
                "INSERT INTO pose_priors VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (new_prior_id, new_data_id) + row[2:]
            )
            new_prior_id += 1
            prior_count += 1
        except sqlite3.Error as e:
            print(f"Failed to insert pose_prior: {e}")
            sys.exit(1)
    
    print(f"Filtered to {prior_count} pose_priors")
    return prior_count


def filter_database(source_db_path: str, 
                   image_list_path: str, 
                   output_db_path: str) -> None:
    """
    Create a filtered COLMAP database containing only specified images.
    
    Args:
        source_db_path: Path to source COLMAP database
        image_list_path: Path to text file with image names (one per line)
        output_db_path: Path to output filtered database
    """
    if not source_db_path or not Path(source_db_path).exists():
        print(f"Source database not found: {source_db_path}")
        sys.exit(1)
    
    if ".db" not in str(source_db_path):
        print("Source database path must end with .db")
        sys.exit(1)
    
    if not image_list_path or not Path(image_list_path).exists():
        print(f"Image list not found: {image_list_path}")
        sys.exit(1)
    
    if ".txt" not in str(image_list_path):
        print("Image list path must end with .txt")
        sys.exit(1)
    
    if not output_db_path:
        print("Output database path is required")
        sys.exit(1)
    
    if ".db" not in str(output_db_path):
        print("Output database path must end with .db")
        sys.exit(1)
    
    output_path = Path(output_db_path)
    if output_path.exists():
        print(f"Output database already exists: {output_db_path}")
        print("Please remove it first or choose a different path")
        sys.exit(1)
    
    image_names = load_image_list(image_list_path)
    
    try:
        source_conn = sqlite3.connect(source_db_path)
        output_conn = sqlite3.connect(output_db_path)
    except sqlite3.Error as e:
        print(f"Failed to connect to database: {e}")
        sys.exit(1)
    
    source_cur = source_conn.cursor()
    output_cur = output_conn.cursor()
    
    try:
        copy_database_schema(source_cur, output_cur)
        
        image_id_map, camera_ids = filter_images(source_cur, output_cur, image_names)
        
        if not image_id_map:
            print("No images matched from the list")
            sys.exit(1)
        
        copy_cameras(source_cur, output_cur, camera_ids)
        filter_keypoints(source_cur, output_cur, image_id_map)
        filter_descriptors(source_cur, output_cur, image_id_map)
        filter_frames(source_cur, output_cur, image_id_map)
        filter_frame_data(source_cur, output_cur, image_id_map)
        filter_rigs(source_cur, output_cur, image_id_map)
        copy_rig_sensors(source_cur, output_cur, image_id_map)
        
        match_count = filter_matches(source_cur, output_cur, image_id_map)
        geom_count = filter_two_view_geometries(source_cur, output_cur, image_id_map)
        prior_count = filter_pose_priors(source_cur, output_cur, image_id_map)
        
        output_conn.commit()
        
        print(f"\nSuccessfully created filtered database: {output_db_path}")
        print(f"Summary: {len(image_id_map)} images, {match_count} matches, {geom_count} geometries, {prior_count} pose priors")
        
    except Exception as e:
        print(f"Unexpected error during filtering: {e}")
        sys.exit(1)
    finally:
        source_conn.close()
        output_conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter COLMAP database to contain only specified images"
    )
    parser.add_argument(
        "--source_db",
        type=str,
        required=True,
        help="Path to source COLMAP database file (.db)"
    )
    parser.add_argument(
        "--image_list",
        type=str,
        required=True,
        help="Path to text file containing image names (one per line)"
    )
    parser.add_argument(
        "--output_db",
        type=str,
        required=True,
        help="Path to output filtered database file (.db)"
    )
    
    args = parser.parse_args()
    
    filter_database(args.source_db, args.image_list, args.output_db)
