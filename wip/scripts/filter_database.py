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
from typing import Dict, Set, Tuple, List, Any, Iterator

# Constants for optimization
BATCH_SIZE = 10000  # Number of rows to process at once
CACHE_SIZE = -2000000  # 2GB cache (negative value = KB)
PAIR_ID_BASE = 2147483647  # COLMAP's constant for pair_id encoding


def pair_id_to_image_ids(pair_id: int) -> Tuple[int, int]:
    """
    Decode COLMAP pair_id to image IDs.
    
    Args:
        pair_id: Encoded pair ID from COLMAP database
        
    Returns:
        Tuple of (image_id1, image_id2) where image_id1 <= image_id2
    """
    image_id2 = pair_id % PAIR_ID_BASE
    image_id1 = (pair_id - image_id2) // PAIR_ID_BASE
    return int(image_id1), int(image_id2)


def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    """
    Encode two image IDs into COLMAP pair_id.
    
    Args:
        image_id1: First image ID
        image_id2: Second image ID
        
    Returns:
        Encoded pair_id for COLMAP database
    """
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return PAIR_ID_BASE * image_id1 + image_id2


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


def create_tables_and_get_indices(source_cur: sqlite3.Cursor, output_cur: sqlite3.Cursor) -> List[str]:
    """
    Copy database schema (tables only) from source to output database.
    Returns list of index creation SQL statements to be executed later.
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        
    Returns:
        List of SQL statements to create indices
    """
    print("Copying database schema (tables only)...")
    indices = []
    
    try:
        schema = source_cur.execute(
            "SELECT sql, type FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    except sqlite3.Error as e:
        print(f"Failed to read database schema: {e}")
        sys.exit(1)
    
    for sql, type_ in schema:
        if type_ == 'index':
            indices.append(sql)
            continue
            
        try:
            output_cur.execute(sql)
        except sqlite3.Error as e:
            print(f"Failed to create table: {e}")
            sys.exit(1)
            
    return indices


def create_indices(output_cur: sqlite3.Cursor, indices: List[str]) -> None:
    """
    Create indices after bulk insertion.
    
    Args:
        output_cur: Output database cursor
        indices: List of SQL statements to create indices
    """
    print(f"Creating {len(indices)} indices...")
    for sql in indices:
        try:
            output_cur.execute(sql)
        except sqlite3.Error as e:
            print(f"Warning: Failed to create index: {e}")


def chunk_list(lst: List[Any], chunk_size: int) -> Iterator[List[Any]]:
    """Yield successive chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


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
        source_cur.execute("SELECT image_id, name, camera_id FROM images ORDER BY image_id")
    except sqlite3.Error as e:
        print(f"Failed to query images table: {e}")
        sys.exit(1)
    
    image_id_map = {}
    camera_ids = set()
    batch_data = []
    
    for row in source_cur:
        image_id, name, camera_id = row
        
        if name not in image_names:
            continue
        
        camera_ids.add(camera_id)
        image_id_map[image_id] = image_id
        
        batch_data.append((image_id, name, camera_id))
        
        if len(batch_data) >= BATCH_SIZE:
            try:
                output_cur.executemany(
                    "INSERT INTO images VALUES (?, ?, ?)",
                    batch_data
                )
                batch_data = []
            except sqlite3.Error as e:
                print(f"Failed to insert images batch: {e}")
                sys.exit(1)
                
    if batch_data:
        try:
            output_cur.executemany(
                "INSERT INTO images VALUES (?, ?, ?)",
                batch_data
            )
        except sqlite3.Error as e:
            print(f"Failed to insert remaining images: {e}")
            sys.exit(1)
    
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
    
    # Get column count from schema
    try:
        col_info = source_cur.execute("PRAGMA table_info(cameras)").fetchall()
        num_cols = len(col_info)
        placeholders_insert = ','.join('?' * num_cols)
    except sqlite3.Error as e:
        print(f"Failed to get cameras schema: {e}")
        sys.exit(1)
    
    camera_id_list = list(camera_ids)
    
    for chunk in chunk_list(camera_id_list, BATCH_SIZE // 10):  # Smaller batch for IN clause
        placeholders_where = ','.join('?' * len(chunk))
        try:
            source_cur.execute(
                f"SELECT * FROM cameras WHERE camera_id IN ({placeholders_where})",
                chunk
            )
            rows = source_cur.fetchall()
            
            if rows:
                output_cur.executemany(
                    f"INSERT INTO cameras VALUES ({placeholders_insert})", 
                    rows
                )
        except sqlite3.Error as e:
            print(f"Failed to copy cameras batch: {e}")
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
    
    # Increase fetch size for BLOBs
    source_cur.arraysize = BATCH_SIZE
    
    old_ids = list(image_id_map.keys())
    
    # Process in chunks to avoid large IN clauses
    for chunk_old_ids in chunk_list(old_ids, 900):  # SQLite limit is often 999 vars
        placeholders = ','.join('?' * len(chunk_old_ids))
        
        try:
            source_cur.execute(
                f"SELECT image_id, rows, cols, data FROM keypoints WHERE image_id IN ({placeholders})",
                chunk_old_ids
            )
            
            batch_data = []
            for row in source_cur:
                old_id = row[0]
                new_id = image_id_map[old_id]
                batch_data.append((new_id,) + row[1:])
            
            if batch_data:
                output_cur.executemany(
                    "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
                    batch_data
                )
                
        except sqlite3.Error as e:
            print(f"Failed to filter keypoints batch: {e}")
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
    
    source_cur.arraysize = BATCH_SIZE
    old_ids = list(image_id_map.keys())
    
    for chunk_old_ids in chunk_list(old_ids, 900):
        placeholders = ','.join('?' * len(chunk_old_ids))
        
        try:
            source_cur.execute(
                f"SELECT image_id, rows, cols, data FROM descriptors WHERE image_id IN ({placeholders})",
                chunk_old_ids
            )
            
            batch_data = []
            for row in source_cur:
                old_id = row[0]
                new_id = image_id_map[old_id]
                batch_data.append((new_id,) + row[1:])
                
            if batch_data:
                output_cur.executemany(
                    "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
                    batch_data
                )
                
        except sqlite3.Error as e:
            print(f"Failed to filter descriptors batch: {e}")
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
    
    # Check if table exists
    try:
        table_exists = source_cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='frames'"
        ).fetchone()
        if not table_exists:
            print("Table frames does not exist, skipping...")
            return
    except sqlite3.Error as e:
        print(f"Warning: Could not check frames table: {e}")
        return
    
    old_ids = list(image_id_map.keys())
    
    for chunk_old_ids in chunk_list(old_ids, 900):
        placeholders = ','.join('?' * len(chunk_old_ids))
        
        try:
            source_cur.execute(
                f"SELECT frame_id, rig_id FROM frames WHERE frame_id IN ({placeholders})",
                chunk_old_ids
            )
            
            batch_data = []
            for row in source_cur:
                old_frame_id, old_rig_id = row
                new_id = image_id_map[old_frame_id]
                # If rig_id is in our filtered set, use the new ID; otherwise keep the default
                new_rig_id = image_id_map.get(old_rig_id, new_id)
                
                batch_data.append((new_id, new_rig_id))
            
            if batch_data:
                output_cur.executemany(
                    "INSERT INTO frames VALUES (?, ?)",
                    batch_data
                )
                
        except sqlite3.Error as e:
            print(f"Failed to filter frames batch: {e}")
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
    
    # Check if table exists
    try:
        table_exists = source_cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='frame_data'"
        ).fetchone()
        if not table_exists:
            print("Table frame_data does not exist, skipping...")
            return
    except sqlite3.Error as e:
        print(f"Warning: Could not check frame_data table: {e}")
        return
    
    old_ids = list(image_id_map.keys())
    
    for chunk_old_ids in chunk_list(old_ids, 900):
        placeholders = ','.join('?' * len(chunk_old_ids))
        
        try:
            source_cur.execute(
                f"SELECT frame_id, data_id, sensor_id, sensor_type FROM frame_data WHERE frame_id IN ({placeholders})",
                chunk_old_ids
            )
            
            batch_data = []
            for row in source_cur:
                old_frame_id, old_data_id, sensor_id, sensor_type = row
                new_id = image_id_map[old_frame_id]
                new_data_id = image_id_map.get(old_data_id, new_id)
                new_sensor_id = image_id_map.get(sensor_id, sensor_id)
                
                batch_data.append((new_id, new_data_id, new_sensor_id, sensor_type))
            
            if batch_data:
                output_cur.executemany(
                    "INSERT INTO frame_data VALUES (?, ?, ?, ?)",
                    batch_data
                )
                
        except sqlite3.Error as e:
            print(f"Failed to filter frame_data batch: {e}")
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
    
    # Check if table exists
    try:
        table_exists = source_cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='rigs'"
        ).fetchone()
        if not table_exists:
            print("Table rigs does not exist, skipping...")
            return
    except sqlite3.Error as e:
        print(f"Warning: Could not check rigs table: {e}")
        return
    
    old_ids = list(image_id_map.keys())
    
    for chunk_old_ids in chunk_list(old_ids, 900):
        placeholders = ','.join('?' * len(chunk_old_ids))
        
        try:
            source_cur.execute(
                f"SELECT rig_id, ref_sensor_id, ref_sensor_type FROM rigs WHERE rig_id IN ({placeholders})",
                chunk_old_ids
            )
            
            batch_data = []
            for row in source_cur:
                old_rig_id, ref_sensor_id, ref_sensor_type = row
                new_id = image_id_map[old_rig_id]
                new_ref_sensor_id = image_id_map.get(ref_sensor_id, ref_sensor_id)
                batch_data.append((new_id, new_ref_sensor_id, ref_sensor_type))
            
            if batch_data:
                output_cur.executemany(
                    "INSERT INTO rigs VALUES (?, ?, ?)",
                    batch_data
                )
        except sqlite3.Error as e:
            print(f"Failed to filter rigs batch: {e}")
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
        # Check if table has data first to avoid unnecessary work
        count = source_cur.execute("SELECT COUNT(*) FROM rig_sensors").fetchone()[0]
        if count == 0:
            return
    except sqlite3.Error:
        return
        
    old_ids = list(image_id_map.keys())
    
    for chunk_old_ids in chunk_list(old_ids, 900):
        placeholders = ','.join('?' * len(chunk_old_ids))
        
        try:
            source_cur.execute(
                f"SELECT rig_id, sensor_id, sensor_type, sensor_from_rig FROM rig_sensors WHERE rig_id IN ({placeholders})",
                chunk_old_ids
            )
            
            batch_data = []
            for row in source_cur:
                old_rig_id = row[0]
                new_id = image_id_map[old_rig_id]
                batch_data.append((new_id,) + row[1:])
            
            if batch_data:
                output_cur.executemany(
                    "INSERT INTO rig_sensors VALUES (?, ?, ?, ?)",
                    batch_data
                )
        except sqlite3.Error as e:
            print(f"Failed to copy rig_sensors batch: {e}")
            sys.exit(1)


def filter_matches(source_cur: sqlite3.Cursor,
                  output_cur: sqlite3.Cursor,
                  image_id_map: Dict[int, int]) -> int:
    """
    Filter matches for specified image pairs.
    NOTE: Matches are NOT required by COLMAP mapper - only two_view_geometries are used.
    This function is skipped to save time and disk space.
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        image_id_map: Mapping from old image_id to new image_id
    
    Returns:
        Number of filtered match pairs (always 0 - skipped)
    """
    print("Skipping matches (not needed by COLMAP mapper)...")
    return 0


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
    
    source_cur.arraysize = BATCH_SIZE
    try:
        source_cur.execute("SELECT * FROM two_view_geometries")
    except sqlite3.Error as e:
        print(f"Failed to query two_view_geometries: {e}")
        sys.exit(1)
    
    geom_count = 0
    batch_data = []
    
    while True:
        rows = source_cur.fetchmany(BATCH_SIZE)
        if not rows:
            break
            
        for row in rows:
            pair_id = row[0]
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            
            if image_id1 not in image_id_map or image_id2 not in image_id_map:
                continue
            
            batch_data.append((pair_id,) + row[1:])
        
        if len(batch_data) >= BATCH_SIZE:
            try:
                output_cur.executemany(
                    "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch_data
                )
                geom_count += len(batch_data)
                batch_data = []
            except sqlite3.Error as e:
                print(f"Failed to insert geometries batch: {e}")
                sys.exit(1)
                
    if batch_data:
        try:
            output_cur.executemany(
                "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch_data
            )
            geom_count += len(batch_data)
        except sqlite3.Error as e:
            print(f"Failed to insert remaining geometries: {e}")
            sys.exit(1)
    
    print(f"Filtered to {geom_count} two_view_geometries")
    return geom_count


def filter_pose_priors(source_cur: sqlite3.Cursor,
                      output_cur: sqlite3.Cursor,
                      image_id_map: Dict[int, int]) -> int:
    """
    Filter pose_priors for specified images.
    Keeps original pose_prior_id (primary key) and only remaps corr_data_id.
    
    Args:
        source_cur: Source database cursor
        output_cur: Output database cursor
        image_id_map: Mapping from old image_id to new image_id
    
    Returns:
        Number of filtered pose priors
    """
    print("Filtering pose_priors...")
    
    # Check if table exists first
    try:
        table_exists = source_cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pose_priors'"
        ).fetchone()
        
        if not table_exists:
            print("Table pose_priors does not exist, skipping...")
            return 0
            
        # Get column count
        col_info = source_cur.execute("PRAGMA table_info(pose_priors)").fetchall()
        num_cols = len(col_info)
        placeholders_insert = ','.join('?' * num_cols)
        
    except sqlite3.Error as e:
        print(f"Warning: Could not check pose_priors table: {e}")
        return 0
    
    try:
        source_cur.execute("SELECT * FROM pose_priors")
    except sqlite3.Error as e:
        print(f"Warning: Failed to query pose_priors: {e}")
        return 0
    
    source_cur.arraysize = BATCH_SIZE
    prior_count = 0
    batch_data = []
    
    while True:
        rows = source_cur.fetchmany(BATCH_SIZE)
        if not rows:
            break
            
        for row in rows:
            pose_prior_id, corr_data_id = row[0], row[1]
            
            if corr_data_id not in image_id_map:
                continue
            
            new_data_id = image_id_map[corr_data_id]
            
            # Keep original pose_prior_id, only remap corr_data_id
            batch_data.append((pose_prior_id, new_data_id) + row[2:])
            prior_count += 1
        
        if len(batch_data) >= BATCH_SIZE:
            try:
                output_cur.executemany(
                    f"INSERT INTO pose_priors VALUES ({placeholders_insert})",
                    batch_data
                )
                batch_data = []
            except sqlite3.Error as e:
                print(f"Failed to insert pose_priors batch: {e}")
                sys.exit(1)
            
    if batch_data:
        try:
            output_cur.executemany(
                f"INSERT INTO pose_priors VALUES ({placeholders_insert})",
                batch_data
            )
        except sqlite3.Error as e:
            print(f"Failed to insert remaining pose_priors: {e}")
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
    
    output_parent = output_path.parent
    if not output_parent.exists():
        try:
            output_parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Failed to create output directory: {e}")
            sys.exit(1)
    
    image_names = load_image_list(image_list_path)

    try: 
        source_conn = sqlite3.connect(source_db_path)
    except sqlite3.Error as e:
        print(f"Failed to connect to source database: {e}")
        sys.exit(1)

    try:
        output_conn = sqlite3.connect(output_db_path)
        # Performance pragmas
        output_conn.execute("PRAGMA synchronous = OFF")
        output_conn.execute("PRAGMA journal_mode = OFF")
        output_conn.execute(f"PRAGMA cache_size = {CACHE_SIZE}")
        output_conn.execute("PRAGMA temp_store = MEMORY")
        output_conn.execute("PRAGMA locking_mode = EXCLUSIVE")
    except sqlite3.Error as e: 
        print(f"Failed to connect to output database: {e}")
        sys.exit(1)        
    
    source_cur = source_conn.cursor()
    output_cur = output_conn.cursor()
    
    try:
        # Start transaction explicitly
        output_cur.execute("BEGIN TRANSACTION")
        
        indices = create_tables_and_get_indices(source_cur, output_cur)
        
        image_id_map, camera_ids = filter_images(source_cur, output_cur, image_names)
        
        if not image_id_map:
            print("No images matched from the list")
            sys.exit(1)
        
        # Verify we found all requested images
        images_found = len(image_id_map)
        images_requested = len(image_names)
        if images_found < images_requested:
            print(f"Warning: Only found {images_found} of {images_requested} requested images")
        
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
        
        # Create indices at the end
        create_indices(output_cur, indices)
        
        output_conn.commit()
        
        print(f"\nSuccessfully created filtered database: {output_db_path}")
        print(f"Summary:")
        print(f"  - Images: {len(image_id_map)} (requested: {images_requested})")
        print(f"  - Cameras: {len(camera_ids)}")
        print(f"  - Matches: {match_count}")
        print(f"  - Two-view geometries: {geom_count}")
        print(f"  - Pose priors: {prior_count}")
        
        # Final verification: check output database
        output_image_count = output_cur.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        if output_image_count != len(image_id_map):
            print(f"ERROR: Image count mismatch! Expected {len(image_id_map)}, got {output_image_count}")
            sys.exit(1)
        
    except Exception as e:
        print(f"Unexpected error during filtering: {e}")
        import traceback
        traceback.print_exc()
        output_conn.rollback()
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
