#!/usr/bin/env python3
"""
GPS Injection Script for COLMAP Database

Injects GPS coordinates from JSON metadata into COLMAP's pose_priors table.
Uses chunked batch queries for efficient processing of 575k+ images.

Lambda Paths:
- GPS JSON: /home/ubuntu/cityzero-sf/data/geodata/download_metadata.json
- Database: /home/ubuntu/cityzero-sf/outputs/ft_match_db/database.db
"""

import json
import sqlite3
import numpy as np
from pathlib import Path

# Configuration
GPS_JSON_PATH = "/home/ubuntu/cityzero-sf/data/geodata/download_metadata.json"
DATABASE_PATH = "/home/ubuntu/cityzero-sf/outputs/ft_match_db/database.db"
CHUNK_SIZE = 30000  # SQL parameter limit safety margin


def load_gps_data(json_path):
    """
    Load GPS coordinates from JSON metadata file.
    
    Args:
        json_path: Path to JSON file containing downloaded_ids with lat/lon
    
    Returns:
        Dictionary mapping filenames to GPS coordinates
        Format: {filename.jpg: {'lat': float, 'lon': float}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract downloaded_ids section
    downloaded_ids = data.get('downloaded_ids', {})
    
    # Convert to filename-based dict with .jpg extension
    gps_dict = {}
    for image_id, coords in downloaded_ids.items():
        filename = f"{image_id}.jpg"
        gps_dict[filename] = {
            'lat': float(coords['lat']),
            'lon': float(coords['lon'])
        }
    
    return gps_dict


def inject_gps_to_database(db_path, gps_dict):
    """
    Inject GPS coordinates into COLMAP database pose_priors table.
    
    Uses chunked batch queries to efficiently handle ~600k images without
    loading entire database into memory. GPS data is stored as binary blobs
    in WGS84 coordinate system format.
    
    Args:
        db_path: Path to COLMAP database file
        gps_dict: Dictionary of {filename: {'lat': x, 'lon': y}}
    
    Returns:
        Tuple of (matched_count, unmatched_count)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"GPS coordinates to inject: {len(gps_dict):,}")
    
    # Batch lookup image IDs using chunked IN queries
    all_filenames = list(gps_dict.keys())
    image_map = {}
    
    print("Looking up image IDs from database...")
    for i in range(0, len(all_filenames), CHUNK_SIZE):
        chunk = all_filenames[i:i+CHUNK_SIZE]
        placeholders = ','.join('?' * len(chunk))
        
        query = f'SELECT image_id, name FROM images WHERE name IN ({placeholders})'
        cursor.execute(query, chunk)
        
        # Build filename -> image_id mapping
        for img_id, name in cursor.fetchall():
            image_map[name] = img_id
        
        processed = min(i + CHUNK_SIZE, len(all_filenames))
        print(f"  Processed {processed:,}/{len(all_filenames):,} lookups")
    
    print(f"Found {len(image_map):,} matching images in database")
    
    # Prepare batch inserts for pose_priors table
    inserts = []
    unmatched = []
    
    print("Encoding GPS coordinates as binary blobs...")
    for filename, coords in gps_dict.items():
        if filename not in image_map:
            unmatched.append(filename)
            continue
        
        image_id = image_map[filename]
        
        # Encode position as numpy array: [latitude, longitude, altitude]
        position = np.array([coords['lat'], coords['lon'], 0.0], dtype=np.float64)
        position_blob = position.tobytes()
        
        # Prepare row for pose_priors table
        # (corr_data_id, corr_sensor_id, corr_sensor_type, position,
        #  position_covariance, gravity, coordinate_system)
        inserts.append((image_id, 1, 1, position_blob, None, None, 1))
    
    print(f"Matched: {len(inserts):,} images")
    if unmatched:
        print(f"Unmatched: {len(unmatched):,} images")
        print(f"  First few: {unmatched[:5]}")
    
    # Batch insert into pose_priors
    print("Inserting GPS data into pose_priors table...")
    cursor.executemany('''
        INSERT INTO pose_priors 
        (corr_data_id, corr_sensor_id, corr_sensor_type, position, 
         position_covariance, gravity, coordinate_system)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', inserts)
    
    conn.commit()
    
    # Verify insertion
    cursor.execute('SELECT COUNT(*) FROM pose_priors')
    total_priors = cursor.fetchone()[0]
    print(f"✓ Successfully inserted {len(inserts):,} pose priors")
    print(f"✓ Total rows in pose_priors: {total_priors:,}")
    
    conn.close()
    return len(inserts), len(unmatched)


def main():
    """Main execution workflow."""
    print("=" * 70)
    print("GPS Injection into COLMAP Database")
    print("=" * 70)
    print()
    
    # Verify input files exist
    if not Path(GPS_JSON_PATH).exists():
        print(f"❌ Error: GPS JSON file not found")
        print(f"   Path: {GPS_JSON_PATH}")
        return
    
    if not Path(DATABASE_PATH).exists():
        print(f"❌ Error: Database not found")
        print(f"   Path: {DATABASE_PATH}")
        return
    
    # Load GPS data from JSON
    print(f"Loading GPS data from:")
    print(f"  {GPS_JSON_PATH}")
    gps_dict = load_gps_data(GPS_JSON_PATH)
    print(f"✓ Loaded {len(gps_dict):,} GPS coordinates")
    print()
    
    # Inject GPS into database
    print(f"Connecting to database:")
    print(f"  {DATABASE_PATH}")
    print()
    
    matched, unmatched = inject_gps_to_database(DATABASE_PATH, gps_dict)
    
    # Summary
    print()
    print("=" * 70)
    print("Injection Complete!")
    print("=" * 70)
    print(f"  ✓ Matched:   {matched:,} images")
    print(f"  ⚠ Unmatched: {unmatched:,} images")
    print(f"  → Success rate: {100 * matched / len(gps_dict):.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
