import json
import os
import sys

def main():
    # Configuration based on user description
    # Run from cityzero-sf root
    GEODATA_DIR = "data/geodata"
    RAW_DIR = "data/raw"
    
    # 1. Find the metadata JSON file
    if len(sys.argv) > 1:
        metadata_path = sys.argv[1]
    else:
        if not os.path.isdir(GEODATA_DIR):
            print(f"Error: Directory '{GEODATA_DIR}' not found.")
            return
        
        json_files = [f for f in os.listdir(GEODATA_DIR) if f.endswith('.json')]
        if not json_files:
            print(f"Error: No .json file found in '{GEODATA_DIR}'.")
            return
        metadata_path = os.path.join(GEODATA_DIR, json_files[0])
    
    print(f"Reading metadata from: {metadata_path}")

    # 2. Load IDs from metadata
    try:
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            
        # The structure from get_gps_coords.py is {"downloaded_ids": { "id": {...}, ... }}
        if "downloaded_ids" not in data:
            print("Error: 'downloaded_ids' key not found in JSON.")
            return
            
        downloaded_ids = data["downloaded_ids"]
        if isinstance(downloaded_ids, dict):
            target_ids = set(downloaded_ids.keys())
        elif isinstance(downloaded_ids, list):
            target_ids = set(str(x) for x in downloaded_ids)
        else:
            print("Error: 'downloaded_ids' is not a list or dict.")
            return
            
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return

    print(f"Metadata contains {len(target_ids)} image IDs.")

    # 3. Check files in raw directory
    if not os.path.isdir(RAW_DIR):
        print(f"Error: Directory '{RAW_DIR}' not found.")
        return

    found_files = os.listdir(RAW_DIR)
    # Remove extensions to get IDs (e.g. "12345.jpg" -> "12345")
    found_ids = set(os.path.splitext(f)[0] for f in found_files if not f.startswith('.'))
    
    print(f"Found {len(found_ids)} files in '{RAW_DIR}'.")

    # 4. Compare
    missing_ids = target_ids - found_ids
    
    if not missing_ids:
        print("\nSUCCESS: All IDs from metadata are present in raw directory.")
    else:
        print(f"\nFAILURE: {len(missing_ids)} IDs are missing from raw directory.")
        print("First 10 missing IDs:")
        for mid in list(missing_ids)[:10]:
            print(f" - {mid}")
    
    # Optional: Check for extra files
    extra_ids = found_ids - target_ids
    if extra_ids:
        print(f"\nNote: {len(extra_ids)} files in raw directory are not in metadata.")

if __name__ == "__main__":
    main()
