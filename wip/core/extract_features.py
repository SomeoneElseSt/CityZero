"""
Extracts features assuming no shared cameras and no shared focal distances. 

"""

import argparse
import subprocess
import sys
from pathlib import Path


def extract_features(image_path, database_path) -> None:

    if image_path is None or not image_path:
        print("Image Path is None. Please provide a valid image path") 
        sys.exit(1)

    if database_path is None or not database_path:
        print("Database Path is None. Please provide a valid database path")
        sys.exit(1)

    print("Starting feature extraction.")

    cmd = [
        "colmap",
        "feature_extractor",
        "--image_path", str(image_path),
        "--database_path", str(database_path),
        "--camera_mode", "3",  # Different model for each Image
        "--ImageReader.single_camera", "0",  # No shared intrinsics
        "--ImageReader.single_camera_per_image", "1",  # Use a different camera for each image 
        "--FeatureExtraction.use_gpu", "1",  # Use GPU for extraction
        "--FeatureExtraction.num_threads", "-1"  # Use all Cores
    ]

    try:
        result = subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\n Feature Extraction failed with exit code: {e.returncode}")
        sys.exit(1)

    print("Feature Extraction is done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from images using COLMAP")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to directory containing images"
    )
    parser.add_argument(
        "--database_path",
        type=str,
        required=True,
        help="Path to output database file"
    )

    args = parser.parse_args()

    extract_features(args.image_path, args.database_path)
