"""
Reconstruction step using COLMAP mapper

Assuming a well connected image graph thanks to query expansion, it uses a normal mapper with fairly default settings. 

If reconstructions fail to converge, it'd be worth trying a hierarchical_mapper instead. 

"""

import argparse
import os
import subprocess
import sys


def reconstruction(database_path, image_path, output_path, image_list_path) -> None:

    if database_path is None or not database_path:
        print("Database Path is None. Please provide a valid database path")
        sys.exit(1)

    if ".db" not in str(database_path):
        print("You must provide a path + .db file for the database path -- you did not provide a file at the end of the path -> database.db")
        sys.exit(1)

    if image_path is None or not image_path:
        print("Image Path is None. Please provide a valid image path")
        sys.exit(1)

    if image_list_path is None or not image_list_path:
        print("Image List Path is None. Please provide a valid image list path")
        sys.exit(1)

    if ".txt" not in str(image_list_path):
        print("You must provide a path + .txt file for the image list path -- you did not provide a file at the end of the path -> image_list.txt")
        sys.exit(1)

    if output_path is None or not output_path:
        print("Output Path is None. Please provide a valid output path")
        sys.exit(1)

    if not os.path.exists(output_path):
        print(f"Creating output directory: {output_path}")
        os.makedirs(output_path, exist_ok=True)

    print("Starting sparse reconstruction...")

    cmd = [
        "colmap",
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_path),
        "--output_path", str(output_path),
        "--Mapper.image_list_path", str(image_list_path),
        "--Mapper.ba_use_gpu", "1",
        "--Mapper.ba_gpu_index", "-1",                 # Auto-detect GPU
        "--Mapper.num_threads", "-1",                  # Auto-detect threads
        "--Mapper.ignore_watermarks", "1",             # Ignore Watermarks
        "--Mapper.init_min_num_inliers", "200",        # Most samples have 30 > inliers. This should constraint the starting pairs better. 
        "--Mapper.init_max_forward_motion", "0.4",     # Should be stricter about having the initial pairs be separate from each other, rather than, for example, a picture of two very-similar looking houses facing forwards (this should have it see the features are different thanks to the angular separation)
        "--Mapper.init_num_trials", "400",             # I saw many initial matches fail a lot, and even at 200 min_inliers, there are plenty, so it's good to try lots for rebudancy 
        "--Mapper.init_min_tri_angle", "32",           # Twice as much as the default for stricter geometry matches
        "--Mapper.abs_pose_min_inlier_ratio", "0.40",  # Almost twice as much as the default. Should significantly cut down on false positives and matches that aren't strong.
        "--Mapper.abs_pose_min_num_inliers", "80",     # Too many images are 30 > inliers, so this should cut down on easy matches and barely related images    ]

    try:
        result = subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\nReconstruction failed with exit code: {e.returncode}")
        sys.exit(1)

    print("\nReconstruction is done!")
    print(f"Sparse models saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sparse reconstruction using COLMAP mapper"
    )
    parser.add_argument(
        "--database_path",
        type=str,
        required=True,
        help="Path to COLMAP database file (.db)",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to directory containing images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output directory for sparse reconstruction",
    )
    parser.add_argument(
        "--image_list_path",
        type=str,
        required=True,
        help="Path to .txt file containing list of images to reconstruct",
    )

    args = parser.parse_args()

    reconstruction(
        args.database_path,
        args.image_path,
        args.output_path,
        args.image_list_path,
    )
