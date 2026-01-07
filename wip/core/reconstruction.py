"""
Reconstruction step using prev. built Vocab tree on a hierarchical mapper 

"""

import argparse
import os
import subprocess
import sys

def hierarchical_reconstruction(database_path, image_path, output_path) -> None:

    if not os.path.isfile(database_path):
        print(f"Database path does not point to a file: {database_path}")
        sys.exit(1)

    if not database_path.endswith(".db"):
        print("Database file should end with .db")
        sys.exit(1)

    if not os.path.isdir(image_path):
        print(f"Image path does not point to a directory: {image_path}")
        sys.exit(1)

    if not os.path.exists(output_path):
        print(f"Creating output directory: {output_path}")
        os.makedirs(output_path, exist_ok=True)

    print("Starting hierarchical sparse reconstruction...")

    cmd = [
        "colmap", "hierarchical_mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_path),
        "--output_path", str(output_path),
        "--num_workers", "6",
        "--leaf_max_num_images", "800",
        "--image_overlap", "50",
        # Sub-mapper options applied to each leaf 800-image cluster
        "--Mapper.num_threads", "5",
        "--Mapper.init_num_trials", "100",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.ba_local_max_num_iterations", "20",
        "--Mapper.ba_global_max_num_iterations", "30",
        "--Mapper.ba_use_gpu", "1",
        "--Mapper.ba_gpu_index", "-1",
        "--Mapper.ignore_watermarks", "1",  # Ignores street view watermarks
        "--Mapper.tri_min_angle", "2.0",
        "--Mapper.filter_min_tri_angle", "2.0",
        "--Mapper.max_num_models", "100",  # 100 Mini-SFs
        "--Mapper.ba_global_frames_ratio", "1.3",
        "--Mapper.ba_global_points_ratio", "1.3",
    ]
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\nHierarchical reconstruction failed with exit code: {e.returncode}")
        sys.exit(1)

    print("\nHierarchical reconstruction is done!")
    print(f"Sparse models saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sparse reconstruction using COLMAP hierarchical mapper"
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

    args = parser.parse_args()

    hierarchical_reconstruction(args.database_path, args.image_path, args.output_path)
