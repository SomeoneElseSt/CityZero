"""
Feature match images using previously computed database.db and custom vocab tree

"""

import argparse
import subprocess
import sys 

def feature_match(database_path, vocab_tree_path) -> None:

    if database_path is None or not database_path:
        print("Database Path is None. Please provide a valid database path")
        sys.exit(1)

    if ".db" not in str(database_path):
        print("You must provide a path + database file -- you did not provide a file at the end of the path -> database.db")
        sys.exit(1)

    if vocab_tree_path is None or not vocab_tree_path:
        print("Vocab Tree Path is None. Please provide a valid vocabulary tree path")
        sys.exit(1)

    if ".bin" not in str(vocab_tree_path):
        print("The vocabulary tree path is not pointing to a tree binary -- make sure you're pointing to the path + .bin file")
        sys.exit(1)
              
    print("Feature matching using Vocab Tree")

    cmd = [
        "colmap",
        "vocab_tree_matcher",
        "--database_path", str(database_path),
        "--FeatureMatching.use_gpu", "1",  # Use GPU for matching
        "--FeatureMatching.gpu_index", "-1", # Auto-detect GPU
        "--FeatureMatching.num_threads", "30",  # Lambda's AMD EPYC 7J13
        "--VocabTreeMatching.vocab_tree_path", str(vocab_tree_path),
        "--VocabTreeMatching.num_images", "200",
        "--VocabTreeMatching.num_nearest_neighbors", "5",
        "--VocabTreeMatching.num_checks", "256",
        "--VocabTreeMatching.num_images_after_verification", "150",
        "--FeatureMatching.guided_matching", "1", 
    ]

    try:
        result = subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\nFeature matching failed with exit code: {e.returncode}")
        sys.exit(1)

    print("\nFeature matching is done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature match images using COLMAP vocab tree matcher")
    parser.add_argument(
        "--database_path",
        type=str,
        required=True,
        help="Path to COLMAP database file"
    )
    parser.add_argument(
        "--vocab_tree_path",
        type=str,
        required=True,
        help="Path to vocabulary tree binary file"
    )

    args = parser.parse_args()

    feature_match(args.database_path, args.vocab_tree_path)
