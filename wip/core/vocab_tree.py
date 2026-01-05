"""
I figured that since the default vocabulary by COLMAP isn't fully centered on urban images
it makes more sense to train a custom vocab tree for feature matching down the line. This script does that. 
"""

import argparse
import subprocess
import sys

def build_vocab_tree(database_path, vocab_tree_path) -> None: 

    if database_path is None or not database_path:
        print("Database Path is None. Please provide a valid database path")
        sys.exit(1)

    if "database.db" not in str(database_path):
        print("You must provide a path + database file -- you did not provide a file at the end of the path -> database.db")
        sys.exit(1)

    if ".bin" not in str(vocab_tree_path):
        print("You must provide a path + a binary file for the custom vocabulary -- you did not provide a .bin file at the end of the path -> custom_vocab_tree.bin")
        sys.exit(1)

    print("Building custom vocab tree.")

    cmd = [
        "colmap",
        "vocab_tree_builder",
        "--database_path", str(database_path),
        "--vocab_tree_path", str(vocab_tree_path),
        "--num_visual_words", "1000000",
        "--max_num_images", "40000", # The images to be sampled for the tree
        "--num_threads", "30", # Lambda's AMD EPYC 7J13 
        "--num_iterations", "20", # Down from default for computational sake
    ]

    try:
        result = subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"\nVocab tree building failed with exit code: {e.returncode}")
        sys.exit(1)

    print("\nVocab tree building is done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build custom vocabulary tree using COLMAP")
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
        help="Path to output vocabulary tree binary file"
    )

    args = parser.parse_args()

    build_vocab_tree(args.database_path, args.vocab_tree_path)
