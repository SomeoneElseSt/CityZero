"""
I figured that since the default vocabulary by COLMAP isn't fully centered on urban images
it makes more sense to train a custom vocab tree for feature matching down the line. This script does that. 
"""

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
        "--VocabTreeBuilding.num_visual_words", "1000000",
        "--VocabTreeBuilding.branching_factor", "10",
        "--VocabTreeBuilding.max_num_images", "15000",
        "--VocabTreeBuilding.num_kmeans_iterations", "10",
        "--VocabTreeBuilding.num_threads", "-1"
    ]
    
    
