#!/usr/bin/env python3
"""
Generate image pairs for each bounding box directory.
Reads {dir}_images.txt files and creates {dir}_pairs.txt with all combinations.
"""

import itertools
import os
from pathlib import Path

BASE_DIR = Path("/home/ubuntu/cityzero-sf/data/geodata/image_boxes")
DIRECTORIES = ["ne", "nw", "se", "sw"]


def read_image_list(file_path):
    """Read image filenames from a text file, stripping whitespace."""
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist, skipping")
        return None
    
    images = []
    with open(file_path, 'r') as f:
        for line in f:
            image_name = line.strip()
            if image_name:
                images.append(image_name)
    
    return images


def write_pairs(file_path, pairs):
    """Write image pairs to a text file, one pair per line."""
    with open(file_path, 'w') as f:
        for img1, img2 in pairs:
            f.write(f"{img1} {img2}\n")


def generate_pairs_for_directory(directory_name):
    """Generate pairs for a single directory."""
    dir_path = BASE_DIR / directory_name
    
    if not dir_path.exists():
        print(f"Warning: Directory {dir_path} does not exist, skipping")
        return
    
    images_file = dir_path / f"{directory_name}_images.txt"
    pairs_file = dir_path / f"{directory_name}_pairs.txt"
    
    images = read_image_list(images_file)
    if images is None:
        return
    
    if len(images) < 2:
        print(f"Warning: {directory_name} has less than 2 images, skipping")
        return
    
    print(f"Processing {directory_name}: {len(images)} images")
    num_pairs = len(images) * (len(images) - 1) // 2
    print(f"  Expected pairs: {num_pairs:,}")
    
    pairs = list(itertools.combinations(images, 2))
    write_pairs(pairs_file, pairs)
    
    print(f"✓ Generated {len(pairs):,} pairs for {directory_name} -> {pairs_file}")


def main():
    """Process all directories and generate pairs."""
    if not BASE_DIR.exists():
        print(f"Error: Base directory {BASE_DIR} does not exist")
        return
    
    for directory_name in DIRECTORIES:
        generate_pairs_for_directory(directory_name)
    
    print("Finished generating all image pairs")


if __name__ == "__main__":
    main()
