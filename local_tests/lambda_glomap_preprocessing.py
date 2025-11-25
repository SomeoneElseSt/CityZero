#!/usr/bin/env python3
"""
GLOMAP preprocessing for Lambda Cloud GPU instance.
This uses GLOMAP instead of COLMAP for 10-100x faster processing.

Usage:
    python3 lambda_glomap_preprocessing.py --images ~/images --output ~/glomap_output
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import shutil


# Directory structure
OUTPUT_SUBDIRS = [
    "database.db",  # Will be created by GLOMAP
    "sparse",       # Sparse reconstruction output
]


def check_gpu():
    """Check if GPU is available and print info."""
    print("Checking GPU availability...")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ GPU detected: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("✗ WARNING: nvidia-smi failed - GPU may not be available")
        return False
    except FileNotFoundError:
        print("✗ ERROR: nvidia-smi not found - NVIDIA drivers not installed")
        return False


def check_dependencies():
    """Check if required tools are installed."""
    print("\nChecking dependencies...")
    
    # Check COLMAP (GLOMAP depends on COLMAP libraries)
    try:
        result = subprocess.run(
            ["colmap", "-h"],
            capture_output=True,
            check=True
        )
        print("✓ COLMAP found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ ERROR: COLMAP not found")
        print("  Install with: sudo apt update && sudo apt install -y colmap")
        return False
    
    # Check GLOMAP
    try:
        result = subprocess.run(
            ["glomap", "-h"],
            capture_output=True,
            check=True
        )
        print("✓ GLOMAP found")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ GLOMAP not found - will attempt to install")
        return install_glomap()


def install_glomap():
    """Install GLOMAP on Ubuntu/Lambda Stack."""
    print("\nInstalling GLOMAP...")
    
    commands = [
        # Install build dependencies
        "sudo apt update",
        "sudo apt install -y git cmake build-essential libboost-all-dev libeigen3-dev",
        
        # Clone GLOMAP
        "cd ~ && git clone --recursive https://github.com/colmap/glomap.git",
        
        # Build GLOMAP
        "cd ~/glomap && mkdir build && cd build",
        "cd ~/glomap/build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=native",
        "cd ~/glomap/build && make -j$(nproc)",
        
        # Install
        "cd ~/glomap/build && sudo make install",
    ]
    
    for cmd in commands:
        print(f"  Running: {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"✗ ERROR: Command failed: {cmd}")
            return False
    
    print("✓ GLOMAP installed successfully")
    return True


def prepare_output_directory(output_dir: Path):
    """Create output directory structure."""
    print(f"\nPreparing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    
    print("✓ Output directory ready")


def run_colmap_feature_extraction(images_dir: Path, database_path: Path):
    """Run COLMAP feature extraction (GPU accelerated)."""
    print("\n" + "="*70)
    print("STEP 1/3: COLMAP FEATURE EXTRACTION (GPU)")
    print("="*70)
    
    cmd = [
        "colmap", "feature_extractor",
        "--image_path", str(images_dir),
        "--database_path", str(database_path),
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "0",
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.gpu_index", "0",
        "--SiftExtraction.max_num_features", "16384",
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1",
    ]
    
    print(f"\nExtracting features from images...")
    print(f"GPU accelerated: Yes")
    print()
    
    start_time = datetime.now()
    
    # Set environment variable to run headless
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    
    result = subprocess.run(cmd, env=env)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if result.returncode != 0:
        print(f"\n✗ ERROR: Feature extraction failed with code {result.returncode}")
        return False
    
    print(f"\n✓ Feature extraction completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    return True


def run_colmap_feature_matching(database_path: Path):
    """Run COLMAP feature matching (GPU accelerated)."""
    print("\n" + "="*70)
    print("STEP 2/3: COLMAP FEATURE MATCHING (GPU)")
    print("="*70)
    
    cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", "1",
        "--SiftMatching.gpu_index", "0",
        "--SiftMatching.guided_matching", "1",
        "--SiftMatching.max_num_matches", "65536",
    ]
    
    print(f"\nMatching features between images...")
    print(f"GPU accelerated: Yes")
    print()
    
    start_time = datetime.now()
    
    # Set environment variable to run headless
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    
    result = subprocess.run(cmd, env=env)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if result.returncode != 0:
        print(f"\n✗ ERROR: Feature matching failed with code {result.returncode}")
        return False
    
    print(f"\n✓ Feature matching completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    return True


def run_glomap_mapper(images_dir: Path, database_path: Path, sparse_dir: Path):
    """
    Run GLOMAP mapper (10-100x faster than COLMAP mapper).
    This uses the database created by COLMAP to do reconstruction.
    """
    print("\n" + "="*70)
    print("STEP 3/3: GLOMAP MAPPER (FAST RECONSTRUCTION)")
    print("="*70)
    
    cmd = [
        "glomap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ]
    
    print(f"\nRunning GLOMAP global reconstruction...")
    print(f"This is 10-100x faster than COLMAP's mapper!")
    print()
    
    start_time = datetime.now()
    
    result = subprocess.run(cmd)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if result.returncode != 0:
        print(f"\n✗ ERROR: GLOMAP mapper failed with code {result.returncode}")
        return False
    
    print(f"\n✓ GLOMAP mapper completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    return True


def verify_output(output_dir: Path):
    """Verify that output files were created."""
    print("\n" + "="*70)
    print("VERIFYING OUTPUT")
    print("="*70)
    
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    
    checks = {
        "Database": database_path.exists(),
        "Sparse directory": sparse_dir.exists(),
        "Cameras file": (sparse_dir / "cameras.bin").exists() or (sparse_dir / "cameras.txt").exists(),
        "Images file": (sparse_dir / "images.bin").exists() or (sparse_dir / "images.txt").exists(),
        "Points file": (sparse_dir / "points3D.bin").exists() or (sparse_dir / "points3D.txt").exists(),
    }
    
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed


def compress_output(output_dir: Path):
    """Compress output for download."""
    print("\n" + "="*70)
    print("COMPRESSING OUTPUT")
    print("="*70)
    
    output_archive = output_dir.parent / f"{output_dir.name}.tar.gz"
    
    cmd = [
        "tar", "-czf",
        str(output_archive),
        "-C", str(output_dir.parent),
        output_dir.name
    ]
    
    print(f"Creating: {output_archive}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("✗ ERROR: Failed to compress output")
        return False
    
    size_mb = output_archive.stat().st_size / (1024 * 1024)
    print(f"✓ Compressed to {output_archive} ({size_mb:.1f} MB)")
    return True


def create_summary(output_dir: Path, images_dir: Path, duration_seconds: float):
    """Create a summary file with processing info."""
    sparse_dir = output_dir / "sparse"
    
    # Try to get image count
    try:
        images_file = sparse_dir / "images.bin"
        if not images_file.exists():
            images_file = sparse_dir / "images.txt"
        
        if images_file.exists():
            if images_file.suffix == ".txt":
                with open(images_file) as f:
                    lines = [l for l in f if not l.startswith("#")]
                    num_images = len(lines) // 2  # Each image has 2 lines in txt format
            else:
                # For binary format, just count input images
                num_images = len(list(images_dir.glob("*.jpg")))
        else:
            num_images = len(list(images_dir.glob("*.jpg")))
    except Exception:
        num_images = "unknown"
    
    summary = {
        "tool": "GLOMAP",
        "version": "latest",
        "processing_time_seconds": duration_seconds,
        "processing_time_minutes": duration_seconds / 60,
        "images_processed": num_images,
        "output_directory": str(output_dir),
        "gpu_used": "A100 (Lambda Cloud)",
        "timestamp": datetime.now().isoformat(),
    }
    
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_file}")
    
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="GLOMAP preprocessing for Lambda Cloud (10-100x faster than COLMAP)"
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to directory containing images"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output directory"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("GLOMAP PREPROCESSING FOR LAMBDA CLOUD")
    print("="*70)
    print(f"Images: {args.images}")
    print(f"Output: {args.output}")
    print("="*70)
    
    # Validate inputs
    if not args.images.exists():
        print(f"\n✗ ERROR: Images directory does not exist: {args.images}")
        return 1
    
    num_images = len(list(args.images.glob("*.jpg")))
    if num_images == 0:
        print(f"\n✗ ERROR: No .jpg images found in {args.images}")
        return 1
    
    print(f"\n✓ Found {num_images} images")
    
    # Check GPU
    if not check_gpu():
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Prepare output directory
    prepare_output_directory(args.output)
    
    database_path = args.output / "database.db"
    sparse_dir = args.output / "sparse"
    
    # Run 3-step process
    start_time = datetime.now()
    
    # Step 1: COLMAP feature extraction
    if not run_colmap_feature_extraction(args.images, database_path):
        return 1
    
    # Step 2: COLMAP feature matching
    if not run_colmap_feature_matching(database_path):
        return 1
    
    # Step 3: GLOMAP mapper (the fast part!)
    if not run_glomap_mapper(args.images, database_path, sparse_dir):
        return 1
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Verify output
    if not verify_output(args.output):
        print("\n✗ ERROR: Output verification failed")
        return 1
    
    # Create summary
    create_summary(args.output, args.images, duration)
    
    # Compress output
    if not compress_output(args.output):
        print("\n✗ WARNING: Failed to compress output (but processing succeeded)")
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"\nOutput files are in: {args.output}")
    print(f"Compressed archive: {args.output.parent / f'{args.output.name}.tar.gz'}")
    print("\nTo download the results:")
    print(f"  scp -i *.pem ubuntu@YOUR_IP:~/{args.output.name}.tar.gz .")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
