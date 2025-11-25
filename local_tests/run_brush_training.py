#!/usr/bin/env python3
"""
Brush CLI Training Script for Mac

This script:
1. Checks if Brush is installed
2. Takes COLMAP output from train_gaussian_mac.py
3. Runs Brush CLI training to produce Gaussian Splat model

Usage:
    python run_brush_training.py

Requirements:
    - Brush installed (run setup_brush.py first)
    - Completed COLMAP processing (from train_gaussian_mac.py)
"""

import subprocess
import json
from pathlib import Path
import sys

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
IMAGES_DIR = SCRIPT_DIR / "financial_district_images" / "images"
COLMAP_SPARSE_DIR = SCRIPT_DIR / "outputs" / "colmap_output" / "colmap_output" / "sparse" / "0"
BRUSH_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "gaussian_splatting" / "brush"
TRAINING_INFO_PATH = SCRIPT_DIR / "outputs" / "gaussian_splatting" / "training_info.json"

# Brush location
BRUSH_DIR = Path.home() / ".brush"
BRUSH_EXECUTABLE = BRUSH_DIR / "target" / "release" / "brush_app"

# Training parameters
TRAINING_STEPS = 15000  # Default from Brush batch script example
MAX_SPLATS = 5000000    # 5M splats max
SH_DEGREE = 3           # Spherical harmonics degree


# ============================================================================
# Brush Check
# ============================================================================

def check_brush_installed():
    """Check if Brush is installed and executable."""
    print("Checking for Brush installation...")
    
    if not BRUSH_EXECUTABLE.exists():
        print()
        print("ERROR: Brush not found!")
        print()
        print(f"Expected location: {BRUSH_EXECUTABLE}")
        print()
        print("Please run setup_brush.py first:")
        print("  python setup_brush.py")
        print()
        return False
    
    # Test that Brush can actually run
    try:
        result = subprocess.run(
            [str(BRUSH_EXECUTABLE), "--help"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
        if result.returncode != 0:
            print()
            print("ERROR: Brush executable found but failed to run!")
            print(f"Error: {result.stderr}")
            print()
            return False
    except subprocess.TimeoutExpired:
        print()
        print("ERROR: Brush executable timed out!")
        print()
        return False
    except Exception as e:
        print()
        print(f"ERROR: Failed to test Brush executable: {e}")
        print()
        return False
    
    print(f"✓ Brush found and working at {BRUSH_EXECUTABLE}")
    return True


# ============================================================================
# COLMAP Validation
# ============================================================================

def validate_colmap_output():
    """Check that COLMAP processing completed successfully."""
    print("Validating COLMAP output...")
    
    # Check for images directory
    if not IMAGES_DIR.exists():
        print(f"✗ Images directory not found: {IMAGES_DIR}")
        print()
        print("Please ensure images are in: financial_district_images/images/")
        return False
    
    # Count images
    image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    if len(image_files) == 0:
        print(f"✗ No images found in {IMAGES_DIR}")
        print()
        return False
    
    print(f"  Found {len(image_files)} images")
    
    # Check for sparse reconstruction
    if not COLMAP_SPARSE_DIR.exists():
        print(f"✗ Sparse reconstruction directory not found: {COLMAP_SPARSE_DIR}")
        print()
        print("Please download COLMAP output from Lambda first.")
        return False
    
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    
    missing_files = []
    for filename in required_files:
        filepath = COLMAP_SPARSE_DIR / filename
        if not filepath.exists():
            missing_files.append(filename)
        else:
            # Check file is not empty
            if filepath.stat().st_size == 0:
                print(f"✗ {filename} is empty (0 bytes)")
                print()
                print("COLMAP processing may not be complete. Please wait for it to finish.")
                return False
    
    if missing_files:
        print(f"✗ Missing COLMAP files in {sparse_dir}:")
        for filename in missing_files:
            print(f"  - {filename}")
        print()
        print("COLMAP processing may not be complete. Please wait for it to finish.")
        return False
    
    print(f"✓ COLMAP output validated at {COLMAP_SPARSE_DIR}")
    return True


# ============================================================================
# Brush Training
# ============================================================================

def run_brush_training():
    """Run Brush CLI training on COLMAP data."""
    print()
    print("="*70)
    print("BRUSH TRAINING")
    print("="*70)
    print()
    print(f"Input:  {COLMAP_SPARSE_DIR}")
    print(f"Output: {BRUSH_OUTPUT_DIR}")
    print()
    print(f"Training steps: {TRAINING_STEPS}")
    print(f"Max splats:     {MAX_SPLATS:,}")
    print(f"SH degree:      {SH_DEGREE}")
    print()
    
    # Create output directory
    BRUSH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build Brush command
    # Format: brush [OPTIONS] <DATA_PATH>
    # DATA_PATH should point to directory containing sparse/ folder
    colmap_root = COLMAP_SPARSE_DIR.parent.parent  # Go up from sparse/0 to colmap_output root
    cmd = [
        str(BRUSH_EXECUTABLE),
        "--total-steps", str(TRAINING_STEPS),
        "--max-splats", str(MAX_SPLATS),
        "--sh-degree", str(SH_DEGREE),
        str(colmap_root),
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    print("-"*70)
    print()
    
    # Run training
    result = subprocess.run(
        cmd,
        cwd=BRUSH_OUTPUT_DIR,
        text=True
    )
    
    if result.returncode != 0:
        print()
        print("ERROR: Brush training failed!")
        return False
    
    print()
    print("="*70)
    print("✓ Brush training completed successfully!")
    print("="*70)
    return True


def save_training_info():
    """Save training information to JSON."""
    info = {
        "colmap_dir": str(COLMAP_SPARSE_DIR),
        "brush_output_dir": str(BRUSH_OUTPUT_DIR),
        "training_steps": TRAINING_STEPS,
        "max_splats": MAX_SPLATS,
        "sh_degree": SH_DEGREE,
        "brush_executable": str(BRUSH_EXECUTABLE),
    }
    
    with open(TRAINING_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)
    
    print()
    print(f"Training info saved to: {TRAINING_INFO_PATH}")


# ============================================================================
# Main
# ============================================================================

def main():
    print()
    print("="*70)
    print("BRUSH CLI TRAINING FOR MAC")
    print("="*70)
    print()
    print("PRE-FLIGHT CHECKS")
    print("-"*70)
    print()
    
    # Check if Brush is installed
    if not check_brush_installed():
        return 1
    
    print()
    
    # Validate COLMAP output exists
    if not validate_colmap_output():
        return 1
    
    print()
    
    # Test output directory permissions early
    print("Testing output directory permissions...")
    try:
        BRUSH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        test_file = BRUSH_OUTPUT_DIR / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        print(f"✓ Output directory writable: {BRUSH_OUTPUT_DIR}")
    except Exception as e:
        print()
        print(f"ERROR: Cannot write to output directory {BRUSH_OUTPUT_DIR}")
        print(f"Error: {e}")
        print()
        return 1
    
    print()
    print("="*70)
    print("ALL PRE-FLIGHT CHECKS PASSED!")
    print("="*70)
    print()
    print("Starting training...")
    print()
    
    # Run Brush training
    if not run_brush_training():
        return 1
    
    # Save training info
    save_training_info()
    
    print()
    print("="*70)
    print("ALL DONE!")
    print("="*70)
    print()
    print("Output files are in:")
    print(f"  {BRUSH_OUTPUT_DIR}")
    print()
    print("To view your trained model:")
    print(f"  {BRUSH_EXECUTABLE} {BRUSH_OUTPUT_DIR}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
