#!/usr/bin/env python3
"""
Gaussian Splatting Training on Mac using Brush
Single-file training script for Financial District dataset.

Requirements:
    - Brush installed: https://github.com/ArthurBrussee/brush
    - COLMAP installed: brew install colmap
    - imagemagick installed: brew install imagemagick

Run:
    python3 train_gaussian_mac.py
"""

import os
import subprocess
import sys
from pathlib import Path
import json

# Configuration
IMAGES_DIR = Path(__file__).parent / "financial_district" / "images"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "gaussian_splatting"
COLMAP_DIR = OUTPUT_DIR / "colmap"
SPLAT_OUTPUT = OUTPUT_DIR / "splat.ply"
TRAINING_STEPS = 7000  # ~4 hours on Mac

# Brush training parameters
BRUSH_PARAMS = {
    "steps": TRAINING_STEPS,
    "eval_every": 500,
    "save_every": 1000,
}


def check_dependencies():
    """Check if required tools are installed."""
    print("🔍 Checking dependencies...")
    
    # Different commands have different ways to check if they exist
    checks = {
        "colmap": (["colmap", "help"], "brew install colmap"),
        "convert": (["convert", "--version"], "brew install imagemagick"),
    }
    
    missing = []
    for cmd, (check_cmd, install_cmd) in checks.items():
        try:
            subprocess.run(
                check_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            print(f"  ✅ {cmd} found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  ❌ {cmd} not found")
            print(f"     Install with: {install_cmd}")
            missing.append(cmd)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("Please install them and try again.")
        return False
    
    print("✅ All dependencies found\n")
    return True


def check_images():
    """Verify images exist."""
    if not IMAGES_DIR.exists():
        print(f"❌ Images directory not found: {IMAGES_DIR}")
        return False
    
    images = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    if len(images) == 0:
        print(f"❌ No images found in {IMAGES_DIR}")
        return False
    
    print(f"✅ Found {len(images)} images in {IMAGES_DIR}\n")
    return True


def run_colmap():
    """Run COLMAP structure from motion."""
    print("=" * 70)
    print("Step 1: Running COLMAP (Structure from Motion)")
    print("=" * 70)
    print(f"Images: {IMAGES_DIR}")
    print(f"Output: {COLMAP_DIR}")
    print("\nThis may take 30-60 minutes...")
    print("=" * 70)
    print()
    
    COLMAP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run COLMAP automatic reconstructor
    # Note: Mapillary images come from different cameras/users, so we let COLMAP
    # detect multiple camera models automatically (no --single_camera flag)
    cmd = [
        "colmap", "automatic_reconstructor",
        "--workspace_path", str(COLMAP_DIR),
        "--image_path", str(IMAGES_DIR),
        "--quality", "high",
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ COLMAP reconstruction complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ COLMAP failed with error code {e.returncode}")
        print("Check if images have enough overlap and features")
        return False


def prepare_brush_dataset():
    """Prepare dataset for Brush training."""
    print("\n" + "=" * 70)
    print("Step 2: Preparing dataset for Brush")
    print("=" * 70)
    
    # Brush expects COLMAP output in specific format
    # The automatic_reconstructor should have created sparse/0/
    sparse_dir = COLMAP_DIR / "sparse" / "0"
    
    if not sparse_dir.exists():
        print(f"❌ COLMAP sparse reconstruction not found at {sparse_dir}")
        return False
    
    # Check for required COLMAP files
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    for f in required_files:
        if not (sparse_dir / f).exists():
            print(f"❌ Missing COLMAP file: {f}")
            return False
    
    print("✅ COLMAP sparse reconstruction ready")
    print(f"   Location: {sparse_dir}")
    
    return True


def train_gaussian_splat():
    """Train Gaussian Splatting with Brush."""
    print("\n" + "=" * 70)
    print("Step 3: Training Gaussian Splatting")
    print("=" * 70)
    print(f"Steps: {TRAINING_STEPS}")
    print(f"Expected time: ~4 hours on Mac")
    print(f"Output: {SPLAT_OUTPUT}")
    print("=" * 70)
    print()
    
    # NOTE: Brush doesn't have a Python CLI, it's a GUI application
    # This is a placeholder for the actual Brush workflow
    # You'll need to:
    # 1. Open Brush GUI
    # 2. Load COLMAP dataset from COLMAP_DIR
    # 3. Set training steps to TRAINING_STEPS
    # 4. Start training
    
    print("⚠️  MANUAL STEP REQUIRED:")
    print()
    print("Brush is a GUI application. Please:")
    print(f"1. Open Brush")
    print(f"2. File > Import > COLMAP Dataset")
    print(f"3. Select: {COLMAP_DIR}")
    print(f"4. Set training steps: {TRAINING_STEPS}")
    print(f"5. Click 'Train'")
    print(f"6. Save output to: {OUTPUT_DIR}")
    print()
    print("Training will take approximately 4 hours on Mac.")
    print()
    
    # Save training info for reference
    info_file = OUTPUT_DIR / "training_info.json"
    info = {
        "images_dir": str(IMAGES_DIR),
        "colmap_dir": str(COLMAP_DIR),
        "training_steps": TRAINING_STEPS,
        "estimated_time_hours": 4,
        "hardware": "Mac",
        "method": "Gaussian Splatting (Brush)",
        "notes": "Mapillary images from multiple cameras/users - COLMAP auto-detects camera models"
    }
    
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"📝 Training info saved to: {info_file}")
    
    return True


def main():
    """Main training pipeline."""
    print("\n" + "=" * 70)
    print("Gaussian Splatting Training - Financial District SF")
    print("=" * 70)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check images
    if not check_images():
        sys.exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: COLMAP
    if not run_colmap():
        print("\n❌ Training failed at COLMAP step")
        sys.exit(1)
    
    # Step 2: Prepare for Brush
    if not prepare_brush_dataset():
        print("\n❌ Training failed at dataset preparation")
        sys.exit(1)
    
    # Step 3: Train (manual in Brush GUI)
    train_gaussian_splat()
    
    print("\n" + "=" * 70)
    print("✅ Setup Complete!")
    print("=" * 70)
    print()
    print("Next: Open Brush and train the model manually (see instructions above)")
    print()


if __name__ == "__main__":
    main()
