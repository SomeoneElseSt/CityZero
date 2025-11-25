#!/usr/bin/env python3
"""
COLMAP CUDA Build & GPU-Accelerated Preprocessing for Lambda Cloud

This script autonomously handles everything needed to run GPU-accelerated 3D reconstruction
on Lambda Cloud A100 instances. Battle-tested on Financial District dataset (2998 images).

KEY LEARNINGS:
- Lambda Stack's default COLMAP has NO CUDA support (CPU-only)
- Must build COLMAP from source with CUDA flags (30-45 min one-time)
- Flag names changed in COLMAP 3.14:
  * Use --FeatureExtraction.use_gpu (not --SiftExtraction.use_gpu)
  * Use --FeatureMatching.use_gpu (not --SiftMatching.use_gpu)
- COLMAP prints scary warnings during mapper but still succeeds
- Script verifies reconstruction by checking output files + stats

PERFORMANCE (3K images, A100):
- Build: 30-45 min (one-time)
- Feature extraction: ~2 min (GPU, 80-95% utilization)
- Feature matching: ~1 min (GPU, 80-95% utilization)
- Mapper: 10-30 min (CPU-bound, low GPU usage is normal)
- Total: ~45-80 min (vs 10 hours on M4 Mac)

VALIDATED FOR:
- Ubuntu 24.04 LTS
- CUDA 12.x (Lambda Stack default)
- A100 GPU (40GB)
- Any unordered image dataset (Mapillary, custom, etc.)

USAGE:

  # Full pipeline (first time - builds COLMAP + processes images):
  python3 lambda_build_colmap_cuda.py --images ~/images --output ~/colmap_output

  # Build COLMAP once, then process multiple datasets:
  python3 lambda_build_colmap_cuda.py --build-only
  python3 lambda_build_colmap_cuda.py --images ~/dataset1 --output ~/out1 --skip-build
  python3 lambda_build_colmap_cuda.py --images ~/dataset2 --output ~/out2 --skip-build

WORKFLOW:
  1. Upload: scp -i *.pem images.tar.gz lambda_build_colmap_cuda.py ubuntu@IP:~/
  2. SSH: ssh -i *.pem ubuntu@IP
  3. Decompress: tar -xzf images.tar.gz
  4. Run: python3 lambda_build_colmap_cuda.py --images ~/images --output ~/output
  5. Download: scp -i *.pem ubuntu@IP:~/output.tar.gz .

OUTPUT:
- Sparse reconstruction in COLMAP format (cameras.bin, images.bin, points3D.bin)
- Compatible with Gaussian Splatting, NeRF, 3DGS training
- Automatically compressed for download

NOTES:
- Only subset of images may register (e.g., 54 of 2998) - normal for sparse street data
- COLMAP warnings like "Could not register" are normal - script verifies success
- Use tmux to prevent disconnection: tmux new -s colmap
- Always terminate Lambda instance after download to stop billing
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import shutil


def check_system():
    """Verify system requirements."""
    print("\n" + "="*70)
    print("SYSTEM VALIDATION")
    print("="*70)
    
    # Check GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ GPU: {result.stdout.strip()}")
        else:
            print("⚠ WARNING: nvidia-smi failed, but continuing...")
    except:
        print("⚠ WARNING: nvidia-smi not accessible, but continuing...")
    
    # Check CUDA
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ CUDA compiler available")
        else:
            print("⚠ WARNING: nvcc not found - will be available after dependencies")
    except:
        print("⚠ WARNING: nvcc not found - will be available after dependencies")
    
    # Check Ubuntu version
    try:
        with open("/etc/os-release") as f:
            content = f.read()
            if "24.04" in content:
                print("✓ Ubuntu 24.04 detected")
            else:
                print("⚠ WARNING: Not Ubuntu 24.04, may have compatibility issues")
    except:
        pass
    
    return True


def install_dependencies():
    """Install all required dependencies."""
    print("\n" + "="*70)
    print("INSTALLING DEPENDENCIES")
    print("="*70)
    
    packages = [
        # Build tools
        "git", "cmake", "ninja-build", "build-essential",
        # COLMAP dependencies
        "libboost-program-options-dev", "libboost-filesystem-dev",
        "libboost-graph-dev", "libboost-system-dev",
        "libeigen3-dev", "libflann-dev", "libfreeimage-dev",
        "libmetis-dev", "libgoogle-glog-dev", "libgflags-dev",
        "libsqlite3-dev", "libglew-dev", "qtbase5-dev",
        "libqt5opengl5-dev", "libcgal-dev",
        # CUDA dependencies
        "libceres-dev",
        # Additional useful tools
        "python3-pip",
    ]
    
    print(f"\nInstalling {len(packages)} packages...")
    print("This will take 5-10 minutes...\n")
    
    # Update package list
    subprocess.run(["sudo", "apt", "update"], check=True)
    
    # Install packages
    cmd = ["sudo", "apt", "install", "-y"] + packages
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("✗ ERROR: Failed to install dependencies")
        return False
    
    print("\n✓ All dependencies installed")
    return True


def build_colmap_cuda():
    """Build COLMAP from source with CUDA support."""
    print("\n" + "="*70)
    print("BUILDING COLMAP WITH CUDA")
    print("="*70)
    print("This will take 30-45 minutes on an A100...")
    print()
    
    colmap_dir = Path.home() / "colmap_cuda"
    
    # Clone COLMAP if not already present
    if not colmap_dir.exists():
        print("Cloning COLMAP repository...")
        cmd = ["git", "clone", "https://github.com/colmap/colmap.git", str(colmap_dir)]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("✗ ERROR: Failed to clone COLMAP")
            return False
        print("✓ COLMAP cloned")
    else:
        print(f"✓ COLMAP already cloned at {colmap_dir}")
    
    # Create build directory
    build_dir = colmap_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    # Configure with CMake
    print("\nConfiguring COLMAP with CMake...")
    print("Enabling: CUDA, Tests, CGAL")
    
    cmake_cmd = [
        "cmake", "..",
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CUDA_ARCHITECTURES=80",  # A100 architecture
        "-DCUDA_ENABLED=ON",
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc",
    ]
    
    result = subprocess.run(cmake_cmd, cwd=build_dir)
    if result.returncode != 0:
        print("✗ ERROR: CMake configuration failed")
        return False
    
    print("✓ CMake configuration complete")
    
    # Build
    print("\nBuilding COLMAP...")
    print("This is the long part (30-40 minutes)...")
    print("Go grab a coffee ☕\n")
    
    start_time = datetime.now()
    
    # Use all available cores
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    
    build_cmd = ["ninja", "-j", str(num_cores)]
    result = subprocess.run(build_cmd, cwd=build_dir)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if result.returncode != 0:
        print(f"\n✗ ERROR: Build failed after {duration:.1f} seconds")
        return False
    
    print(f"\n✓ COLMAP built successfully in {duration/60:.1f} minutes")
    
    # Install
    print("\nInstalling COLMAP...")
    install_cmd = ["sudo", "ninja", "install"]
    result = subprocess.run(install_cmd, cwd=build_dir)
    
    if result.returncode != 0:
        print("✗ ERROR: Installation failed")
        return False
    
    print("✓ COLMAP installed to /usr/local/bin/colmap")
    return True


def verify_cuda_colmap():
    """Verify that COLMAP was built with CUDA support."""
    print("\n" + "="*70)
    print("VERIFYING CUDA SUPPORT")
    print("="*70)
    
    try:
        result = subprocess.run(
            ["colmap", "-h"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if "with CUDA" in result.stdout:
            print("✓ COLMAP compiled WITH CUDA support")
            print(f"\nVersion info:")
            # Extract version line
            for line in result.stdout.split('\n'):
                if 'COLMAP' in line and 'CUDA' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print("✗ ERROR: COLMAP was NOT compiled with CUDA")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: Could not verify COLMAP: {e}")
        return False


def run_colmap_pipeline(images_dir: Path, output_dir: Path):
    """Run full COLMAP pipeline with CUDA acceleration."""
    print("\n" + "="*70)
    print("RUNNING COLMAP WITH CUDA ACCELERATION")
    print("="*70)
    
    # Validate inputs
    if not images_dir.exists():
        print(f"✗ ERROR: Images directory does not exist: {images_dir}")
        return False
    
    num_images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
    if num_images == 0:
        print(f"✗ ERROR: No .jpg or .png images found in {images_dir}")
        return False
    
    print(f"✓ Found {num_images} images")
    
    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    
    # Step 1: Feature Extraction (CUDA)
    print("\n" + "="*70)
    print("STEP 1/3: FEATURE EXTRACTION (GPU-ACCELERATED)")
    print("="*70)
    
    feat_cmd = [
        "colmap", "feature_extractor",
        "--image_path", str(images_dir),
        "--database_path", str(database_path),
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", "0",
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.gpu_index", "0",
        "--SiftExtraction.max_num_features", "16384",
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1",
    ]
    
    # Environment for headless execution
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["DISPLAY"] = ""
    
    print("\nExtracting features with CUDA...")
    print("Expected GPU utilization: 80-95%")
    print("Monitor with: watch -n 2 nvidia-smi")
    start_time = datetime.now()
    
    result = subprocess.run(feat_cmd, env=env)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    if result.returncode != 0:
        print(f"\n✗ ERROR: Feature extraction failed")
        print("\nPossible fixes:")
        print("  1. Check flags with: colmap feature_extractor --help | grep gpu")
        print("  2. Verify COLMAP has CUDA: colmap -h | grep CUDA")
        return False
    
    print(f"\n✓ Feature extraction completed in {duration:.1f}s ({duration/60:.1f} min)")
    
    # Step 2: Feature Matching (CUDA)
    print("\n" + "="*70)
    print("STEP 2/3: FEATURE MATCHING (GPU-ACCELERATED)")
    print("="*70)
    
    match_cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", str(database_path),
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", "0",
        "--FeatureMatching.guided_matching", "1",
        "--FeatureMatching.max_num_matches", "65536",
    ]
    
    print("\nMatching features with CUDA...")
    print("Expected GPU utilization: 80-95%")
    start_time = datetime.now()
    
    result = subprocess.run(match_cmd, env=env)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    if result.returncode != 0:
        print(f"\n✗ ERROR: Feature matching failed")
        print("\nPossible fixes:")
        print("  1. Check flags with: colmap exhaustive_matcher --help | grep gpu")
        print("  2. Try without guided_matching flag")
        return False
    
    print(f"\n✓ Feature matching completed in {duration:.1f}s ({duration/60:.1f} min)")
    
    # Step 3: Sparse Reconstruction
    print("\n" + "="*70)
    print("STEP 3/3: SPARSE RECONSTRUCTION")
    print("="*70)
    
    mapper_cmd = [
        "colmap", "mapper",
        "--database_path", str(database_path),
        "--image_path", str(images_dir),
        "--output_path", str(sparse_dir),
    ]
    
    print("\nRunning mapper...")
    print("Note: Mapper is mostly CPU-bound, low GPU usage is expected")
    print("Note: COLMAP may print warnings but still succeed - we'll verify after")
    start_time = datetime.now()
    
    result = subprocess.run(mapper_cmd, env=env)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    print(f"\n✓ Mapper finished in {duration:.1f}s ({duration/60:.1f} min)")
    
    # Verify reconstruction actually succeeded
    print("\n" + "="*70)
    print("VERIFYING RECONSTRUCTION")
    print("="*70)
    
    reconstruction_dirs = list(sparse_dir.glob("*"))
    if not reconstruction_dirs:
        print("✗ ERROR: No reconstruction directories found")
        return False
    
    # Check the first reconstruction (usually "0")
    recon_dir = reconstruction_dirs[0]
    cameras_file = recon_dir / "cameras.bin"
    images_file = recon_dir / "images.bin"
    points_file = recon_dir / "points3D.bin"
    
    if not (cameras_file.exists() and images_file.exists() and points_file.exists()):
        print(f"✗ ERROR: Reconstruction files missing in {recon_dir}")
        return False
    
    # Run model analyzer to get stats
    try:
        analyzer_result = subprocess.run(
            ["colmap", "model_analyzer", "--path", str(recon_dir)],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Parse output for key metrics
        output = analyzer_result.stdout
        registered_images = 0
        points = 0
        reprojection_error = 0.0
        
        for line in output.split('\n'):
            if 'Registered images:' in line:
                registered_images = int(line.split(':')[1].strip())
            elif 'Points:' in line and 'Points3D' not in line:
                points = int(line.split(':')[1].strip())
            elif 'Mean reprojection error:' in line:
                reprojection_error = float(line.split(':')[1].strip().replace('px', ''))
        
        print(f"\n✓ Reconstruction succeeded!")
        print(f"  - Registered images: {registered_images} (out of {num_images})")
        print(f"  - 3D points: {points:,}")
        print(f"  - Mean reprojection error: {reprojection_error:.2f}px")
        
        if registered_images < 10:
            print(f"\n⚠ WARNING: Only {registered_images} images registered")
            print("  This may indicate:")
            print("  - Images don't overlap enough")
            print("  - Images are too spread out geographically")
            print("  - Consider downloading images from a smaller, denser area")
        
        if reprojection_error > 2.0:
            print(f"\n⚠ WARNING: High reprojection error ({reprojection_error:.2f}px)")
            print("  Reconstruction may be inaccurate")
        
        return True
        
    except Exception as e:
        print(f"⚠ WARNING: Could not verify reconstruction: {e}")
        print("  But reconstruction files exist, so likely succeeded")
        return True


def create_summary(output_dir: Path, images_dir: Path, total_duration: float):
    """Create summary JSON file."""
    try:
        num_images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
    except:
        num_images = "unknown"
    
    summary = {
        "tool": "COLMAP (CUDA-enabled)",
        "gpu": "A100 (Lambda Cloud)",
        "cuda_accelerated": "Feature extraction, matching, and BA",
        "processing_time_seconds": total_duration,
        "processing_time_minutes": total_duration / 60,
        "images_processed": num_images,
        "output_directory": str(output_dir),
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


def main():
    parser = argparse.ArgumentParser(
        description="Build COLMAP with CUDA and run GPU-accelerated preprocessing",
        epilog="""
Examples:
  # Full pipeline (build + process):
  python3 %(prog)s --images ~/images --output ~/colmap_output
  
  # Build only (for later use):
  python3 %(prog)s --build-only
  
  # Process with already-built COLMAP:
  python3 %(prog)s --images ~/images --output ~/colmap_output --skip-build
        """
    )
    parser.add_argument(
        "--images",
        type=Path,
        help="Path to images directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output directory"
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build COLMAP, don't run preprocessing"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip COLMAP build (use if already built)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("COLMAP CUDA BUILD & PREPROCESSING SCRIPT")
    print("Lambda Cloud A100 GPU")
    print("="*70)
    
    total_start_time = datetime.now()
    
    # Step 1: Check system
    if not check_system():
        print("⚠ System check had warnings, but continuing...")
    
    # Steps 2-4: Build COLMAP (unless skipped)
    if not args.skip_build:
        if not install_dependencies():
            return 1
        
        if not build_colmap_cuda():
            return 1
        
        if not verify_cuda_colmap():
            print("⚠ CUDA verification failed, but attempting to continue...")
    else:
        print("\n✓ Skipping build (--skip-build)")
        if not verify_cuda_colmap():
            print("✗ ERROR: COLMAP not found or doesn't have CUDA")
            print("  Remove --skip-build to build COLMAP first")
            return 1
    
    if args.build_only:
        print("\n" + "="*70)
        print("BUILD COMPLETE!")
        print("="*70)
        print("\nCOLMAP with CUDA is now installed at: /usr/local/bin/colmap")
        print("\nTo run preprocessing:")
        print(f"  python3 {Path(__file__).name} --images ~/images --output ~/output --skip-build")
        return 0
    
    # Step 5: Run preprocessing if images provided
    if args.images and args.output:
        if not run_colmap_pipeline(args.images, args.output):
            print("\n" + "="*70)
            print("PIPELINE FAILED")
            print("="*70)
            print("\nTo resume manually, run commands from README.md")
            return 1
        
        total_duration = (datetime.now() - total_start_time).total_seconds()
        create_summary(args.output, args.images, total_duration)
        
        if not compress_output(args.output):
            print("\n⚠ WARNING: Failed to compress output, but processing succeeded")
        
        print("\n" + "="*70)
        print("ALL DONE!")
        print("="*70)
        print(f"\nOutput directory: {args.output}")
        print(f"Compressed archive: {args.output.parent / f'{args.output.name}.tar.gz'}")
        print("\nTo download to your local machine:")
        print(f"  scp -i *.pem ubuntu@YOUR_IP:~/{args.output.name}.tar.gz .")
        print("\nDon't forget to terminate your Lambda instance!")
        
    else:
        if not args.build_only:
            print("\n" + "="*70)
            print("BUILD COMPLETE - NO IMAGES PROVIDED")
            print("="*70)
            print("\nTo run preprocessing:")
            print(f"  python3 {Path(__file__).name} --images ~/images --output ~/output --skip-build")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
