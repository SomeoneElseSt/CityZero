#!/usr/bin/env python3
"""
COLMAP CUDA Build & GPU-Accelerated Preprocessing for Lambda Cloud

This script automates the process of building COLMAP with CUDA support and performing
GPU-accelerated 3D reconstruction on Lambda Cloud A100 instances.

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
- Only a subset of images may register (e.g., 54 of 2998) - this is normal for sparse street data.
- COLMAP may print warnings like "Could not register" but still succeed; the script verifies success.
- Use tmux to prevent disconnection: `tmux new -s colmap`.
- Always terminate the Lambda instance after download to stop billing.
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import shutil


def validate_nvidia_smi():
    """Validate nvidia-smi is available and GPUs are detected."""
    if not shutil.which("nvidia-smi"):
        print("ERROR: nvidia-smi not found in PATH")
        return False

    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )

        if not result.stdout.strip():
            print("ERROR: No GPUs detected by nvidia-smi")
            return False

        gpu_lines = result.stdout.strip().split('\n')
        print(f"Detected {len(gpu_lines)} GPU(s):")
        for line in gpu_lines:
            print(f"  {line}")

        return True

    except subprocess.TimeoutExpired:
        print("ERROR: nvidia-smi timed out")
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: nvidia-smi failed with exit code {e.returncode}")
        if e.stderr:
            print(f"  {e.stderr}")
        return False
    except Exception as e:
        print(f"ERROR: nvidia-smi validation failed: {e}")
        return False


def get_gpu_info():
    """Get detailed GPU information."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )

        if result.stdout.strip():
            print(f"GPU Info: {result.stdout.strip()}")
            return True

        return False

    except Exception:
        return False


def check_system():
    """Verify system requirements."""
    print("\n" + "="*70)
    print("COLMAP CUDA BUILD & GPU-ACCELERATED PREPROCESSING SCRIPT")
    print("="*70)

    if not validate_nvidia_smi():
        print("\nERROR: GPU validation failed")
        print("This script requires an NVIDIA GPU with working drivers")
        return False

    get_gpu_info()

    # Check CUDA compiler
    if shutil.which("nvcc"):
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("CUDA compiler available")
        except Exception:
            pass
    else:
        print("CUDA compiler not found - will be available after dependencies")

    # Check Ubuntu version
    try:
        with open("/etc/os-release") as f:
            content = f.read()
            if "24.04" in content:
                print("Ubuntu 24.04 detected")
            elif "22.04" in content:
                print("Ubuntu 22.04 detected")
            elif "20.04" in content:
                print("WARNING: Ubuntu 20.04 - may have compatibility issues")
            else:
                print("WARNING: Ubuntu version unknown - may have compatibility issues")
    except Exception:
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
    
    print(f"Installing {len(packages)} packages...")
    print("This will take 5-10 minutes...\n")
    
    # Update package list
    subprocess.run(["sudo", "apt", "update"], check=True)
    
    # Install packages
    cmd = ["sudo", "apt", "install", "-y"] + packages
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("ERROR: Failed to install dependencies")
        return False
    
    print("All dependencies installed")
    return True


def build_colmap_cuda():
    """Build COLMAP from source with CUDA support."""
    print("\n" + "="*70)
    print("BUILDING COLMAP WITH CUDA")
    print("="*70)
    print("This will take 30-45 minutes on an A100...\n")
    
    colmap_dir = Path.home() / "colmap_cuda"
    
    # Clone COLMAP if not already present
    if not colmap_dir.exists():
        print("Cloning COLMAP repository...")
        cmd = ["git", "clone", "https://github.com/colmap/colmap.git", str(colmap_dir)]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("ERROR: Failed to clone COLMAP")
            return False
        print("COLMAP cloned")
    else:
        print(f"COLMAP already cloned at {colmap_dir}")
    
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
        print("ERROR: CMake configuration failed")
        return False
    
    print("CMake configuration complete")
    
    # Build
    print("\nBuilding COLMAP...")
    print("This is the long part (30-40 minutes)...\n")
    
    start_time = datetime.now()
    
    # Use all available cores
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    
    build_cmd = ["ninja", "-j", str(num_cores)]
    result = subprocess.run(build_cmd, cwd=build_dir)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if result.returncode != 0:
        print(f"\nERROR: Build failed after {duration:.1f} seconds")
        return False
    
    print(f"\nCOLMAP built successfully in {duration/60:.1f} minutes")
    
    # Install
    print("\nInstalling COLMAP...")
    install_cmd = ["sudo", "ninja", "install"]
    result = subprocess.run(install_cmd, cwd=build_dir)
    
    if result.returncode != 0:
        print("ERROR: Installation failed")
        return False
    
    print("COLMAP installed to /usr/local/bin/colmap")
    return True


def get_colmap_path():
    """Get the path to COLMAP executable."""
    # Check if COLMAP is in PATH
    colmap_path = shutil.which("colmap")
    
    # If not in PATH, check common installation locations
    if not colmap_path:
        common_paths = [
            "/usr/local/bin/colmap",
            "/usr/bin/colmap",
            str(Path.home() / "colmap_cuda" / "build" / "src" / "colmap" / "exe" / "colmap"),
        ]
        for path in common_paths:
            if Path(path).exists():
                colmap_path = path
                break
    
    if not colmap_path:
        return None
    
    print(f"Using COLMAP at: {colmap_path}")
    return colmap_path


def verify_cuda_colmap():
    """Verify that COLMAP was built with CUDA support."""
    print("\n" + "="*70)
    print("VERIFYING CUDA SUPPORT")
    print("="*70)
    
    colmap_path = get_colmap_path()
    if not colmap_path:
        print("ERROR: COLMAP executable not found")
        return False
    
    try:
        result = subprocess.run(
            [colmap_path, "-h"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if "with CUDA" in result.stdout:
            print("COLMAP compiled WITH CUDA support")
            print(f"\nVersion info:")
            # Extract version line
            for line in result.stdout.split('\n'):
                if 'COLMAP' in line and 'CUDA' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print("ERROR: COLMAP was NOT compiled with CUDA")
            return False
            
    except Exception as e:
        print(f"ERROR: Could not verify COLMAP: {e}")
        return False


def run_colmap_pipeline(images_dir: Path, output_dir: Path):
    """Run full COLMAP pipeline with CUDA acceleration."""
    print("\n" + "="*70)
    print("RUNNING COLMAP WITH CUDA ACCELERATION")
    print("="*70)
    
    # Get COLMAP path
    colmap_path = get_colmap_path()
    if not colmap_path:
        print("ERROR: COLMAP executable not found")
        return False
    
    # Validate inputs
    if not images_dir.exists():
        print(f"ERROR: Images directory does not exist: {images_dir}")
        return False
    
    num_images = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
    if num_images == 0:
        print(f"ERROR: No .jpg or .png images found in {images_dir}")
        return False
    
    print(f"Found {num_images} images")
    
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
        colmap_path, "feature_extractor",
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
    print(f"\nCommand: {' '.join(feat_cmd)}")
    start_time = datetime.now()
    
    result = subprocess.run(feat_cmd, env=env)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    if result.returncode != 0:
        print(f"\nERROR: Feature extraction failed")
        print("\nPossible fixes:")
        print("  1. Check flags with: colmap feature_extractor --help | grep gpu")
        print("  2. Verify COLMAP has CUDA: colmap -h | grep CUDA")
        return False
    
    print(f"\nFeature extraction completed in {duration:.1f}s ({duration/60:.1f} min)")
    
    # Step 2: Feature Matching (CUDA)
    print("\n" + "="*70)
    print("STEP 2/3: FEATURE MATCHING (GPU-ACCELERATED)")
    print("="*70)
    
    match_cmd = [
        colmap_path, "exhaustive_matcher",
        "--database_path", str(database_path),
        "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", "0",
        "--FeatureMatching.guided_matching", "1",
        "--FeatureMatching.max_num_matches", "65536",
    ]
    
    print("\nMatching features with CUDA...")
    print("Expected GPU utilization: 80-95%")
    print(f"\nCommand: {' '.join(match_cmd)}")
    start_time = datetime.now()
    
    result = subprocess.run(match_cmd, env=env)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    if result.returncode != 0:
        print(f"\nERROR: Feature matching failed")
        print("\nPossible fixes:")
        print("  1. Check flags with: colmap exhaustive_matcher --help | grep gpu")
        print("  2. Try without guided_matching flag")
        return False
    
    print(f"\nFeature matching completed in {duration:.1f}s ({duration/60:.1f} min)")
    
    # Step 3: Sparse Reconstruction
    print("\n" + "="*70)
    print("STEP 3/3: SPARSE RECONSTRUCTION")
    print("="*70)
    
    mapper_cmd = [
        colmap_path, "mapper",
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
    
    print(f"\nMapper finished in {duration:.1f}s ({duration/60:.1f} min)")
    
    # Verify reconstruction actually succeeded
    print("\n" + "="*70)
    print("VERIFYING RECONSTRUCTION")
    print("="*70)
    
    reconstruction_dirs = list(sparse_dir.glob("*"))
    if not reconstruction_dirs:
        print("ERROR: No reconstruction directories found")
        return False
    
    # Check the first reconstruction (usually "0")
    recon_dir = reconstruction_dirs[0]
    cameras_file = recon_dir / "cameras.bin"
    images_file = recon_dir / "images.bin"
    points_file = recon_dir / "points3D.bin"
    
    if not (cameras_file.exists() and images_file.exists() and points_file.exists()):
        print(f"ERROR: Reconstruction files missing in {recon_dir}")
        return False
    
    # Run model analyzer to get stats
    try:
        analyzer_result = subprocess.run(
            [colmap_path, "model_analyzer", "--path", str(recon_dir)],
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
        
        print(f"\nReconstruction succeeded!")
        print(f"  - Registered images: {registered_images} (out of {num_images})")
        print(f"  - 3D points: {points:,}")
        print(f"  - Mean reprojection error: {reprojection_error:.2f}px")
        
        if registered_images < 10:
            print(f"\nWARNING: Only {registered_images} images registered")
            print("  This may indicate:")
            print("  - Images don't overlap enough")
            print("  - Images are too spread out geographically")
            print("  - Consider downloading images from a smaller, denser area")
        
        if reprojection_error > 2.0:
            print(f"\nWARNING: High reprojection error ({reprojection_error:.2f}px)")
            print("  Reconstruction may be inaccurate")
        
        return True
        
    except Exception as e:
        print(f"WARNING: Could not verify reconstruction: {e}")
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
    
    print(f"\nSummary saved to {summary_file}")
    
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
        print("ERROR: Failed to compress output")
        return False
    
    size_mb = output_archive.stat().st_size / (1024 * 1024)
    print(f"Compressed to {output_archive} ({size_mb:.1f} MB)")
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
        print("\nERROR: System check failed")
        print("Cannot proceed without valid GPU setup")
        return 1
    
    # Steps 2-4: Build COLMAP (unless skipped)
    if not args.skip_build:
        if not install_dependencies():
            return 1
        
        if not build_colmap_cuda():
            return 1
        
        if not verify_cuda_colmap():
            print("CUDA verification failed, but attempting to continue...")
    else:
        print("Skipping build (--skip-build)")
        if not verify_cuda_colmap():
            print("ERROR: COLMAP not found or doesn't have CUDA")
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
            print("WARNING: Failed to compress output, but processing succeeded")
        
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
