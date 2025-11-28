#!/usr/bin/env python3
"""
Gaussian Splatting Training Script for Lambda Cloud GPU Instances

This script automates GPU-accelerated 3D Gaussian Splatting training using gsplat
on Lambda Cloud GPU instances.

USAGE:
  # Full pipeline (install dependencies + train):
  python3 lambda_train_gsplat.py --colmap ~/colmap_output --output ~/gsplat_output

  # Train only (skip dependency installation):
  python3 lambda_train_gsplat.py --colmap ~/colmap_output --output ~/gsplat_output --skip-install

  # Custom training parameters:
  python3 lambda_train_gsplat.py --colmap ~/colmap_output --output ~/gsplat_output \
    --iterations 30000 --save-interval 5000 --data-factor 2

WORKFLOW:
  1. Upload: scp -i *.pem colmap_output.tar.gz lambda_train_gsplat.py ubuntu@IP:~/
  2. SSH: ssh -i *.pem ubuntu@IP
  3. Decompress: tar -xzf colmap_output.tar.gz
  4. Run: python3 lambda_train_gsplat.py --colmap ~/colmap_output --output ~/gsplat_output
  5. Download: scp -i *.pem ubuntu@IP:~/gsplat_output.tar.gz .

INPUT:
- COLMAP sparse reconstruction (cameras.bin, images.bin, points3D.bin)
- Original images used for COLMAP

OUTPUT:
- Trained Gaussian Splatting model (.ply files at checkpoints)
- Final .ply file compatible with Brush viewer and other splat viewers
- Training checkpoints for resuming if interrupted

NOTES:
- gsplat uses 4x less GPU memory than original 3DGS implementation
- Training is faster than original implementation on Lambda GPUs
- All .ply checkpoints are saved for quality comparison
- Use tmux to prevent disconnection: `tmux new -s gsplat`
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
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
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
    print("GAUSSIAN SPLATTING GPU TRAINING SCRIPT")
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
                version_line = [l for l in result.stdout.split('\n') if 'release' in l.lower()]
                if version_line:
                    print(f"CUDA compiler: {version_line[0].strip()}")
        except Exception:
            pass
    else:
        print("CUDA compiler not found - will be installed with dependencies")

    # Check Python version
    import sys
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python version: {py_version}")

    if sys.version_info < (3, 8):
        print("WARNING: Python 3.8+ recommended for gsplat")

    # Check Ubuntu version
    try:
        with open("/etc/os-release") as f:
            content = f.read()
            if "24.04" in content:
                print("Ubuntu 24.04 detected")
            elif "22.04" in content:
                print("Ubuntu 22.04 detected")
            elif "20.04" in content:
                print("Ubuntu 20.04 detected")
            else:
                print("Ubuntu version: Unknown")
    except Exception:
        pass

    return True


def install_dependencies():
    """Install Python dependencies including PyTorch and gsplat."""
    print("\n" + "="*70)
    print("INSTALLING DEPENDENCIES")
    print("="*70)

    # Update system packages
    print("Updating system package list...")
    subprocess.run(["sudo", "apt", "update"], check=True)

    # Install system dependencies
    system_packages = ["python3-pip", "python3-dev", "git"]
    print(f"Installing system packages: {', '.join(system_packages)}")
    subprocess.run(["sudo", "apt", "install", "-y"] + system_packages, check=True)

    # Upgrade pip
    print("\nUpgrading pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    # Install PyTorch with CUDA support
    print("\n" + "-"*70)
    print("INSTALLING PYTORCH WITH CUDA")
    print("-"*70)
    print("This will take 5-10 minutes...")

    # Detect CUDA version
    cuda_version = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            cuda_version = result.stdout.strip().split('.')[0]
            print(f"Detected CUDA version: {cuda_version}")
    except Exception:
        pass

    # Remove system torchvision if present (conflicts with pip version)
    print("Checking for system torchvision package...")
    system_torchvision = Path("/usr/lib/python3/dist-packages/torchvision")
    if system_torchvision.exists():
        print("Removing system torchvision directory (conflicts with pip version)...")
        subprocess.run(["sudo", "rm", "-rf", "/usr/lib/python3/dist-packages/torchvision*"], check=False)
    
    apt_check = subprocess.run(["dpkg", "-l", "python3-torchvision"], capture_output=True)
    if apt_check.returncode == 0:
        print("Removing system torchvision package...")
        subprocess.run(["sudo", "apt", "remove", "-y", "python3-torchvision"], check=False)
    
    # Install PyTorch (use CUDA 12.1 by default for Lambda GPUs)
    torch_cmd = [
        sys.executable, "-m", "pip", "install", "--upgrade",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]

    result = subprocess.run(torch_cmd)
    if result.returncode != 0:
        print("ERROR: Failed to install PyTorch")
        return False

    print("PyTorch installed successfully")

    # Verify PyTorch CUDA availability
    print("\nVerifying PyTorch CUDA support...")
    verify_cmd = [
        sys.executable, "-c",
        "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else None"
    ]
    result = subprocess.run(verify_cmd)
    if result.returncode != 0:
        print("WARNING: Could not verify PyTorch CUDA support")

    # Install gsplat
    print("\n" + "-"*70)
    print("INSTALLING GSPLAT")
    print("-"*70)
    print("Installing gsplat library (CUDA kernels will compile on first use)...")

    # Install numpy 1.x first (pycolmap incompatible with numpy 2.0)
    print("Installing NumPy 1.x (required for pycolmap)...")
    numpy_cmd = [sys.executable, "-m", "pip", "install", "numpy<2.0"]
    subprocess.run(numpy_cmd, check=True)
    
    gsplat_packages = [
        "gsplat",
        "tqdm",           # Progress bars
        "Pillow",         # Image processing
        "scipy",          # Scientific computing
        "imageio",        # Image I/O for training script
        "tyro",           # CLI argument parsing
        "opencv-python",  # cv2 for dataset loading
        "torchmetrics",  # Metrics for torch
    ]

    pip_cmd = [sys.executable, "-m", "pip", "install"] + gsplat_packages
    result = subprocess.run(pip_cmd)

    if result.returncode != 0:
        print("ERROR: Failed to install gsplat")
        return False

    print("\nAll dependencies installed successfully")

    # Clone gsplat repository for training examples
    print("\n" + "-"*70)
    print("CLONING GSPLAT REPOSITORY")
    print("-"*70)

    gsplat_repo = Path.home() / "gsplat"

    if not gsplat_repo.exists():
        print("Cloning gsplat repository for training scripts...")
        cmd = ["git", "clone", "https://github.com/nerfstudio-project/gsplat.git", str(gsplat_repo)]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("ERROR: Failed to clone gsplat repository")
            return False
        print("gsplat repository cloned")
    else:
        print(f"gsplat repository already exists at {gsplat_repo}")

    # Install example requirements
    print("\nInstalling example requirements...")
    requirements_file = gsplat_repo / "examples" / "requirements.txt"
    if requirements_file.exists():
        # Ensure numpy 1.x is installed first
        print("Ensuring NumPy 1.x before pycolmap...")
        numpy_reinstall_cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", "numpy<2.0"]
        subprocess.run(numpy_reinstall_cmd, check=True)
        
        # Install specific pycolmap version (required for SceneManager)
        print("Installing pycolmap from rmbrualla fork...")
        pycolmap_cmd = [
            sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
            "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"
        ]
        subprocess.run(pycolmap_cmd, check=True)
        
        # Then install other requirements
        print("Installing remaining packages from requirements.txt...")
        normal_cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        result = subprocess.run(normal_cmd, check=False)
        
        if result.returncode != 0:
            print("\nSome packages failed, attempting to install fused-ssim with --no-build-isolation...")
            # Install fused-ssim specifically with --no-build-isolation
            fused_cmd = [
                sys.executable, "-m", "pip", "install", "--no-build-isolation",
                "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"
            ]
            subprocess.run(fused_cmd, check=True)
            print("fused-ssim installed")
        
        print("Example requirements installed")

    return True


def validate_colmap_input(colmap_dir: Path, images_dir: Path = None):
    """Validate COLMAP output structure."""
    print("\n" + "="*70)
    print("VALIDATING COLMAP INPUT")
    print("="*70)

    if not colmap_dir.exists():
        print(f"ERROR: COLMAP directory does not exist: {colmap_dir}")
        return False

    # Check for sparse reconstruction
    sparse_dir = colmap_dir / "sparse"
    if not sparse_dir.exists():
        print(f"ERROR: sparse/ directory not found in {colmap_dir}")
        print("Expected structure: colmap_output/sparse/0/")
        return False

    # Find reconstruction directory (usually "0")
    recon_dirs = list(sparse_dir.glob("*"))
    if not recon_dirs:
        print(f"ERROR: No reconstruction found in {sparse_dir}")
        return False

    recon_dir = recon_dirs[0]
    print(f"Found reconstruction at: {recon_dir}")

    # Check required files
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    missing_files = []

    for filename in required_files:
        filepath = recon_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
        else:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  {filename}: {size_mb:.2f} MB")

    if missing_files:
        print(f"\nERROR: Missing required files: {', '.join(missing_files)}")
        return False

    # Check for images directory
    if images_dir is None:
        images_dir = colmap_dir / "images"
    
    if not images_dir.exists():
        print(f"\nERROR: Images directory not found: {images_dir}")
        print("Specify images location with --images flag")
        return False

    # Count images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.JPG")) + list(images_dir.glob("*.PNG"))
    if len(image_files) == 0:
        print(f"\nERROR: No images found in {images_dir}")
        return False

    print(f"\nFound {len(image_files)} images in {images_dir}")
    print("\nCOLMAP input validated successfully")
    return True


def run_gsplat_training(
    colmap_dir: Path,
    output_dir: Path,
    iterations: int = 30000,
    save_interval: int = 5000,
    data_factor: int = 4
):
    """Run gsplat training on COLMAP data."""
    print("\n" + "="*70)
    print("GAUSSIAN SPLATTING TRAINING")
    print("="*70)
    print()
    print(f"Input COLMAP:  {colmap_dir}")
    print(f"Output:        {output_dir}")
    print(f"Iterations:    {iterations:,}")
    print(f"Save interval: {save_interval:,}")
    print(f"Data factor:   {data_factor}x downsampling")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get gsplat training script
    gsplat_repo = Path.home() / "gsplat"
    training_script = gsplat_repo / "examples" / "simple_trainer.py"

    if not training_script.exists():
        print(f"ERROR: Training script not found at {training_script}")
        return False

    # Build training command
    # gsplat expects data_dir to contain: images/ and sparse/ folders
    cmd = [
        sys.executable,
        str(training_script),
        "default",  # Use default config
        "--data_dir", str(colmap_dir),
        "--result_dir", str(output_dir),
        "--data_factor", str(data_factor),
        "--max_steps", str(iterations),
        "--save_ply",  # Enable .ply export
        "--disable_viewer",  # Headless mode for Lambda
        "--eval_steps", "-1",  # Disable evaluation during training
    ]

    # Add checkpoint saving at intervals
    # Note: gsplat saves checkpoints automatically, but we can specify ply save intervals
    if save_interval > 0:
        # Calculate save steps
        save_steps = list(range(save_interval, iterations + 1, save_interval))
        if iterations not in save_steps:
            save_steps.append(iterations)

        # gsplat will save .ply at these steps
        cmd.extend(["--ply_steps"] + [str(s) for s in save_steps])

    print("Running gsplat training...")
    print("Command:")
    print(" ".join(cmd))
    print()
    print("-"*70)
    print()

    # Set environment variables
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU

    # Run training
    start_time = datetime.now()

    result = subprocess.run(cmd, env=env)

    duration = (datetime.now() - start_time).total_seconds()

    if result.returncode != 0:
        print()
        print("="*70)
        print("TRAINING FAILED")
        print("="*70)
        print(f"Training failed after {duration/60:.1f} minutes")
        return False

    print()
    print("="*70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Training time: {duration/60:.1f} minutes ({duration/3600:.1f} hours)")

    # Verify output files
    print("\nVerifying output files...")
    ply_dir = output_dir / "ply"
    if ply_dir.exists():
        ply_files = sorted(ply_dir.glob("*.ply"))
        print(f"Found {len(ply_files)} .ply checkpoint files:")
        for ply_file in ply_files:
            size_mb = ply_file.stat().st_size / (1024 * 1024)
            print(f"  {ply_file.name}: {size_mb:.1f} MB")
    else:
        print("WARNING: No .ply files found in output")

    return True


def create_summary(output_dir: Path, colmap_dir: Path, total_duration: float, iterations: int):
    """Create summary JSON file."""
    try:
        images_dir = colmap_dir / "images"
        num_images = len(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    except:
        num_images = "unknown"

    # Count output .ply files
    ply_dir = output_dir / "ply"
    ply_files = sorted(ply_dir.glob("*.ply")) if ply_dir.exists() else []

    summary = {
        "tool": "gsplat (GPU-accelerated Gaussian Splatting)",
        "gpu": "Lambda Cloud GPU Instance",
        "training_iterations": iterations,
        "processing_time_seconds": total_duration,
        "processing_time_minutes": total_duration / 60,
        "processing_time_hours": total_duration / 3600,
        "images_used": num_images,
        "colmap_input": str(colmap_dir),
        "output_directory": str(output_dir),
        "ply_checkpoints": len(ply_files),
        "ply_files": [f.name for f in ply_files],
        "timestamp": datetime.now().isoformat(),
    }

    summary_file = output_dir / "training_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_file}")

    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for key, value in summary.items():
        if key != "ply_files":  # Don't print all filenames
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
        description="GPU-accelerated Gaussian Splatting training using gsplat on Lambda Cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (install + train):
  python3 %(prog)s --colmap ~/colmap_output --output ~/gsplat_output

  # Skip installation (if already done):
  python3 %(prog)s --colmap ~/colmap_output --output ~/gsplat_output --skip-install

  # Custom training parameters:
  python3 %(prog)s --colmap ~/colmap_output --output ~/gsplat_output \\
    --iterations 30000 --save-interval 5000 --data-factor 2

Notes:
  - COLMAP directory should contain: images/ and sparse/0/ folders
  - gsplat uses 4x less GPU memory than original 3DGS
  - Training is faster than original 3DGS on Lambda GPUs
  - All checkpoint .ply files are saved for comparison
        """
    )
    parser.add_argument(
        "--colmap",
        type=Path,
        required=True,
        help="Path to COLMAP output directory (containing sparse/ reconstruction)"
    )
    parser.add_argument(
        "--images",
        type=Path,
        help="Path to images directory (if not in COLMAP directory)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output directory for trained model"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Number of training iterations (default: 30000)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="Save .ply checkpoint every N iterations (default: 5000, 0 to disable)"
    )
    parser.add_argument(
        "--data-factor",
        type=int,
        default=4,
        help="Downsample images by this factor (default: 4, 1=full resolution)"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation (use if already installed)"
    )

    args = parser.parse_args()

    print("="*70)
    print("GAUSSIAN SPLATTING GPU TRAINING SCRIPT")
    print("Lambda Cloud GPU Instance")
    print("="*70)

    total_start_time = datetime.now()

    # Step 1: Check system
    if not check_system():
        print("\nERROR: System check failed")
        print("Cannot proceed without valid GPU setup")
        return 1

    # Step 2: Install dependencies (unless skipped)
    if not args.skip_install:
        if not install_dependencies():
            return 1
    else:
        print("\nSkipping dependency installation (--skip-install)")

    # Step 3: Validate COLMAP input
    if not validate_colmap_input(args.colmap, args.images):
        return 1
    
    # Create symlink if images are in separate directory
    images_dir = args.images if args.images else args.colmap / "images"
    if args.images and args.images != args.colmap / "images":
        symlink_path = args.colmap / "images"
        if not symlink_path.exists():
            print(f"\nCreating symlink: {symlink_path} -> {images_dir}")
            symlink_path.symlink_to(images_dir.resolve())

    # Step 4: Run training
    if not run_gsplat_training(
        args.colmap,
        args.output,
        args.iterations,
        args.save_interval,
        args.data_factor
    ):
        print("\n" + "="*70)
        print("TRAINING FAILED")
        print("="*70)
        return 1

    total_duration = (datetime.now() - total_start_time).total_seconds()
    create_summary(args.output, args.colmap, total_duration, args.iterations)

    if not compress_output(args.output):
        print("WARNING: Failed to compress output, but training succeeded")

    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"\nOutput directory: {args.output}")
    print(f"Compressed archive: {args.output.parent / f'{args.output.name}.tar.gz'}")
    print("\nTo download to your local machine:")
    print(f"  scp -i *.pem ubuntu@YOUR_IP:~/{args.output.name}.tar.gz .")
    print("\nTo view .ply files on your Mac with Brush:")
    print(f"  # Extract the archive first")
    print(f"  tar -xzf {args.output.name}.tar.gz")
    print(f"  # Then view with Brush")
    print(f"  ~/.brush/target/release/brush_app {args.output.name}/ply/point_cloud_XXXXX.ply")
    print("\nDon't forget to terminate your Lambda instance!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
