"""
This this the same as build_colmap.py except optimized for AMD builds. It has 'off' all CUDA options in installation commands, skips installing Nvidia-specific librariers, and adds some additional libraries for CPU-specific optimizations.

Will build to colmap_amd
 
"""

import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
import multiprocessing
from glob import glob


def run_command(cmd: list[str], error_msg: str, cwd: Path | None = None, env: dict | None = None, 
                capture_output: bool = True, text: bool = True, shell: bool = False,
                continue_on_error: bool = False) -> subprocess.CompletedProcess:
    """
    Run a command and handle errors consistently.
    
    Args:
        cmd: Command to run as a list
        error_msg: Error message to display on failure
        cwd: Working directory for the command
        env: Environment variables dict
        capture_output: Whether to capture stdout/stderr
        text: Whether to return text output
        shell: Whether to use shell execution
        continue_on_error: If True, print warning instead of exiting on error
    
    Returns:
        CompletedProcess result
    """
    kwargs = {}
    if cwd:
        kwargs["cwd"] = cwd
    if env:
        kwargs["env"] = env
    if capture_output:
        kwargs["capture_output"] = True
    if text:
        kwargs["text"] = True
    if shell:
        kwargs["shell"] = True
    
    result = subprocess.run(cmd, **kwargs)
    
    if result.returncode != 0:
        prefix = "WARNING" if continue_on_error else "ERROR"
        print(f"{prefix}: {error_msg} (exit code: {result.returncode})")
        if capture_output:
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
        if not continue_on_error:
            sys.exit(1)
    
    return result


def install_dependencies() -> None:
    """Install required dependencies for building COLMAP."""
    packages = [
        "git",
        "cmake",
        "ninja-build",
        "build-essential",
        "libboost-program-options-dev",
        "libboost-filesystem-dev",
        "libboost-graph-dev",
        "libboost-system-dev",
        "libeigen3-dev",
        "libfreeimage-dev",
        "libmetis-dev",
        "libgoogle-glog-dev",
        "libgflags-dev",
        "libabsl-dev",
        "libgtest-dev",
        "libgmock-dev",
        "libsuitesparse-dev",
        "libsqlite3-dev",
        "libglew-dev",
        "libcgal-dev",
        "libcurl4-openssl-dev",
        "libssl-dev",
        "libmkl-full-dev",
        "libtbb-dev",
        "libomp-dev",
        "libopenimageio-dev",
        "libopenexr-dev",
        "openimageio-tools",
        "libopencv-dev",
        "qt6-base-dev",
        "libqt6opengl6-dev",
        "libqt6openglwidgets6",
        # Non-COLMAP but useful libraries
        "micro",
        "sqlite3",
    ]

    print("Updating apt package list...")
    env = dict(os.environ, 
               DEBIAN_FRONTEND="noninteractive",
               NEEDRESTART_MODE="a",
               NEEDRESTART_SUSPEND="1")
    run_command(["sudo", "apt", "update"], "Failed to update apt package list", env=env)

    print("Resolving package conflicts...")
    subprocess.run(["sudo", "apt", "remove", "-y", "ucx", "libucx0"], env=env, capture_output=True)
    subprocess.run(["sudo", "apt", "upgrade", "-y"], env=env, capture_output=True)

    print("Installing dependencies...")
    cmd = ["sudo", "apt", "install", "-y"] + packages
    run_command(cmd, "Failed to install dependencies", env=env)

    print("All dependencies installed")


def build_ceres_cpu_optimized() -> None:
    """Build Ceres Solver from source with CPU optimizations (MKL, OpenMP)."""
    print("Building Ceres Solver from source with CPU optimizations (Intel MKL, OpenMP)")
    
    ceres_dir = Path.home() / "ceres-solver"
    
    if not ceres_dir.exists():
        print("Cloning Ceres Solver repository...")
        cmd = ["git", "clone", "https://github.com/ceres-solver/ceres-solver", str(ceres_dir)]
        run_command(cmd, "Failed to clone Ceres Solver")
        print("Ceres Solver cloned")
    else:
        print(f"Ceres Solver already cloned at {ceres_dir}")
        print("Pulling latest changes from master...")
        run_command(["git", "pull"], "Failed to pull latest changes", cwd=ceres_dir)
    
    print("Checking out master branch...")
    run_command(["git", "checkout", "master"], "Failed to checkout master branch", cwd=ceres_dir)
    
    print("Initializing git submodules (including bundled abseil)...")
    run_command(["git", "submodule", "update", "--init", "--recursive"], "Failed to initialize git submodules", cwd=ceres_dir)
    
    build_dir = ceres_dir / "build"
    build_dir.mkdir(exist_ok=True)
    
    print("Cleaning build directory...")
    if build_dir.exists():
        for item in build_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    
    print("\nConfiguring Ceres Solver with CMake...")
    
    cmake_cmd = [
        "cmake", "..",
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DUSE_CUDA=OFF",
        "-DBLA_VENDOR=Intel10_64lp",
        "-DOPENMP=ON",
        "-DBUILD_SHARED_LIBS=ON",
        "-DBUILD_TESTING=OFF",
        "-DBUILD_EXAMPLES=OFF",
    ]
    
    run_command(cmake_cmd, "CMake configuration failed", cwd=build_dir)
    
    print("CMake configuration complete")
    
    print("\nBuilding Ceres Solver...")
    num_cores = multiprocessing.cpu_count()
    build_cmd = ["ninja", "-j", str(num_cores)]
    start_time = datetime.now()
    
    result = subprocess.run(build_cmd, cwd=build_dir, capture_output=True, text=True)
    duration = (datetime.now() - start_time).total_seconds()
    
    if result.returncode != 0:
        print(f"ERROR: Build failed after {duration:.1f} seconds (exit code: {result.returncode})")
        if result.stderr:
            print(result.stderr)
        sys.exit(1)
    
    print(f"Ceres Solver built successfully in {duration/60:.1f} minutes")
    
    print("\nInstalling Ceres Solver...")
    install_cmd = ["sudo", "ninja", "install"]
    run_command(install_cmd, "Installation failed", cwd=build_dir)
    
    print("Registering new library...")
    run_command(["sudo", "ldconfig"], "ldconfig failed", continue_on_error=True)
    
    print("Ceres Solver installed successfully")


def build_colmap_from_source() -> None:
    """Build COLMAP from source with CPU optimizations."""
    print("Building COLMAP from source (CPU-only with Intel MKL and OpenMP)")

    colmap_dir = Path.home() / "colmap_amd"

    if not colmap_dir.exists():
        print("Cloning COLMAP repository...")
        cmd = ["git", "clone", "https://github.com/colmap/colmap.git", str(colmap_dir)]
        run_command(cmd, "Failed to clone COLMAP")
        print("COLMAP cloned")
    else:
        print(f"COLMAP already cloned at {colmap_dir}")

    build_dir = colmap_dir / "build"
    build_dir.mkdir(exist_ok=True)

    print("\nConfiguring COLMAP with CMake...")

    cmake_cmd = [
        "cmake", "..",
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBLA_VENDOR=Intel10_64lp",
        "-DCUDA_ENABLED=OFF",
        "-DOPENMP_ENABLED=ON",
        "-DGUI_ENABLED=OFF",
    ]

    run_command(cmake_cmd, "CMake configuration failed", cwd=build_dir)

    print("CMake configuration complete")

    print("\nBuilding COLMAP...")
    num_cores = multiprocessing.cpu_count()
    build_cmd = ["ninja", "-j", str(num_cores)]
    start_time = datetime.now()

    result = subprocess.run(build_cmd, cwd=build_dir, capture_output=True, text=True)
    duration = (datetime.now() - start_time).total_seconds()

    if result.returncode != 0:
        print(f"ERROR: Build failed after {duration:.1f} seconds (exit code: {result.returncode})")
        if result.stderr:
            print(result.stderr)
        sys.exit(1)

    print(f"COLMAP built successfully in {duration/60:.1f} minutes")

    print("\nInstalling COLMAP...")
    install_cmd = ["sudo", "ninja", "install"]
    run_command(install_cmd, "Installation failed", cwd=build_dir)

    print("COLMAP installed to /usr/local/bin/colmap")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build COLMAP from source with CPU optimizations for AMD GPU droplets")
    parser.add_argument(
        "--skip-dependencies",
        action="store_true",
        help="Skip dependency installation (use if already installed)"
    )
    parser.add_argument(
        "--skip-ceres",
        action="store_true",
        help="Skip building and installing Ceres Solver"
    )

    args = parser.parse_args()

    print("Starting COLMAP build (CPU-optimized for AMD GPU droplet)")
    print("Note: COLMAP does not support AMD GPU acceleration. Building with Intel MKL and OpenMP instead.\n")

    if not args.skip_dependencies:
        install_dependencies()
    else:
        print("Skipping dependency installation (--skip-dependencies)")

    if not args.skip_ceres:
        build_ceres_cpu_optimized()
    else:
        print("Skipping Ceres Solver build (--skip-ceres)")
    
    build_colmap_from_source()

    print("\nBuild complete!")
    print("COLMAP is now available at /usr/local/bin/colmap")
