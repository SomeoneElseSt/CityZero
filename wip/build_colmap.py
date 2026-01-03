"""
This is a new script to simplfiy what I've learnt from debugging COLMAP. I'm trying to take away as much complexity as possible, get down to basics. 

Should:

1. Build Colmap from source 
2. Compute Features
3. Match them using a Vocab Tree
"""

import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
import multiprocessing


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
        "libgtest-dev",
        "libgmock-dev",
        "libsqlite3-dev",
        "libglew-dev",
        "libcgal-dev",
        "libceres-dev",
        "libcurl4-openssl-dev",
        "libssl-dev",
        "libmkl-full-dev",
        "libtbb-dev",
        "libopenimageio-dev",
        "libopenexr-dev",
        "openimageio-tools",
        "libopencv-dev",
        "qt6-base-dev",
        "libqt6opengl6-dev",
        "libqt6openglwidgets6",
        "nvidia-cuda-toolkit",
        "nvidia-cuda-toolkit-gcc",
    ]

    print("Updating apt package list...")
    env = dict(os.environ, 
               DEBIAN_FRONTEND="noninteractive",
               NEEDRESTART_MODE="a",
               NEEDRESTART_SUSPEND="1")
    update_result = subprocess.run(["sudo", "apt", "update"], env=env)
    if update_result.returncode != 0:
        print(f"ERROR: Failed to update apt package list (exit code: {update_result.returncode})")
        sys.exit(1)

    print("Resolving package conflicts...")
    subprocess.run(["sudo", "apt", "remove", "-y", "ucx", "libucx0"], env=env)
    subprocess.run(["sudo", "apt", "upgrade", "-y"], env=env)

    print("Installing dependencies...")
    cmd = ["sudo", "apt", "install", "-y"] + packages
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"ERROR: Failed to install dependencies (exit code: {result.returncode})")
        sys.exit(1)

    print("All dependencies installed")


def resolve_nvcc_path() -> str | None:
    """Find nvcc compiler path."""
    local_nvcc = Path("/usr/local/cuda/bin/nvcc")
    if local_nvcc.exists():
        return str(local_nvcc)


    apt_nvcc = Path("/usr/bin/nvcc")
    if apt_nvcc.exists():
        return str(apt_nvcc)

    found = shutil.which("nvcc")
    if found:
        return found

    return None


def build_colmap_from_source() -> None:
    """Build COLMAP from source with CUDA support."""
    print("Building COLMAP from source with CUDA support")

    colmap_dir = Path.home() / "colmap_cuda"

    if not colmap_dir.exists():
        print("Cloning COLMAP repository...")
        cmd = ["git", "clone", "https://github.com/colmap/colmap.git", str(colmap_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Failed to clone COLMAP (exit code: {result.returncode})")
            if result.stderr:
                print(result.stderr)
            sys.exit(1)
        print("COLMAP cloned")
    else:
        print(f"COLMAP already cloned at {colmap_dir}")

    build_dir = colmap_dir / "build"
    build_dir.mkdir(exist_ok=True)

    print("\nConfiguring COLMAP with CMake...")
    nvcc_path = resolve_nvcc_path()
    if not nvcc_path:
        print("ERROR: nvcc not found")
        sys.exit(1)

    cmake_cmd = [
        "cmake", "..",
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBLA_VENDOR=Intel10_64lp",
        "-DCMAKE_CUDA_ARCHITECTURES=80;86",
        "-DCUDA_ENABLED=ON",
        "-DCMAKE_CUDA_COMPILER=" + nvcc_path,
        "-DGUI_ENABLED=OFF",
    ]

    result = subprocess.run(cmake_cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: CMake configuration failed (exit code: {result.returncode})")
        if result.stderr:
            print(result.stderr)
        sys.exit(1)

    print("CMake configuration complete")

    print("\nBuilding COLMAP...")
    num_cores = multiprocessing.cpu_count()
    build_cmd = ["ninja", "-j", str(num_cores)]
    start_time = datetime.now()

    result = subprocess.run(build_cmd, cwd=build_dir)
    duration = (datetime.now() - start_time).total_seconds()

    if result.returncode != 0:
        print(f"ERROR: Build failed after {duration:.1f} seconds (exit code: {result.returncode})")
        if result.stderr:
            print(result.stderr)
        sys.exit(1)

    print(f"COLMAP built successfully in {duration/60:.1f} minutes")

    print("\nInstalling COLMAP...")
    install_cmd = ["sudo", "ninja", "install"]
    result = subprocess.run(install_cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Installation failed (exit code: {result.returncode})")
        if result.stderr:
            print(result.stderr)
        sys.exit(1)

    print("COLMAP installed to /usr/local/bin/colmap")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build COLMAP from source with CUDA support")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building COLMAP (use if already built)"
    )

    args = parser.parse_args()

    if args.skip_build:
        print("Skipping build (--skip-build)")
        sys.exit(0)

    print("Starting build")

    install_dependencies()
    build_colmap_from_source()

    print("\nBuild complete!")
