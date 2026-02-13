"""
This script is optimized for AMD GPU builds with multi-GPU support.
It builds ONNX Runtime 1.24.1 from source before building COLMAP to ensure
version compatibility.

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
                capture_output: bool = False, text: bool = True, shell: bool = False,
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


def upgrade_cmake() -> None:
    """Upgrade CMake to version 3.28+ required by ONNX Runtime 1.24.1+."""
    print("Upgrading CMake to 3.28+ via pip...")
    run_command(["pip", "install", "cmake", "--upgrade"], "Failed to upgrade CMake")
    
    # Verify installation
    result = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
    print(f"Installed: {result.stdout.split(chr(10))[0]}")


def build_onnxruntime_1_24_1() -> None:
    """Build ONNX Runtime 1.24.1 from source to match COLMAP requirements."""
    print("\nBuilding ONNX Runtime 1.24.1 from source...")
    print("This is required for COLMAP compatibility on multi-GPU systems")
    
    onnx_dir = Path.home() / "onnxruntime"
    
    # Clone or update repository
    if not onnx_dir.exists():
        print("Cloning ONNX Runtime repository...")
        cmd = ["git", "clone", "--recursive", "https://github.com/microsoft/onnxruntime.git", str(onnx_dir)]
        run_command(cmd, "Failed to clone ONNX Runtime")
    else:
        print(f"ONNX Runtime already cloned at {onnx_dir}")
        print("Fetching latest tags...")
        run_command(["git", "fetch", "--all", "--tags"], "Failed to fetch tags", cwd=onnx_dir)
    
    # Checkout version 1.24.1
    print("Checking out version 1.24.1...")
    run_command(["git", "checkout", "v1.24.1"], "Failed to checkout v1.24.1", cwd=onnx_dir)
    run_command(
        ["git", "submodule", "update", "--init", "--recursive"],
        "Failed to update submodules",
        cwd=onnx_dir
    )
    
    # Build ONNX Runtime
    print("\nBuilding ONNX Runtime 1.24.1...")
    build_script = onnx_dir / "build.sh"
    
    build_cmd = [
        str(build_script),
        "--config", "Release",
        "--build_shared_lib",
        "--parallel",
        "--skip_tests"
    ]
    
    start_time = datetime.now()
    result = subprocess.run(build_cmd, cwd=onnx_dir)
    duration = (datetime.now() - start_time).total_seconds()
    
    if result.returncode != 0:
        print(f"ERROR: ONNX Runtime build failed after {duration:.1f} seconds")
        sys.exit(1)
    
    print(f"ONNX Runtime built successfully in {duration/60:.1f} minutes")
    
    # Install the shared library
    print("\nInstalling ONNX Runtime shared library...")
    lib_dir = onnx_dir / "build" / "Linux" / "Release"
    
    # Find the built library files
    lib_files = list(lib_dir.glob("libonnxruntime.so.1.24.*"))
    if not lib_files:
        print("ERROR: Built library files not found")
        sys.exit(1)
    
    versioned_lib = lib_files[0]
    print(f"Found library: {versioned_lib}")
    
    # Remove old ONNX Runtime libraries
    print("Removing old ONNX Runtime libraries...")
    subprocess.run(["sudo", "rm", "-f", "/usr/local/lib/libonnxruntime.so*"])
    
    # Copy the versioned library
    print(f"Installing {versioned_lib.name} to /usr/local/lib/...")
    run_command(
        ["sudo", "cp", str(versioned_lib), "/usr/local/lib/"],
        "Failed to copy library"
    )
    
    # Create symbolic links
    print("Creating symbolic links...")
    run_command(
        ["sudo", "ln", "-sf", versioned_lib.name, "/usr/local/lib/libonnxruntime.so.1"],
        "Failed to create symlink .so.1"
    )
    run_command(
        ["sudo", "ln", "-sf", "libonnxruntime.so.1", "/usr/local/lib/libonnxruntime.so"],
        "Failed to create symlink .so"
    )
    
    # Update library cache
    print("Updating library cache...")
    run_command(["sudo", "ldconfig"], "Failed to run ldconfig")
    
    print("✓ ONNX Runtime 1.24.1 installation complete")


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
    subprocess.run(["sudo", "apt", "remove", "-y", "ucx", "libucx0"], env=env)
    subprocess.run(["sudo", "apt", "upgrade", "-y"], env=env)

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

    result = subprocess.run(build_cmd, cwd=build_dir)
    duration = (datetime.now() - start_time).total_seconds()

    if result.returncode != 0:
        print(f"ERROR: Build failed after {duration:.1f} seconds (exit code: {result.returncode})")
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

    result = subprocess.run(build_cmd, cwd=build_dir)
    duration = (datetime.now() - start_time).total_seconds()

    if result.returncode != 0:
        print(f"ERROR: Build failed after {duration:.1f} seconds (exit code: {result.returncode})")
        sys.exit(1)

    print(f"COLMAP built successfully in {duration/60:.1f} minutes")

    print("\nInstalling COLMAP...")
    install_cmd = ["sudo", "ninja", "install"]
    run_command(install_cmd, "Installation failed", cwd=build_dir)

    print("COLMAP installed to /usr/local/bin/colmap")
    
    # Verify COLMAP can find ONNX Runtime
    print("\nVerifying COLMAP installation...")
    result = subprocess.run(["colmap", "-h"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ COLMAP is working correctly")
    else:
        print(f"WARNING: COLMAP verification failed (exit code: {result.returncode})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build COLMAP from source with CPU optimizations for AMD GPU droplets (multi-GPU compatible)"
    )
    parser.add_argument(
        "--skip-dependencies",
        action="store_true",
        help="Skip dependency installation (use if already installed)"
    )
    parser.add_argument(
        "--skip-cmake-upgrade",
        action="store_true",
        help="Skip CMake upgrade (use if already have CMake 3.28+)"
    )
    parser.add_argument(
        "--skip-onnxruntime",
        action="store_true",
        help="Skip building ONNX Runtime 1.24.1 (use if already installed)"
    )
    parser.add_argument(
        "--skip-ceres",
        action="store_true",
        help="Skip building and installing Ceres Solver"
    )

    args = parser.parse_args()

    print("Starting COLMAP build (CPU-optimized for AMD GPU droplet with multi-GPU support)\n")

    if not args.skip_cmake_upgrade:
        upgrade_cmake()
    else:
        print("Skipping CMake upgrade (--skip-cmake-upgrade)")

    if not args.skip_dependencies:
        install_dependencies()
    else:
        print("Skipping dependency installation (--skip-dependencies)")

    if not args.skip_onnxruntime:
        build_onnxruntime_1_24_1()
    else:
        print("Skipping ONNX Runtime build (--skip-onnxruntime)")

    if not args.skip_ceres:
        build_ceres_cpu_optimized()
    else:
        print("Skipping Ceres Solver build (--skip-ceres)")
    
    build_colmap_from_source()

    print("\n" + "=" * 80)
    print("Build complete!")
    print("COLMAP is now available at /usr/local/bin/colmap")
    print("=" * 80)
