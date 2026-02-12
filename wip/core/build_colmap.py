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
        "libopenimageio-dev",
        "libopenexr-dev",
        "openimageio-tools",
        "libopencv-dev",
        "qt6-base-dev",
        "libqt6opengl6-dev",
        "libqt6openglwidgets6",
        "nvidia-cuda-toolkit",
        "nvidia-cuda-toolkit-gcc",
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


def build_ceres_with_cuda() -> None:
    """Build Ceres Solver from source with CUDA support."""
    print("Installing CUDS...")
    
    cudss_deb = "cudss-local-repo-ubuntu2204-0.7.1_0.7.1-1_amd64.deb"
    cudss_url = "https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-repo-ubuntu2204-0.7.1_0.7.1-1_amd64.deb"
    
    print(f"Downloading CUDS installer...")
    run_command(["sudo", "wget", cudss_url], "Failed to download CUDS installer")
    
    print("Installing CUDS repository...")
    run_command(["sudo", "dpkg", "-i", cudss_deb], "Failed to install CUDS repository")
    
    print("Copying CUDS keyring...")
    keyring_pattern = "/var/cudss-local-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg"
    keyring_files = glob(keyring_pattern)
    if not keyring_files:
        print(f"ERROR: No keyring files found matching {keyring_pattern}")
        sys.exit(1)
    keyring_src = max(keyring_files, key=os.path.getmtime)
    keyring_dest = "/usr/share/keyrings/"
    run_command(["sudo", "cp", keyring_src, keyring_dest], "Failed to copy CUDS keyring")
    
    print("Updating apt package list...")
    env = dict(os.environ, 
               DEBIAN_FRONTEND="noninteractive",
               NEEDRESTART_MODE="a",
               NEEDRESTART_SUSPEND="1")
    run_command(["sudo", "apt-get", "update"], "Failed to update apt package list", env=env)
    
    print("Installing CUDS...")
    run_command(["sudo", "apt-get", "-y", "install", "cudss"], "Failed to install CUDS", env=env)
    
    print("CUDS installed successfully")
    
    print("Removing CUDA 13 cuDSS packages to avoid libcublas.so.13 conflicts...")
    cuda13_packages = [
        "cudss-cuda-13",
        "libcudss0-cuda-13", 
        "libcudss0-dev-cuda-13",
        "libcudss0-static-cuda-13"
    ]
    run_command(
        ["sudo", "apt", "remove", "-y"] + cuda13_packages,
        "Failed to remove CUDA 13 packages",
        env=env
    )
    
    print("Building Ceres Solver from source with CUDA and cuDSS support")
    
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
    
    print("Checking out master branch (cuDSS requires master, not stable 2.2.0)...")
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
    nvcc_path = resolve_nvcc_path()
    if not nvcc_path:
        print("ERROR: nvcc not found")
        sys.exit(1)
    
    cmake_cmd = [
        "cmake", "..",
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DUSE_CUDA=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=80",
        "-DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss",
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
        run_command(cmd, "Failed to clone COLMAP")
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
        "-DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build COLMAP from source with CUDA support")
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

    print("Starting COLMAP build")

    if not args.skip_dependencies:
        install_dependencies()
    else:
        print("Skipping dependency installation (--skip-dependencies)")

    if not args.skip_ceres:
        build_ceres_with_cuda()
    else:
        print("Skipping Ceres Solver build (--skip-ceres)")
    build_colmap_from_source()

    print("\nBuild complete!")
