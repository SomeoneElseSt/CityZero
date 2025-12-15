#!/usr/bin/env python3
"""
YouTube Video Frame Extractor for Lambda Cloud
Downloads a YouTube video and splits it into individual frames.
Automatically installs dependencies and outputs to Lambda-friendly paths.
"""

import sys
import os
import subprocess
import shutil
import argparse
from pathlib import Path

# Constants - Lambda root-level paths
YOUTUBE_TRAIN_DIR = str(Path.home() / "youtube_train")
OUTPUT_DIR = str(Path.home() / "youtube_train" / "images")
TEMP_VIDEO_PATH = str(Path.home() / "temp_video.mp4")
FFMPEG_FRAME_PATTERN = "frame_%06d.jpg"
BYTES_PER_MB = 1024 * 1024


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download YouTube video and extract frames for COLMAP reconstruction (Lambda Cloud version)",
        epilog="""
Frame rate recommendations for COLMAP:
  - 10-15 fps: Dashcam/vehicle footage (recommended for moving scenes)
  - 5 fps: Slow-moving scenes or static camera pans
  - 1-2 fps: Very slow scenes with minimal motion

Higher fps = better overlap but more frames to process.

Output paths:
  - Images: ~/youtube_train/images/
  - Compatible with: python3 lambda_build_colmap_cuda.py --images ~/youtube_train/images --output ~/colmap_output
        """
    )
    parser.add_argument(
        "url",
        help="YouTube video URL"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second to extract (default: 15, recommended for dashcam footage)"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip yt-dlp installation (use if already installed)"
    )
    parser.add_argument(
        "--cookies",
        type=str,
        help="Path to cookies file for YouTube authentication (use if getting bot detection errors)"
    )
    return parser.parse_args()


def install_ytdlp() -> bool:
    """Install yt-dlp if not already installed."""
    if shutil.which("yt-dlp"):
        print("yt-dlp is already installed")
        return True
    
    print("Installing yt-dlp...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", "yt-dlp"],
            check=True,
            capture_output=True
        )
        print("yt-dlp installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to install yt-dlp: {e}")
        return False


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg is not installed or not in PATH")
        print("On Lambda Stack, ffmpeg should be pre-installed")
        return False

    try:
        import yt_dlp
        return True
    except ImportError:
        print("yt-dlp not found - will install automatically")
        return True  # Will install in main()


def create_output_directory(directory: str) -> bool:
    """Create output directory if it doesn't exist."""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {directory}")
        return True
    except Exception as e:
        print(f"Error: Failed to create output directory: {e}")
        return False


def download_video(url: str, output_path: str, cookies_path: str = None) -> bool:
    """Download YouTube video to temporary file."""
    try:
        import yt_dlp
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False,
        }
        
        # Add cookies if provided
        if cookies_path:
            if not os.path.exists(cookies_path):
                print(f"Error: Cookies file not found: {cookies_path}")
                return False
            ydl_opts['cookiefile'] = cookies_path
            print(f"Using cookies from: {cookies_path}")

        print(f"Downloading video from: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not os.path.exists(output_path):
            print("Error: Video download failed - file not created")
            return False

        video_size_mb = os.path.getsize(output_path) / BYTES_PER_MB
        print(f"Video downloaded successfully: {video_size_mb:.1f} MB")
        return True

    except Exception as e:
        print(f"Error: Failed to download video: {e}")
        return False


def extract_frames(video_path: str, output_dir: str, fps: int = 15) -> bool:
    """Extract frames from video using ffmpeg.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 15)
             - 10-15 fps recommended for dashcam/vehicle footage
             - 5 fps is too sparse for moving vehicles (insufficient overlap)
             - Higher fps = better overlap but more frames to process
    """
    output_pattern = os.path.join(output_dir, FFMPEG_FRAME_PATTERN)

    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",  # Quality level (2 is high quality)
        output_pattern
    ]

    print(f"\nExtracting frames from video at {fps} fps...")
    print(f"Note: {fps} fps provides good overlap for COLMAP reconstruction from moving vehicles")
    try:
        result = subprocess.run(
            ffmpeg_command,
            capture_output=True,
            text=True,
            check=True
        )

        # Count extracted frames
        frame_files = list(Path(output_dir).glob("frame_*.jpg"))
        print(f"Successfully extracted {len(frame_files)} frames")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error: ffmpeg failed to extract frames")
        print(f"ffmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error: Failed to extract frames: {e}")
        return False


def cleanup_temp_file(file_path: str) -> None:
    """Remove temporary video file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary video file")
    except Exception as e:
        print(f"Warning: Failed to clean up temporary file: {e}")


def main() -> int:
    """Main execution function."""
    args = parse_args()

    print("="*70)
    print("YOUTUBE FRAME EXTRACTOR FOR LAMBDA CLOUD")
    print("="*70)

    if not args.skip_install:
        if not install_ytdlp():
            print("\nNote: If installation failed, try:")
            print("  pip install --user yt-dlp")
            print("Then re-run with --skip-install flag")
            return 1

    if not check_dependencies():
        return 1

    if not create_output_directory(OUTPUT_DIR):
        return 1

    if not download_video(args.url, TEMP_VIDEO_PATH, cookies_path=args.cookies):
        cleanup_temp_file(TEMP_VIDEO_PATH)
        return 1

    if not extract_frames(TEMP_VIDEO_PATH, OUTPUT_DIR, fps=args.fps):
        cleanup_temp_file(TEMP_VIDEO_PATH)
        return 1

    cleanup_temp_file(TEMP_VIDEO_PATH)

    print("\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\nFrames saved to: {OUTPUT_DIR}")
    print(f"Total frames: {len(list(Path(OUTPUT_DIR).glob('frame_*.jpg')))}")
    print("\nNext step - run COLMAP:")
    print(f"  python3 lambda_build_colmap_cuda.py --images {OUTPUT_DIR} --output ~/colmap_output --skip-build")
    print("\nNote: Use --skip-build if COLMAP is already installed")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
