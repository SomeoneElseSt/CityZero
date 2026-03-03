#!/usr/bin/env python3
"""
YouTube Video Frame Extractor
Downloads a YouTube video and splits it into individual frames.

Use --lambda for Lambda Cloud / remote VM: auto-installs yt-dlp and writes to ~/youtube_train/.
"""

import sys
import os
import subprocess
import shutil
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
FFMPEG_FRAME_PATTERN = "frame_%06d.jpg"
BYTES_PER_MB = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download YouTube video and extract frames for COLMAP reconstruction",
        epilog="""
Frame rate recommendations for COLMAP:
  - 10-15 fps: Dashcam/vehicle footage (recommended for moving scenes)
  - 5 fps: Slow-moving scenes or static camera pans
  - 1-2 fps: Very slow scenes with minimal motion
        """
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second to extract (default: 15)")
    parser.add_argument("--compress", action="store_true", help="Compress extracted frames into tar.gz archive")
    parser.add_argument("--cookies", type=str, help="Path to cookies file for YouTube authentication")
    parser.add_argument("--lambda", dest="is_lambda", action="store_true", help="Lambda Cloud mode: auto-installs yt-dlp, writes to ~/youtube_train/")
    parser.add_argument("--skip-install", action="store_true", help="Skip yt-dlp auto-install (Lambda mode only)")
    return parser.parse_args()


def resolve_paths(is_lambda: bool) -> tuple[str, str, str]:
    if is_lambda:
        base = Path.home() / "youtube_train"
    else:
        base = (SCRIPT_DIR / "../outputs/youtube_train").resolve()
    images_dir = str(base / "images")
    temp_video = str(Path.home() / "temp_video.mp4") if is_lambda else str(SCRIPT_DIR / "temp_video.mp4")
    return str(base), images_dir, temp_video


def install_ytdlp() -> bool:
    if shutil.which("yt-dlp"):
        print("yt-dlp already installed")
        return True
    print("Installing yt-dlp...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "yt-dlp"], check=True, capture_output=True)
        print("yt-dlp installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install yt-dlp: {e}")
        return False


def check_dependencies() -> bool:
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg is not installed or not in PATH")
        return False
    try:
        import yt_dlp
        return True
    except ImportError:
        print("Error: yt-dlp is not installed. Run with --lambda to auto-install, or: uv pip install yt-dlp")
        return False


def create_output_directory(directory: str) -> bool:
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error: Failed to create output directory: {e}")
        return False


def download_video(url: str, output_path: str, cookies_path: str = None) -> bool:
    try:
        import yt_dlp

        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False,
        }

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

        print(f"Video downloaded: {os.path.getsize(output_path) / BYTES_PER_MB:.1f} MB")
        return True

    except Exception as e:
        print(f"Error: Failed to download video: {e}")
        return False


def extract_frames(video_path: str, output_dir: str, fps: int) -> bool:
    output_pattern = os.path.join(output_dir, FFMPEG_FRAME_PATTERN)
    ffmpeg_command = ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "2", output_pattern]

    print(f"Extracting frames at {fps} fps...")
    try:
        subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
        frame_count = len(list(Path(output_dir).glob("frame_*.jpg")))
        print(f"Extracted {frame_count} frames to: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: ffmpeg failed\n{e.stderr}")
        return False
    except Exception as e:
        print(f"Error: Failed to extract frames: {e}")
        return False


def cleanup_temp_file(file_path: str) -> None:
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Failed to clean up temp file: {e}")


def compress_images(base_dir: str, images_dir: str) -> bool:
    archive_path = os.path.join(base_dir, "images.tar.gz")

    if not os.path.exists(images_dir):
        print(f"Error: Images directory does not exist: {images_dir}")
        return False

    print(f"Compressing images to: {archive_path}")
    result = subprocess.run(["tar", "-czf", archive_path, "-C", base_dir, "-v", "images"], capture_output=False)

    if result.returncode != 0 or not os.path.exists(archive_path):
        print("Error: Compression failed")
        return False

    print(f"Archive: {os.path.getsize(archive_path) / BYTES_PER_MB:.1f} MB → {archive_path}")
    return True


def main() -> int:
    args = parse_args()
    base_dir, output_dir, temp_video = resolve_paths(args.is_lambda)

    if args.is_lambda and not args.skip_install:
        if not install_ytdlp():
            return 1

    if not check_dependencies():
        return 1

    if not create_output_directory(output_dir):
        return 1

    if not download_video(args.url, temp_video, cookies_path=args.cookies):
        cleanup_temp_file(temp_video)
        return 1

    if not extract_frames(temp_video, output_dir, fps=args.fps):
        cleanup_temp_file(temp_video)
        return 1

    cleanup_temp_file(temp_video)

    if args.compress:
        if not compress_images(base_dir, output_dir):
            print(f"Warning: Compression failed, but frames saved to: {output_dir}")
            return 0

    print(f"\nDone. Frames saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
