#!/usr/bin/env python3
"""
YouTube Video Frame Extractor
Downloads a YouTube video and splits it into individual frames.
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path

# Constants
OUTPUT_DIR = "../outputs/youtube_train"
TEMP_VIDEO_PATH = "temp_video.mp4"
FFMPEG_FRAME_PATTERN = "frame_%06d.jpg"


def validate_args() -> str | None:
    """Validate command-line arguments and return video URL."""
    if len(sys.argv) != 2:
        print("Error: Exactly one argument required (YouTube video URL)")
        print("Usage: python3 youtube_splits.py <youtube_url>")
        return None

    return sys.argv[1]


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg is not installed or not in PATH")
        return False

    try:
        import yt_dlp
        return True
    except ImportError:
        print("Error: yt-dlp is not installed")
        print("Install it with: uv pip install yt-dlp")
        return False


def create_output_directory(directory: str) -> bool:
    """Create output directory if it doesn't exist."""
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error: Failed to create output directory: {e}")
        return False


def download_video(url: str, output_path: str) -> bool:
    """Download YouTube video to temporary file."""
    try:

        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_path,
            'quiet': False,
            'no_warnings': False,
        }

        print(f"Downloading video from: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not os.path.exists(output_path):
            print("Error: Video download failed - file not created")
            return False

        print(f"Video downloaded successfully to: {output_path}")
        return True

    except Exception as e:
        print(f"Error: Failed to download video: {e}")
        return False


def extract_frames(video_path: str, output_dir: str) -> bool:
    """Extract frames from video using ffmpeg."""
    output_pattern = os.path.join(output_dir, FFMPEG_FRAME_PATTERN)

    ffmpeg_command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "fps=5",  # fps extracts N frame per second (adjust as needed)
        "-q:v", "2",      # Quality level (2 is high quality)
        output_pattern
    ]

    print("Extracting frames from video...")
    try:
        result = subprocess.run(
            ffmpeg_command,
            capture_output=True,
            text=True,
            check=True
        )

        # Count extracted frames
        frame_files = list(Path(output_dir).glob("frame_*.jpg"))
        print(f"Successfully extracted {len(frame_files)} frames to: {output_dir}")
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
            print(f"Cleaned up temporary video file: {file_path}")
    except Exception as e:
        print(f"Warning: Failed to clean up temporary file: {e}")


def main() -> int:
    """Main execution function."""
    video_url = validate_args()
    if not video_url:
        return 1

    if not check_dependencies():
        return 1

    if not create_output_directory(OUTPUT_DIR):
        return 1

    if not download_video(video_url, TEMP_VIDEO_PATH):
        cleanup_temp_file(TEMP_VIDEO_PATH)
        return 1

    if not extract_frames(TEMP_VIDEO_PATH, OUTPUT_DIR):
        cleanup_temp_file(TEMP_VIDEO_PATH)
        return 1

    cleanup_temp_file(TEMP_VIDEO_PATH)
    print("\nProcess completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
