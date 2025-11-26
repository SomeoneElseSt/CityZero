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
import yt_dlp

# Constants - paths relative to script location
SCRIPT_DIR = Path(__file__).parent.resolve()
YOUTUBE_TRAIN_DIR = str((SCRIPT_DIR / "../outputs/youtube_train").resolve())
OUTPUT_DIR = str((SCRIPT_DIR / "../outputs/youtube_train/images").resolve())
TEMP_VIDEO_PATH = str((SCRIPT_DIR / "temp_video.mp4").resolve())
FFMPEG_FRAME_PATTERN = "frame_%06d.jpg"
BYTES_PER_MB = 1024 * 1024


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


def compress_images(youtube_train_dir: str, images_dir: str) -> bool:
    """Compress images folder into tar.gz archive."""
    archive_path = os.path.join(youtube_train_dir, "images.tar.gz")

    if not os.path.exists(images_dir):
        print(f"Error: Images directory does not exist: {images_dir}")
        return False

    print(f"\nCompressing images to: {archive_path}")
    print("This may take a few minutes depending on the number of frames...")

    tar_cmd = [
        "tar",
        "-czf",
        archive_path,
        "-C", youtube_train_dir,
        "-v",
        "images"
    ]

    result = subprocess.run(tar_cmd, capture_output=False)

    if result.returncode != 0:
        print(f"Error: tar command failed with exit code {result.returncode}")
        return False

    if not os.path.exists(archive_path):
        print("Error: Archive creation failed - file not created")
        return False

    archive_size_mb = os.path.getsize(archive_path) / BYTES_PER_MB
    print(f"\nCompression complete!")
    print(f"Archive size: {archive_size_mb:.1f} MB")
    print(f"Location: {archive_path}")
    return True


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

    print("\nStarting video frame extraction now - your laptop's fans may spin up quickly.")
    print("This is normal. You can monitor CPU usage with sudo asitop or htop depending on which you have installed.\n")

    if not extract_frames(TEMP_VIDEO_PATH, OUTPUT_DIR):
        cleanup_temp_file(TEMP_VIDEO_PATH)
        return 1

    cleanup_temp_file(TEMP_VIDEO_PATH)

    if not compress_images(YOUTUBE_TRAIN_DIR, OUTPUT_DIR):
        print("\nWarning: Compression failed, but frames were extracted successfully")
        print(f"Frames location: {OUTPUT_DIR}")
        return 0

    print("\nProcess completed successfully!")
    print(f"Frames saved to: {OUTPUT_DIR}")
    print(f"Compressed archive: {YOUTUBE_TRAIN_DIR}/images.tar.gz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
