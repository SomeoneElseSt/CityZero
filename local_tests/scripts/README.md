# Scripts

## youtube_splits.py

Downloads a YouTube video and splits it into individual frames.

**Usage:**
```bash
uv run python youtube_splits.py "https://youtube.com/watch?v=VIDEO_ID"
```

**Output:** JPG frames saved to `../outputs/youtube_train/`

**Frame rate:** 1 fps (change `fps=1` on line 101 for more/fewer frames)

**Dependencies:** yt-dlp, ffmpeg
