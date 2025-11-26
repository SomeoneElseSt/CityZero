# Scripts

## youtube_splits.py

Downloads a YouTube video and splits it into individual frames.

**Usage:**
```bash
# Extract frames only
uv run python youtube_splits.py "https://youtube.com/watch?v=VIDEO_ID"

# Extract and compress into archive
uv run python youtube_splits.py "https://youtube.com/watch?v=VIDEO_ID" --compress
```

**Output:**
- JPG frames saved to `../outputs/youtube_train/images/`
- Compressed archive at `../outputs/youtube_train/images.tar.gz` (if --compress flag used)

**Compression note:** JPEGs are already optimally compressed. The `--compress` flag bundles files into a single archive for easier transfer, but won't reduce total size. Only use if you need a single file for upload/download.

**Frame rate:** 1 fps (change `fps=1` in extract_frames function for more/fewer frames)

**Dependencies:** yt-dlp, ffmpeg
