# Scripts

Helper scripts for the local_tests pipeline.

## youtube_splits.py

Downloads a YouTube video and extracts frames for COLMAP processing (local machine version).

### Usage

```bash
# Basic usage (15 fps, recommended for dashcam)
python youtube_splits.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Custom frame rate
python youtube_splits.py "https://www.youtube.com/watch?v=VIDEO_ID" --fps 10

# With compression (optional, JPEGs are already compressed)
python youtube_splits.py "https://www.youtube.com/watch?v=VIDEO_ID" --compress
```

### Requirements

- ffmpeg (must be installed and in PATH)
- yt-dlp (install with: `uv pip install yt-dlp`)

### Output

- Extracts frames to `../outputs/youtube_train/images/`
- Optional: Creates compressed archive at `../outputs/youtube_train/images.tar.gz`

### Notes

- Default: 15 fps (recommended for vehicle/dashcam footage)
- Frame naming: `frame_000001.jpg`, `frame_000002.jpg`, etc.
- Video is temporarily downloaded and deleted after extraction

---

## youtube_splits_lambda.py

Downloads a YouTube video and extracts frames directly on Lambda Cloud GPU instances (avoids hours of upload time).

### Usage

```bash
# On Lambda instance (requires cookies for YouTube authentication)
python3 youtube_splits_lambda.py "https://www.youtube.com/watch?v=VIDEO_ID" --fps 15 --cookies ~/cookies.txt

# Skip yt-dlp installation if already installed
python3 youtube_splits_lambda.py "URL" --fps 15 --cookies ~/cookies.txt --skip-install
```

### Requirements

- **cookies.txt**: Required to bypass YouTube bot detection
  - Export from browser using "Get cookies.txt LOCALLY" extension (Chrome/Firefox)
  - Upload to Lambda: `scp -i *.pem scripts/cookies.txt ubuntu@IP:~/`
  - **Never commit to git** (already in .gitignore)
- ffmpeg (pre-installed on Lambda Stack)
- yt-dlp (auto-installed by script)

### Output

- Extracts frames to `~/youtube_train/images/`
- Compatible with `lambda_build_colmap_cuda.py --images ~/youtube_train/images`

### Why Use This on Lambda?

- **Saves 8-15 hours** of upload time for 55k frames
- Lambda has faster download speeds than home upload
- Processes directly where COLMAP will run
- Takes ~20-30 minutes total (download + extraction)

### Cookies Setup

1. Install browser extension:
   - Chrome: "Get cookies.txt LOCALLY"
   - Firefox: "cookies.txt"

2. Go to YouTube while logged in

3. Click extension icon → Export cookies.txt

4. Save as `scripts/cookies.txt` (already gitignored)

5. Upload to Lambda when needed

### Notes

- Automatically installs yt-dlp on first run
- Use `--skip-install` on subsequent runs
- Frame rate recommendations:
  - 15 fps: Dashcam/vehicle footage (recommended)
  - 10 fps: Slower scenes
  - 5 fps: Very slow scenes (insufficient for fast-moving vehicles)
