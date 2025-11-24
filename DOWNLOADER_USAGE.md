# Image Downloader - Quick Reference

## Basic Usage

### Download a predefined city
```bash
python download_images.py --city "San Francisco"
```

### Custom area (using coordinates)
```bash
python download_images.py --bbox "west,south,east,north"
# Example:
python download_images.py --bbox "-122.52,37.70,-122.35,37.83"
```

### Test with limited images
```bash
python download_images.py --city "San Francisco" --limit 100
```

### Resume interrupted download
```bash
# Just run the same command again - it auto-resumes
python download_images.py --city "San Francisco"
```

### Custom output directory
```bash
python download_images.py --city "Miami" --output-dir data/miami_images
```

### List available cities
```bash
python download_images.py --list-cities
```

## Available Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--city` | City name (default: San Francisco) | `--city "New York"` |
| `--bbox` | Custom bounding box (west,south,east,north) | `--bbox "-74.05,40.68,-73.91,40.88"` |
| `--limit` | Max images to download (for testing) | `--limit 50` |
| `--output-dir` | Where to save images | `--output-dir data/my_images` |
| `--list-cities` | Show available cities and exit | `--list-cities` |

## Predefined Cities

- San Francisco
- New York
- Los Angeles
- Chicago
- Miami

## Output

Images saved to: `data/raw/{image_id}.jpg`  
Metadata saved to: `data/raw/download_metadata.json`

## Resume Behavior

The downloader automatically tracks what's been downloaded. If interrupted:
1. Just run the same command again
2. It will skip already-downloaded images
3. Continue from where it left off

No need to do anything special!
