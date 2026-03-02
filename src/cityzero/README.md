# cityzero — Quick Reference

## Module Structure

| File | Role |
|------|------|
| `config.py` | `MapillaryConfig`, `BoundingBox` dataclasses, env loading, `CITY_BBOXES` |
| `mapillary.py` | `MapillaryClient` (API calls) + `ImageDownloader` (cache mgmt, grid split, download loop) |
| `cli.py` | Interactive CLI — argparse, folium map preview, questionary prompts |

**Standalone utility** (not part of the package):
`scripts/get_gps_coords.py` — post-hoc GPS enricher; use this if lat/lon was not saved at download time.

---

## Basic Usage

**Important**: Use `uv run` to activate the virtual environment and run via `-m cityzero.cli`.

### Interactive mode (recommended)
```bash
uv run python -m cityzero.cli
```
Arrow-key city selection → map preview in browser → download confirmation.

### Download a predefined city
```bash
uv run python -m cityzero.cli --city "San Francisco"
```

### Custom area (using coordinates)
```bash
uv run python -m cityzero.cli --bbox "west,south,east,north"
# Example:
uv run python -m cityzero.cli --bbox "-122.52,37.70,-122.35,37.83"
```

### Test with limited images
```bash
uv run python -m cityzero.cli --city "San Francisco" --limit 100
```

### Resume interrupted download
```bash
# Just run the same command again — it auto-resumes
uv run python -m cityzero.cli --city "San Francisco"
```

### Custom output directory
```bash
uv run python -m cityzero.cli --city "Miami" --output-dir data/miami_images
```

### Show map preview before downloading (non-interactive)
```bash
uv run python -m cityzero.cli --city "New York" --preview
```

### List available cities
```bash
uv run python -m cityzero.cli --list-cities
```

## Available Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--city` | City name (enables non-interactive mode) | `--city "New York"` |
| `--bbox` | Custom bounding box (west,south,east,north) | `--bbox "-74.05,40.68,-73.91,40.88"` |
| `--limit` | Max images to download (for testing) | `--limit 50` |
| `--output-dir` | Where to save images | `--output-dir data/my_images` |
| `--preview` | Show map preview before download (non-interactive) | `--preview` |
| `--list-cities` | Show available cities and exit | `--list-cities` |

## Predefined Cities

- San Francisco
- New York
- Los Angeles
- Chicago
- Miami

## Output

Images saved to: `data/raw/{city}/{image_id}.jpg`
Download progress: `data/raw/{city}/download_metadata.json`
Image metadata (GPS, timestamps): `data/raw/{city}/images_metadata.json`

## Resume Behavior

The downloader tracks what's been downloaded via `download_metadata.json`. If interrupted:
1. Run the same command again
2. It skips already-downloaded images
3. Continues from where it left off
