# cityzero — Quick Reference

## Module Structure

| File | Role |
|------|------|
| `config.py` | `MapillaryConfig`, `BoundingBox` dataclasses, env loading, `CITY_BBOXES` |
| `client.py` | `MapillaryClient` (API calls) + `ImageDownloader` (cache mgmt, grid split, download loop) |
| `cli.py` | Interactive CLI — argparse, folium map preview, questionary prompts |

**Standalone utility** (not part of the package):
`scripts/get_gps_coords.py` — post-hoc GPS enricher; use this if lat/lon was not saved at download time.

---

## Usage

Run from `src/cityzero/`. For full usage and examples:

```bash
uv run python3 cli.py --help
```

## Output

Images saved to: `data/{city}/{image_id}.jpg`
Download progress: `data/{city}/download_metadata.json`
Image metadata (GPS, timestamps): `data/{city}/images_metadata.json`

## Resume Behavior

The downloader tracks what's been downloaded via `download_metadata.json`. If interrupted:
1. Run the same command again
2. It skips already-downloaded images
3. Continues from where it left off
