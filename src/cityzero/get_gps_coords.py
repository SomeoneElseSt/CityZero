"""
I recently learned that Mapillary's API has an endpoint for pulling image metadata specifically -- most is useless, but it lets me get the GPS coordinates of where each image was taken. 
I've realized throwing all the images to COLMAP at once is a bit naive and computationally very expensive, so I now plan to use geo-fencing and build smaller graphs that can be connected with each other. 
I also think since most of these images do share some sequential nature (i.e., they're usually only of specific segments in the city like roads) there is a locally optimized way to run the matchers. 
The idea is to reconstruct patches of the city and join them together in the image graph instead of all at once. 
This script is a loop on a file I kept of the image ID's of the dataset that gets that info and writes it as a dict with lan and lon coords. 
"""

import json
import logging
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

LOG_EVERY = 250
SAVE_EVERY = 1
TIMEOUT_SECONDS = 30
MAX_RETRIES = 3


def atomic_write_json(path: str, data: dict, logger: logging.Logger) -> bool:
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, path)
        return True
    except Exception as exc:
        logger.error(f"Failed writing {path}: {exc}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


def fetch_lat_lon(session: requests.Session, image_id: str, token: str, logger: logging.Logger) -> tuple[str, str]:
    url = f"https://graph.mapillary.com/{image_id}"
    params = {"access_token": token, "fields": "geometry"}

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=TIMEOUT_SECONDS)
        except Exception as exc:
            last_error = str(exc)
            time.sleep(min(10, 2 ** (attempt - 1)))
            continue

        if resp.status_code != 200:
            last_error = f"status={resp.status_code} body={resp.text[:200]!r}"
            time.sleep(min(10, 2 ** (attempt - 1)))
            continue

        try:
            payload = resp.json()
        except Exception as exc:
            last_error = str(exc)
            time.sleep(min(10, 2 ** (attempt - 1)))
            continue

        geometry = payload.get("geometry", {}) if isinstance(payload, dict) else {}
        coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
        if not isinstance(coords, list) or len(coords) < 2:
            return "", ""

        lon = coords[0]
        lat = coords[1]
        if lon is None or lat is None:
            return "", ""

        return str(lat), str(lon)

    logger.error(f"Giving up on {image_id} after {MAX_RETRIES} retries: {last_error}")
    return "", ""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("cityzero.get_gps_coords")

    token = os.getenv("MAPILLARY_CLIENT_TOKEN", "")
    if not token:
        logger.error("MAPILLARY_CLIENT_TOKEN not found in environment (.env)")
        return

    metadata_path = os.path.join(os.path.dirname(__file__), "download_metadata.json")
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        logger.error(f"Failed reading {metadata_path}: {exc}")
        return

    if not isinstance(metadata, dict):
        logger.error(f"Expected JSON object at top-level in {metadata_path}")
        return

    downloaded_ids = metadata.get("downloaded_ids")
    if downloaded_ids is None:
        logger.error('Missing required key "downloaded_ids"')
        return

    did_migrate = False
    if isinstance(downloaded_ids, list):
        downloaded_ids = {str(image_id): None for image_id in downloaded_ids if isinstance(image_id, str)}
        metadata["downloaded_ids"] = downloaded_ids
        did_migrate = True

    if not isinstance(downloaded_ids, dict):
        logger.error('"downloaded_ids" must be a list or dict')
        return

    if did_migrate:
        if not atomic_write_json(metadata_path, metadata, logger):
            return

    total = len(downloaded_ids)
    processed = 0
    first_unprocessed = None
    for image_id, entry in downloaded_ids.items():
        if entry is None or not isinstance(entry, dict):
            if first_unprocessed is None:
                first_unprocessed = image_id
            continue
        lat_ok = bool(entry.get("lat", ""))
        lon_ok = bool(entry.get("lon", ""))
        if lat_ok and lon_ok:
            processed += 1
            continue
        if first_unprocessed is None:
            first_unprocessed = image_id

    logger.info(f"Loaded {total} image IDs from {metadata_path}. Found {processed} already processed; will skip.")
    if first_unprocessed is None:
        logger.info("All image IDs already have lat/lon. Nothing to do.")
        return

    logger.info(f"Resuming from first unprocessed image_id={first_unprocessed}")

    session = requests.Session()
    started = False
    processed_this_run = 0

    for image_id, entry in downloaded_ids.items():
        if not started and image_id != first_unprocessed:
            continue
        started = True

        if entry is not None and isinstance(entry, dict):
            lat_ok = bool(entry.get("lat", ""))
            lon_ok = bool(entry.get("lon", ""))
            if lat_ok and lon_ok:
                continue

        lat, lon = fetch_lat_lon(session, image_id, token, logger)
        downloaded_ids[image_id] = {"lat": lat, "lon": lon}

        processed_this_run += 1
        processed += 1
        left = total - processed

        if processed_this_run % LOG_EVERY == 0 or left == 0:
            logger.info(f"Progress: {processed}/{total} processed | left: {left}")

        if processed_this_run % SAVE_EVERY != 0 and left != 0:
            continue

        if not atomic_write_json(metadata_path, metadata, logger):
            return

    logger.info("Done.")


if __name__ == "__main__":
    main()


