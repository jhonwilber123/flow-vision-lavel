"""Download the public Peru traffic dataset into this project.

The Google Drive folder contains:
  data/raw/metadata.csv
  data/raw/images/train/*.jpg

Usage:
  python download_dataset.py
  python download_dataset.py --limit 200
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gdown
import requests


FOLDER_ID = "1zJud5tvylJ7AxAdcO8iWtyyjJfSCzc7v"
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"


def list_drive_files():
    return gdown.download_folder(
        id=FOLDER_ID,
        output=str(RAW_DIR),
        quiet=True,
        skip_download=True,
    )


def direct_download(file_id, destination: Path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    with requests.get(url, stream=True, allow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        ctype = response.headers.get("Content-Type", "")
        if "text/html" in ctype.lower():
            raise RuntimeError("Google returned HTML instead of file bytes")
        tmp = destination.with_suffix(destination.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(destination)


def download_file(file_info, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return "skip"
    try:
        direct_download(file_info.id, destination)
    except Exception:
        gdown.download(id=file_info.id, output=str(destination), quiet=True, use_cookies=False)
    return "download"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="download only N images; 0 means all")
    parser.add_argument("--workers", type=int, default=16, help="parallel downloads")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] Listing Drive folder {FOLDER_ID} ...")
    files = list_drive_files()

    metadata = [f for f in files if f.path.lower().endswith(".csv")]
    images = [f for f in files if f.path.lower().endswith((".jpg", ".jpeg", ".png"))]
    images.sort(key=lambda x: x.path)

    if args.limit > 0:
        images = images[: args.limit]

    todo = metadata + images
    print(f"[download] metadata files: {len(metadata)}")
    print(f"[download] images to fetch: {len(images)}")
    print(f"[download] workers: {args.workers}")

    def one(f):
        rel = Path(f.path.replace("\\", "/"))
        dst = RAW_DIR / rel
        try:
            status = download_file(f, dst)
            return status, str(rel), None
        except Exception as exc:
            return "error", str(rel), str(exc)

    done = 0
    errors = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(one, f) for f in todo]
        for fut in as_completed(futures):
            status, rel, err = fut.result()
            done += 1
            if err:
                errors.append((rel, err))
            if done <= 10 or done % 100 == 0 or err:
                elapsed = max(0.1, time.time() - start)
                print(
                    f"[download] {done:5d}/{len(todo)} {status:8s} "
                    f"errors={len(errors)} elapsed={elapsed:.1f}s {rel}",
                    flush=True,
                )

    print("[download] Done.")
    if errors:
        print(f"[download] errors: {len(errors)}")
        for rel, err in errors[:20]:
            print(f"  - {rel}: {err}")
    print(f"[download] Dataset root: {RAW_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[download] Interrupted. Run again to resume.")
        sys.exit(130)
