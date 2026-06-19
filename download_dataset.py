"""Download the Peru traffic dataset in configurable batches.

Usage:
  python download_dataset.py                         # list + download up to 5 batches of 100
  python download_dataset.py --batch-size 50 --max-batches 3
  python download_dataset.py --workers 8
"""
import argparse
import json
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
BATCHES_DIR = RAW_DIR / "batches"
STATE_FILE = BATCHES_DIR / "state.json"


def list_drive_files():
    print("[download] Listando archivos en Google Drive (puede tardar un momento)...")
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


def download_file(file_id, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return "skip"
    try:
        direct_download(file_id, destination)
    except Exception:
        gdown.download(id=file_id, output=str(destination), quiet=True, use_cookies=False)
    return "download"


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return None


def save_state(state):
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def build_initial_state(files):
    metadata_files = [f for f in files if f.path.lower().endswith(".csv")]
    images = [f for f in files if f.path.lower().endswith((".jpg", ".jpeg", ".png"))]
    images.sort(key=lambda x: x.path)
    return {
        "total_images": len(images),
        "next_index": 0,
        "next_batch_num": 1,
        "metadata_id": metadata_files[0].id if metadata_files else None,
        "metadata_downloaded": False,
        "image_ids": [[f.id, Path(f.path).name] for f in images],
        "batches": [],
    }


def count_active(state):
    return sum(1 for b in state["batches"] if b.get("status") == "active")


def download_one_batch(state, batch_size, workers):
    start_idx = state["next_index"]
    end_idx = min(start_idx + batch_size, state["total_images"])
    if start_idx >= end_idx:
        return None

    num = state["next_batch_num"]
    folder = f"batch_{num:03d}"
    img_dir = BATCHES_DIR / folder / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    slice_ids = state["image_ids"][start_idx:end_idx]
    print(f"\n[download] Lote {num:03d}: imagenes {start_idx + 1}–{end_idx} de {state['total_images']}")

    def one(item):
        fid, name = item
        dst = img_dir / name
        try:
            status = download_file(fid, dst)
            return status, name, None
        except Exception as exc:
            return "error", name, str(exc)

    done, errors = 0, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(one, item) for item in slice_ids]
        for fut in as_completed(futures):
            status, name, err = fut.result()
            done += 1
            if err:
                errors.append((name, err))
            if done <= 5 or done % 25 == 0 or err:
                print(f"  {done:3d}/{len(slice_ids)} {status:8s} {name}", flush=True)

    elapsed = time.time() - t0
    print(f"[download] Lote {num:03d} listo: {done} archivos, {len(errors)} errores, {elapsed:.1f}s")

    batch = {
        "num": num,
        "folder": folder,
        "start": start_idx,
        "count": end_idx - start_idx,
        "images": [item[1] for item in slice_ids],
        "errors": len(errors),
        "status": "active",
    }
    state["batches"].append(batch)
    state["next_index"] = end_idx
    state["next_batch_num"] = num + 1
    return batch


def main():
    parser = argparse.ArgumentParser(description="Descarga el dataset Peru traffic en lotes")
    parser.add_argument("--batch-size", type=int, default=100, help="imagenes por lote (default: 100)")
    parser.add_argument("--max-batches", type=int, default=5, help="max lotes activos a la vez (default: 5)")
    parser.add_argument("--workers", type=int, default=16, help="descargas paralelas (default: 16)")
    args = parser.parse_args()

    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    if state is None:
        files = list_drive_files()
        if not files:
            print("[download] Error: no se pudieron listar los archivos de Drive.")
            sys.exit(1)
        state = build_initial_state(files)
        print(f"[download] {state['total_images']} imagenes encontradas en Drive")

        # Download metadata.csv once
        if state["metadata_id"]:
            meta_dst = RAW_DIR / "metadata.csv"
            if not meta_dst.exists():
                print("[download] Descargando metadata.csv...")
                try:
                    download_file(state["metadata_id"], meta_dst)
                    state["metadata_downloaded"] = True
                    print("[download] metadata.csv descargado")
                except Exception as exc:
                    print(f"[download] Advertencia: no se pudo descargar metadata.csv: {exc}")
        save_state(state)
    else:
        active = count_active(state)
        print(f"[download] Estado: {state['total_images']} total | "
              f"{state['next_index']} descargadas | {active} lotes activos")

    slots = args.max_batches - count_active(state)
    if slots <= 0:
        print(f"[download] Ya hay {count_active(state)} lotes activos (max={args.max_batches}).")
        print("[download] Complete y libere algunos lotes desde la interfaz antes de descargar mas.")
        sys.exit(0)

    if state["next_index"] >= state["total_images"]:
        print("[download] Todas las imagenes ya fueron descargadas.")
        sys.exit(0)

    downloaded = 0
    for _ in range(slots):
        if state["next_index"] >= state["total_images"]:
            break
        batch = download_one_batch(state, args.batch_size, args.workers)
        if batch:
            save_state(state)
            downloaded += 1

    print(f"\n[download] {downloaded} lotes nuevos descargados.")
    print(f"[download] Lotes activos: {count_active(state)}")
    remaining = state["total_images"] - state["next_index"]
    print(f"[download] Imagenes restantes en Drive: {remaining}")
    if downloaded > 0:
        print(f"[download] Editor listo en: http://127.0.0.1:8877")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[download] Interrumpido. Ejecuta de nuevo para continuar.")
        sys.exit(130)
