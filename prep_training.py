"""Deja todo listo para entrenar:

1) Descarga (full-res, desde Google Drive) las imagenes ETIQUETADAS que faltan
   localmente. Reanudable: salta las que ya existen.
2) Arma el dataset YOLO completo en exports/yolo_full/ (imagenes + labels en
   formato YOLO, split train/val, las 15 clases del proyecto) usando las
   correcciones humanas de data/corrected_labels.

Uso:
  python prep_training.py                 # descarga faltantes + arma export
  python prep_training.py --workers 8     # menos descargas paralelas (anti-quota)
  python prep_training.py --no-download    # solo re-arma el export con lo local
  python prep_training.py --val-frac 0.1
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import server
from download_dataset import download_file  # direct_download + gdown fallback

ROOT = server.ROOT
RAW = ROOT / "data" / "raw"
STATE = RAW / "batches" / "state.json"
DL_DIR = RAW / "labeled_full"           # imagenes etiquetadas re-descargadas
CORR = ROOT / "data" / "corrected_labels"
OUT = ROOT / "exports" / "yolo_full"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def index_local():
    """stem -> Path de la primera imagen local encontrada."""
    idx = {}
    for base in (RAW, ROOT / "exports"):
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                idx.setdefault(p.stem, p)
    return idx


def download_missing(missing_stems, fid_by_stem, workers):
    DL_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(s, *fid_by_stem[s]) for s in missing_stems if s in fid_by_stem]
    no_fid = [s for s in missing_stems if s not in fid_by_stem]
    if no_fid:
        print(f"[warn] {len(no_fid)} etiquetadas sin id en Drive (no se pueden bajar)")
    print(f"[download] a descargar: {len(todo)} imagenes (workers={workers})")

    ok, skip, errs = 0, 0, []

    def one(t):
        stem, fid, name = t
        dst = DL_DIR / name
        try:
            status = download_file(fid, dst)
            return stem, dst, status, None
        except Exception as exc:  # noqa: BLE001
            return stem, None, "error", str(exc)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(one, t) for t in todo]
        for fut in as_completed(futures):
            stem, dst, status, err = fut.result()
            done += 1
            if err:
                errs.append((stem, err))
            elif status == "skip":
                skip += 1
            else:
                ok += 1
            if done % 50 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}  bajadas={ok} saltadas={skip} errores={len(errs)}", flush=True)

    print(f"[download] listo: bajadas={ok} saltadas={skip} errores={len(errs)}")
    if errs:
        print("[download] primeros errores:")
        for stem, e in errs[:5]:
            print(f"   {stem}: {e}")
    return ok, errs


def build_export(val_frac):
    local = index_local()  # re-index tras descargar
    corrected = sorted(p.stem for p in CORR.glob("*.json"))
    have = [s for s in corrected if s in local]
    missing = [s for s in corrected if s not in local]
    print(f"[export] etiquetadas={len(corrected)} con_imagen={len(have)} sin_imagen={len(missing)}")

    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    order = have[:]
    random.shuffle(order)
    n_val = max(1, int(len(order) * val_frac)) if order else 0
    val_set = set(order[:n_val])

    n_box = 0
    per_split = {"train": 0, "val": 0}
    for stem in have:
        split = "val" if stem in val_set else "train"
        img = local[stem]
        w, h = server.read_image_size(img)
        data = json.loads((CORR / (stem + ".json")).read_text(encoding="utf-8"))
        lines = []
        for b in data.get("boxes", []):
            nb = server.normalize_box(b, w, h)
            if nb is None:
                continue
            cls, cx, cy, bw, bh = nb
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_box += 1
        server.link_or_copy(img, OUT / "images" / split / img.name)
        (OUT / "labels" / split / (stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        per_split[split] += 1

    server.write_data_yaml(OUT, server.CLASS_NAMES, train_has_val=True)
    print(f"[export] train={per_split['train']} val={per_split['val']} cajas={n_box}")
    print(f"[export] dataset listo: {OUT / 'data.yaml'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()

    if not args.no_download:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        fid_by_stem = {Path(n).stem: (fid, n) for fid, n in state["image_ids"]}
        corrected = sorted(p.stem for p in CORR.glob("*.json"))
        local = index_local()
        missing = [s for s in corrected if s not in local]
        print(f"[plan] etiquetadas={len(corrected)} con_imagen={len(corrected) - len(missing)} faltan={len(missing)}")
        if missing:
            download_missing(missing, fid_by_stem, args.workers)

    build_export(args.val_frac)


if __name__ == "__main__":
    main()
