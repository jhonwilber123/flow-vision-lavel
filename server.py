"""Local AABB label editor for the Peru traffic dataset."""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
IMAGE_DIR = RAW_DIR / "images" / "train"
METADATA_CSV = RAW_DIR / "metadata.csv"
CORRECTED_DIR = ROOT / "data" / "corrected_labels"
EXPORTS_DIR = ROOT / "exports"
BACKUP_DIR = ROOT / "data" / "backups"
HOST = os.environ.get("PERU_LABEL_HOST", "127.0.0.1")
PORT = int(os.environ.get("PERU_LABEL_PORT", "8877"))

DEFAULT_CLASSES = {
    0: "car",
    1: "mototaxi",
    2: "van-minivan",
    3: "motorcycle",
    4: "microbus",
    5: "truck",
    6: "bus",
    7: "person",
    8: "combi",
    9: "trailer",
    10: "taxi",
    11: "bici-triciclo",
    12: "bicycle",
    13: "camioneta",
}

TRAIN_PROC = None
TRAIN_LOG = None


def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def read_image_size(path: Path, fallback=(2304, 1296)):
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return fallback


def sanitize_filename(name):
    return Path(name).name


def corrected_path(image_name):
    return CORRECTED_DIR / (Path(image_name).stem + ".json")


def load_metadata():
    by_file = defaultdict(list)
    class_votes = defaultdict(Counter)
    class_counts = Counter()
    image_dims = {}
    if not METADATA_CSV.exists():
        return by_file, DEFAULT_CLASSES.copy(), class_counts, image_dims

    with METADATA_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = sanitize_filename(row.get("filename_imagen", ""))
            if not name:
                continue
            try:
                class_id = int(float(row.get("class_id", 0)))
                x1 = float(row.get("bbox_x1", 0))
                y1 = float(row.get("bbox_y1", 0))
                x2 = float(row.get("bbox_x2", 0))
                y2 = float(row.get("bbox_y2", 0))
            except ValueError:
                continue
            class_name = (row.get("clase_corregida") or f"class_{class_id}").strip()
            class_votes[class_id][class_name] += 1
            class_counts[class_name] += 1
            try:
                fw = int(float(row.get("frame_width") or 0))
                fh = int(float(row.get("frame_height") or 0))
                if fw > 0 and fh > 0:
                    image_dims[name] = (fw, fh)
            except ValueError:
                pass
            by_file[name].append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "x1": min(x1, x2),
                    "y1": min(y1, y2),
                    "x2": max(x1, x2),
                    "y2": max(y1, y2),
                    "confidence": safe_float(row.get("confidence_original")),
                    "origin_line": row.get("origin_line", ""),
                    "destination_line": row.get("destination_line", ""),
                    "turn_name": row.get("turn_name", ""),
                    "source": "metadata",
                }
            )

    classes = DEFAULT_CLASSES.copy()
    for class_id, votes in class_votes.items():
        if votes:
            classes[class_id] = votes.most_common(1)[0][0]
    return by_file, classes, class_counts, image_dims


def safe_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except ValueError:
        return None


def scan_images():
    if not IMAGE_DIR.exists():
        return []
    return sorted(
        p.name
        for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def load_corrected(name):
    path = corrected_path(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def save_corrected(name, boxes):
    CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": name,
        "reviewed": True,
        "updated_at": now_stamp(),
        "boxes": boxes,
    }
    corrected_path(name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def boxes_for_image(name):
    saved = load_corrected(name)
    if saved is not None:
        return "corrected", saved.get("boxes", [])
    return "metadata", META_BY_FILE.get(name, [])


def reviewed_names():
    if not CORRECTED_DIR.exists():
        return set()
    return {p.stem + ".jpg" for p in CORRECTED_DIR.glob("*.json")}


def image_path(name):
    safe = sanitize_filename(name)
    path = IMAGE_DIR / safe
    if not path.exists():
        return None
    return path


def get_image_dims(name):
    if name in IMAGE_DIMS:
        return IMAGE_DIMS[name]
    path = image_path(name)
    if path:
        return read_image_size(path)
    return 2304, 1296


def normalize_box(box, w, h):
    x1 = max(0.0, min(float(box.get("x1", 0)), float(w)))
    y1 = max(0.0, min(float(box.get("y1", 0)), float(h)))
    x2 = max(0.0, min(float(box.get("x2", 0)), float(w)))
    y2 = max(0.0, min(float(box.get("y2", 0)), float(h)))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    cls = int(float(box.get("class_id", 0)))
    return cls, cx, cy, bw, bh


def write_data_yaml(out_dir, classes, train_has_val=True):
    max_id = max(classes.keys()) if classes else 0
    names = [classes.get(i, f"class_{i}") for i in range(max_id + 1)]
    lines = [
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/val" if train_has_val else "val: images/train",
        "names:",
    ]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def link_or_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_yolo(reviewed_only=True):
    reviewed = reviewed_names()
    if reviewed_only:
        names = [n for n in IMAGE_IDS if n in reviewed]
    else:
        names = IMAGE_IDS[:]
    if not names:
        raise RuntimeError("No hay imagenes revisadas para exportar.")

    out_dir = EXPORTS_DIR / "yolo_latest"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    split_at = int(len(names) * 0.9)
    if len(names) < 10:
        split_at = len(names)
    train_names = set(names[:split_at])
    val_names = [n for n in names if n not in train_names]

    exported = 0
    for name in names:
        split = "train" if name in train_names else "val"
        src = image_path(name)
        if not src:
            continue
        source, boxes = boxes_for_image(name)
        w, h = get_image_dims(name)
        txt_lines = []
        for box in boxes:
            norm = normalize_box(box, w, h)
            if not norm:
                continue
            cls, cx, cy, bw, bh = norm
            txt_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out_dir / "labels" / split / (Path(name).stem + ".txt")).write_text(
            "\n".join(txt_lines) + ("\n" if txt_lines else ""),
            encoding="utf-8",
        )
        link_or_copy(src, out_dir / "images" / split / name)
        exported += 1

    write_data_yaml(out_dir, CLASS_NAMES, train_has_val=bool(val_names))
    return {
        "dir": str(out_dir),
        "data_yaml": str(out_dir / "data.yaml"),
        "images": exported,
        "train": len(train_names),
        "val": len(val_names),
    }


def corrected_row(template, box, image_name, frame_w, frame_h):
    row = dict(template)
    cls = int(float(box.get("class_id", 0)))
    x1 = max(0.0, min(float(box.get("x1", 0)), float(frame_w)))
    y1 = max(0.0, min(float(box.get("y1", 0)), float(frame_h)))
    x2 = max(0.0, min(float(box.get("x2", 0)), float(frame_w)))
    y2 = max(0.0, min(float(box.get("y2", 0)), float(frame_h)))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    cx = ((x1 + x2) / 2.0) / frame_w
    cy = ((y1 + y2) / 2.0) / frame_h
    bw = (x2 - x1) / frame_w
    bh = (y2 - y1) / frame_h

    row["filename_imagen"] = image_name
    row["class_id"] = str(cls)
    row["clase_corregida"] = CLASS_NAMES.get(cls, box.get("class_name", f"class_{cls}"))
    row["bbox_x1"] = str(int(round(x1)))
    row["bbox_y1"] = str(int(round(y1)))
    row["bbox_x2"] = str(int(round(x2)))
    row["bbox_y2"] = str(int(round(y2)))
    row["cx_norm"] = f"{cx:.6f}"
    row["cy_norm"] = f"{cy:.6f}"
    row["w_norm"] = f"{bw:.6f}"
    row["h_norm"] = f"{bh:.6f}"
    row["frame_width"] = str(int(frame_w))
    row["frame_height"] = str(int(frame_h))
    if "confidence_original" in row and box.get("confidence") is not None:
        row["confidence_original"] = str(box.get("confidence"))
    if "origin_line" in row:
        row["origin_line"] = box.get("origin_line", row.get("origin_line", ""))
    if "destination_line" in row:
        row["destination_line"] = box.get("destination_line", row.get("destination_line", ""))
    if "turn_name" in row:
        row["turn_name"] = box.get("turn_name", row.get("turn_name", ""))
    return row


def export_corrected_csv(overwrite=True):
    if not METADATA_CSV.exists():
        raise RuntimeError("No existe data/raw/metadata.csv")
    corrected_files = sorted(CORRECTED_DIR.glob("*.json")) if CORRECTED_DIR.exists() else []
    if not corrected_files:
        raise RuntimeError("No hay correcciones guardadas en data/corrected_labels")

    with METADATA_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        original_rows = list(reader)

    rows_by_image = defaultdict(list)
    for row in original_rows:
        rows_by_image[sanitize_filename(row.get("filename_imagen", ""))].append(row)

    corrected_by_image = {}
    for path in corrected_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        image_name = sanitize_filename(payload.get("image") or (path.stem + ".jpg"))
        corrected_by_image[image_name] = payload.get("boxes", [])

    emitted_corrected = set()
    out_rows = []
    for row in original_rows:
        image_name = sanitize_filename(row.get("filename_imagen", ""))
        if image_name not in corrected_by_image:
            out_rows.append(row)
            continue
        if image_name in emitted_corrected:
            continue
        templates = rows_by_image.get(image_name) or [row]
        frame_w, frame_h = get_image_dims(image_name)
        for i, box in enumerate(corrected_by_image[image_name]):
            template = templates[i] if i < len(templates) else templates[0]
            out_rows.append(corrected_row(template, box, image_name, frame_w, frame_h))
        emitted_corrected.add(image_name)

    for image_name, boxes in corrected_by_image.items():
        if image_name in emitted_corrected:
            continue
        frame_w, frame_h = get_image_dims(image_name)
        blank = {name: "" for name in fieldnames}
        for box in boxes:
            out_rows.append(corrected_row(blank, box, image_name, frame_w, frame_h))

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_path = BACKUP_DIR / f"metadata_original_{now_stamp()}.csv"
    corrected_path_out = RAW_DIR / "metadata_corrected.csv"
    if overwrite:
        shutil.copy2(METADATA_CSV, backup_path)
    with corrected_path_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)
    if overwrite:
        shutil.copy2(corrected_path_out, METADATA_CSV)
        reload_globals()
    return {
        "corrected_images": len(corrected_by_image),
        "rows": len(out_rows),
        "csv": str(corrected_path_out),
        "metadata": str(METADATA_CSV),
        "backup": str(backup_path) if overwrite else "",
        "overwritten": bool(overwrite),
    }


def start_training(params):
    global TRAIN_PROC, TRAIN_LOG
    if TRAIN_PROC is not None and TRAIN_PROC.poll() is None:
        return {"ok": False, "error": "Ya hay un entrenamiento corriendo."}

    data_yaml = EXPORTS_DIR / "yolo_latest" / "data.yaml"
    if not data_yaml.exists():
        export_yolo(reviewed_only=True)

    TRAIN_LOG = ROOT / "runs" / f"train_{now_stamp()}.log"
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "train_yolo.py"),
        "--data",
        str(data_yaml),
        "--model",
        str(params.get("model") or "yolo11n.pt"),
        "--epochs",
        str(int(params.get("epochs") or 50)),
        "--imgsz",
        str(int(params.get("imgsz") or 1280)),
    ]
    if params.get("device"):
        cmd += ["--device", str(params["device"])]

    log_file = TRAIN_LOG.open("w", encoding="utf-8")
    log_file.write(" ".join(cmd) + "\n\n")
    log_file.flush()
    TRAIN_PROC = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {"ok": True, "pid": TRAIN_PROC.pid, "log": str(TRAIN_LOG)}


def train_status():
    running = TRAIN_PROC is not None and TRAIN_PROC.poll() is None
    code = None if TRAIN_PROC is None else TRAIN_PROC.poll()
    tail = ""
    if TRAIN_LOG and TRAIN_LOG.exists():
        data = TRAIN_LOG.read_text(encoding="utf-8", errors="replace")
        tail = data[-6000:]
    return {"running": running, "returncode": code, "log": str(TRAIN_LOG or ""), "tail": tail}


def reload_globals():
    global META_BY_FILE, CLASS_NAMES, CLASS_COUNTS, IMAGE_DIMS, IMAGE_IDS
    META_BY_FILE, CLASS_NAMES, CLASS_COUNTS, IMAGE_DIMS = load_metadata()
    IMAGE_IDS = scan_images()


reload_globals()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, data, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        path = unquote(u.path)

        if path in ("/", "/index.html"):
            self.send_bytes((ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/reload":
            reload_globals()
            self.send_json({"ok": True, "images": len(IMAGE_IDS)})
            return

        if path == "/api/state":
            reviewed = reviewed_names()
            classes = [
                {"id": i, "name": CLASS_NAMES.get(i, f"class_{i}"), "count": CLASS_COUNTS.get(CLASS_NAMES.get(i, ""), 0)}
                for i in sorted(CLASS_NAMES)
            ]
            self.send_json(
                {
                    "images": IMAGE_IDS,
                    "reviewed": [name in reviewed for name in IMAGE_IDS],
                    "initial_counts": [len(META_BY_FILE.get(name, [])) for name in IMAGE_IDS],
                    "classes": classes,
                    "paths": {
                        "root": str(ROOT),
                        "raw": str(RAW_DIR),
                        "images": str(IMAGE_DIR),
                        "metadata": str(METADATA_CSV),
                        "corrected": str(CORRECTED_DIR),
                    },
                    "metadata_exists": METADATA_CSV.exists(),
                    "image_count": len(IMAGE_IDS),
                    "reviewed_count": len(reviewed),
                }
            )
            return

        if path.startswith("/api/image/"):
            name = sanitize_filename(path[len("/api/image/") :])
            p = image_path(name)
            if not p:
                self.send_json({"error": "image not found"}, 404)
                return
            ctype = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            self.send_bytes(p.read_bytes(), ctype)
            return

        if path.startswith("/api/annotations/"):
            name = sanitize_filename(path[len("/api/annotations/") :])
            source, boxes = boxes_for_image(name)
            w, h = get_image_dims(name)
            self.send_json({"image": name, "source": source, "boxes": boxes, "width": w, "height": h})
            return

        if path == "/api/train_status":
            self.send_json(train_status())
            return

        self.send_json({"error": "not found", "path": path}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        path = unquote(u.path)
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n) if n else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_json({"error": "bad json"}, 400)
            return

        if path.startswith("/api/save/"):
            name = sanitize_filename(path[len("/api/save/") :])
            boxes = payload.get("boxes") or []
            clean = []
            for box in boxes:
                try:
                    cls = int(float(box.get("class_id", 0)))
                    clean.append(
                        {
                            "class_id": cls,
                            "class_name": CLASS_NAMES.get(cls, f"class_{cls}"),
                            "x1": float(box["x1"]),
                            "y1": float(box["y1"]),
                            "x2": float(box["x2"]),
                            "y2": float(box["y2"]),
                            "confidence": safe_float(box.get("confidence")),
                            "origin_line": box.get("origin_line", ""),
                            "destination_line": box.get("destination_line", ""),
                            "turn_name": box.get("turn_name", ""),
                            "source": box.get("source", "human"),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            save_corrected(name, clean)
            self.send_json({"ok": True, "reviewed_count": len(reviewed_names())})
            return

        if path == "/api/export_yolo":
            try:
                result = export_yolo(reviewed_only=bool(payload.get("reviewed_only", True)))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, **result})
            return

        if path == "/api/export_csv":
            try:
                result = export_corrected_csv(overwrite=bool(payload.get("overwrite", True)))
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, 400)
                return
            self.send_json({"ok": True, **result})
            return

        if path == "/api/train":
            self.send_json(start_training(payload))
            return

        self.send_json({"error": "not found", "path": path}, 404)


def main():
    print("[init] Peru label app")
    print(f"[init] root: {ROOT}")
    print(f"[init] metadata: {METADATA_CSV} ({'ok' if METADATA_CSV.exists() else 'missing'})")
    print(f"[init] images: {len(IMAGE_IDS)}")
    print(f"[init] corrected: {len(reviewed_names())}")
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\nEditor listo: http://{HOST}:{PORT}\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
        print("\n[stop] detenido")


if __name__ == "__main__":
    main()
