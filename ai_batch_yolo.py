"""Lote YOLO: detecta cajas precisas en varias imagenes y prepara material para
refinar la clase con vision (Claude).

Salidas:
  - data/ai_labels/<stem>.json     -> cajas (COCO mapeado a clase de proyecto, source ai)
  - data/ai_review/<stem>_num.jpg  -> imagen con cajas numeradas (1..N) para clasificar
  - data/ai_review/manifest.json   -> lista [{image_id, overlay, boxes:[{i,class_name,w,h,cx,cy}]}]

Uso:
  python ai_batch_yolo.py --limit 12 [--pending-only] [--model yolo11m.pt] [--scales 1280,1920]
  python ai_batch_yolo.py --ids id1.jpg id2.jpg ...
"""
import argparse
import json
from pathlib import Path

import server

COCO_TO_PROJECT = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
REVIEW_DIR = server.ROOT / "data" / "ai_review"


def ensemble_boxes(model, path, name_to_id, w, h, conf, scales, iou, device="cpu", half=False):
    import torch
    from torchvision.ops import nms

    raw = []
    for sz in scales:
        res = model.predict(source=str(path), conf=conf, imgsz=sz, verbose=False,
                            device=device, half=half)[0]
        for b in res.boxes:
            proj = COCO_TO_PROJECT.get(int(b.cls[0]))
            if proj is None or proj not in name_to_id:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            raw.append((x1, y1, x2, y2, float(b.conf[0]), proj))
    out = []
    if raw:
        coords = torch.tensor([r[:4] for r in raw], dtype=torch.float32)
        scores = torch.tensor([r[4] for r in raw], dtype=torch.float32)
        for i in nms(coords, scores, iou).tolist():
            x1, y1, x2, y2, cf, proj = raw[i]
            x1 = max(0.0, min(float(w), x1)); x2 = max(0.0, min(float(w), x2))
            y1 = max(0.0, min(float(h), y1)); y2 = max(0.0, min(float(h), y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            out.append((round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), round(cf, 3), proj))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def run(image_ids, model_name, conf, scales, iou, device="cpu", half=False):
    import cv2
    from ultralytics import YOLO

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    name_to_id = {server.CLASS_NAMES[i]: i for i in server.CLASS_NAMES}
    dev = f"cuda:{device}" if str(device).isdigit() else device
    model = YOLO(model_name)
    if dev != "cpu":
        model.to(dev)
    print(f"modelo={model_name} device={dev} half={half} escalas={scales}")
    manifest = []

    for image_id in image_ids:
        path = server.image_path(image_id)
        if not path:
            print(f"  SKIP (no existe): {image_id}")
            continue
        w, h = server.get_image_dims(image_id)
        dets = ensemble_boxes(model, path, name_to_id, w, h, conf, scales, iou, dev, half)

        boxes = [{
            "class_id": name_to_id[proj], "class_name": proj,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": cf,
            "origin_line": "", "destination_line": "", "turn_name": "", "source": "ai",
        } for (x1, y1, x2, y2, cf, proj) in dets]
        server.save_ai_label(image_id, boxes)

        img = cv2.imread(str(path))
        meta = []
        for idx, (x1, y1, x2, y2, cf, proj) in enumerate(dets, start=1):
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 3)
            cv2.putText(img, str(idx), (p1[0] + 3, max(22, p1[1] + 26)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            meta.append({"i": idx, "yolo_class": proj,
                         "w": int(x2 - x1), "h": int(y2 - y1),
                         "cx": int((x1 + x2) / 2), "cy": int((y1 + y2) / 2)})
        overlay = REVIEW_DIR / (Path(image_id).stem + "_num.jpg")
        cv2.imwrite(str(overlay), img)

        manifest.append({"image_id": image_id, "overlay": str(overlay),
                         "img_w": w, "img_h": h, "boxes": meta})
        print(f"  {image_id}: {len(boxes)} cajas")

    (REVIEW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {REVIEW_DIR / 'manifest.json'} ({len(manifest)} imagenes)")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12, help="0 = sin limite (todas)")
    ap.add_argument("--pending-only", action="store_true", default=True)
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="saltar imagenes que ya tienen data/ai_labels/<id>.json")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--scales", default="1280,1920")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="auto", help="auto|cpu|0|cuda:0")
    ap.add_argument("--half", action="store_true")
    a = ap.parse_args()

    device = a.device
    half = a.half
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device, half = "0", True
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

    if a.ids:
        ids = a.ids
    else:
        reviewed = server.reviewed_names()
        pool = [n for n in server.IMAGE_IDS if n not in reviewed] if a.pending_only else server.IMAGE_IDS
        if a.skip_existing:
            pool = [n for n in pool if server.load_ai_label(n) is None]
        ids = pool if a.limit <= 0 else pool[:a.limit]

    scales = [int(s) for s in a.scales.split(",") if s.strip()]
    run(ids, a.model, a.conf, scales, a.iou, device, half)
