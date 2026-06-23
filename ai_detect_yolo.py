"""Deteccion local con YOLO preentrenado (COCO) -> cajas para el editor.

Uso:
    python ai_detect_yolo.py <image_id> [--model yolo11m.pt] [--conf 0.20] [--imgsz 1280] [--render]

Las cajas se guardan como pre-anotacion de IA en data/ai_labels/<id>.json y, si
existe una correccion previa para esa imagen, se elimina para que la app muestre
las nuevas cajas de YOLO como 'pendiente' (a revisar). Las clases COCO se mapean a
las clases del proyecto; las clases peruanas (mototaxi, combi, camioneta, etc.) no
estan en COCO, asi que se asigna la mas cercana y el humano reclasifica.
"""
import argparse
import sys
from pathlib import Path

import server  # reutiliza image_path, get_image_dims, CLASS_NAMES, etc.

# COCO id -> nombre de clase del proyecto (la mas cercana)
COCO_TO_PROJECT = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def detect(image_id, model_name="yolo11m.pt", conf=0.15, imgsz=1280, render=False,
           iou=0.5, agnostic=True, scales=None):
    import torch
    from torchvision.ops import nms
    from ultralytics import YOLO

    path = server.image_path(image_id)
    if not path:
        print(f"ERROR: imagen no encontrada: {image_id}")
        return 1

    name_to_id = {server.CLASS_NAMES[i]: i for i in server.CLASS_NAMES}
    w, h = server.get_image_dims(image_id)
    scales = scales or [imgsz]

    model = YOLO(model_name)

    # Ensamble multi-escala: junta detecciones de varias resoluciones.
    raw = []  # (x1,y1,x2,y2,conf,proj_name)
    for sz in scales:
        res = model.predict(source=str(path), conf=conf, imgsz=sz, verbose=False)[0]
        for b in res.boxes:
            proj_name = COCO_TO_PROJECT.get(int(b.cls[0]))
            if proj_name is None or proj_name not in name_to_id:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            raw.append((x1, y1, x2, y2, float(b.conf[0]), proj_name))

    boxes = []
    if raw:
        coords = torch.tensor([r[:4] for r in raw], dtype=torch.float32)
        scores = torch.tensor([r[4] for r in raw], dtype=torch.float32)
        keep = nms(coords, scores, iou) if agnostic else range(len(raw))
        for i in (keep.tolist() if hasattr(keep, "tolist") else keep):
            x1, y1, x2, y2, cf, proj_name = raw[i]
            x1 = max(0.0, min(float(w), x1)); x2 = max(0.0, min(float(w), x2))
            y1 = max(0.0, min(float(h), y1)); y2 = max(0.0, min(float(h), y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append(
                {
                    "class_id": name_to_id[proj_name],
                    "class_name": proj_name,
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                    "confidence": round(cf, 3),
                    "origin_line": "", "destination_line": "", "turn_name": "",
                    "source": "ai",
                }
            )

    boxes.sort(key=lambda d: (d["x1"], d["y1"]))
    server.save_ai_label(image_id, boxes)

    # Si habia una correccion previa (basada en intentos viejos), la quitamos para
    # que la app muestre las cajas de YOLO como pendiente / a revisar.
    cp = server.corrected_path(image_id)
    removed = False
    if cp.exists():
        cp.unlink()
        removed = True

    print(f"modelo: {model_name} | imagen {w}x{h} | detecciones: {len(boxes)} | correccion previa eliminada: {removed}")
    for d in boxes:
        print(f"  {d['class_name']:12} conf={d['confidence']:.2f} [{int(d['x1'])},{int(d['y1'])},{int(d['x2'])},{int(d['y2'])}]")

    if render:
        import cv2

        img = cv2.imread(str(path))
        for d in boxes:
            p1 = (int(d["x1"]), int(d["y1"]))
            p2 = (int(d["x2"]), int(d["y2"]))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 3)
            label = f"{d['class_name']} {d['confidence']:.2f}"
            cv2.putText(img, label, (p1[0], max(0, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        out = Path("data") / "ai_labels" / (Path(image_id).stem + "_yolo.jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
        print(f"render: {out}")

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image_id")
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--scales", default="1280,1920", help="escalas multi-pass separadas por coma")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    scales = [int(s) for s in a.scales.split(",") if s.strip()]
    sys.exit(detect(a.image_id, a.model, a.conf, a.imgsz, a.render, a.iou, scales=scales))
