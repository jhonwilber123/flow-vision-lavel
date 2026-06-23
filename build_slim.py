"""Genera data/ai_review/slim.json para el workflow de vision, SOLO con las
imagenes que ya tienen cajas YOLO (ai_labels) pero aun NO tienen resultado de
vision (data/ai_review/results/<stem>.json). Re-ejecutable.
"""
import json
from pathlib import Path

import server

REVIEW = server.ROOT / "data" / "ai_review"
RESULTS = REVIEW / "results"


def fwd(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def main():
    manifest = json.loads((REVIEW / "manifest.json").read_text(encoding="utf-8"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in RESULTS.glob("*.json")}

    slim = []
    for m in manifest:
        stem = Path(m["image_id"]).stem
        if stem in done:
            continue
        overlay = REVIEW / (stem + "_num.jpg")
        if not overlay.exists():
            continue
        slim.append({
            "image_id": m["image_id"],
            "overlay": fwd(overlay),
            "results": fwd(RESULTS / (stem + ".json")),
            "boxes": [{"i": b["i"], "w": b["w"], "h": b["h"], "yolo": b["yolo_class"]}
                      for b in m["boxes"]],
        })

    out = REVIEW / "slim.json"
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"slim.json: {len(slim)} imagenes pendientes de vision -> {fwd(out)}")


if __name__ == "__main__":
    main()
