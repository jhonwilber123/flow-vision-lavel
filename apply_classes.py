"""Aplica las clases refinadas por vision a las cajas de YOLO en data/ai_labels.

Lee data/ai_review/results/<stem>.json  (escritos por los agentes de vision):
    {"image_id": "...jpg", "classes": [{"index": 1, "class_name": "trailer"}, ...]}
y actualiza class_id/class_name de cada caja por indice (1-based) en data/ai_labels/<stem>.json.
"""
import json
from pathlib import Path

import server

RESULTS_DIR = server.ROOT / "data" / "ai_review" / "results"


def main():
    name_to_id = {server.CLASS_NAMES[i]: i for i in server.CLASS_NAMES}
    if not RESULTS_DIR.exists():
        print("no hay results dir")
        return
    total_imgs = 0
    total_changed = 0
    skipped = []
    for rf in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(rf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped.append(rf.name)
            continue
        image_id = data.get("image_id")
        classes = data.get("classes", [])
        ai = server.load_ai_label(image_id)
        if ai is None:
            skipped.append(image_id or rf.name)
            continue
        boxes = ai.get("boxes", [])
        changed = 0
        for c in classes:
            idx = int(c.get("index", 0)) - 1
            cname = c.get("class_name")
            if 0 <= idx < len(boxes) and cname in name_to_id:
                if boxes[idx]["class_name"] != cname:
                    changed += 1
                boxes[idx]["class_id"] = name_to_id[cname]
                boxes[idx]["class_name"] = cname
        server.save_ai_label(image_id, boxes)
        total_imgs += 1
        total_changed += changed
        print(f"  {image_id}: {len(boxes)} cajas, {changed} reclasificadas")
    print(f"OK: {total_imgs} imagenes, {total_changed} cajas reclasificadas. saltadas: {skipped}")


if __name__ == "__main__":
    main()
