from pathlib import Path

base = Path("data")
results = {p.stem for p in (base / "ai_review" / "results").glob("*.json")}
labels = list((base / "ai_labels").glob("*.json"))
deleted = 0
for lp in labels:
    if lp.stem not in results:
        lp.unlink()
        ov = base / "ai_review" / (lp.stem + "_num.jpg")
        if ov.exists():
            ov.unlink()
        deleted += 1
print(f"results(vision)={len(results)}  ai_labels_antes={len(labels)}  borrados={deleted}  quedan={len(labels) - deleted}")
