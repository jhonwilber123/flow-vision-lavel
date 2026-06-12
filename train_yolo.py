"""Train a YOLO detector from an exported dataset.

Run after exporting from the label app:
  python train_yolo.py --data exports/yolo_latest/data.yaml --model yolo11n.pt
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="exports/yolo_latest/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="peru_vehicle_detector")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parent / data_path
    if not data_path.exists():
        raise SystemExit(f"data.yaml not found: {data_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install it with:\n"
            "  python -m pip install ultralytics\n"
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=str(Path(__file__).resolve().parent / args.project),
        name=args.name,
    )


if __name__ == "__main__":
    main()
