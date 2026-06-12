# Flow Vision Label

Local label editor for Peru traffic video frames. The app is designed to correct
axis-aligned vehicle boxes, persist reviewed annotations, export YOLO datasets,
and optionally launch YOLO training.

This repository contains only the application code. Dataset files, corrected
labels, exports, model weights, logs, and training runs are intentionally ignored.

## Install

```powershell
python -m pip install -r requirements.txt
```

Training support is optional:

```powershell
python -m pip install -r requirements-train.txt
```

## Download Dataset

The downloader writes into `data/raw/`, which is ignored by git.

```powershell
python download_dataset.py
```

For a small local smoke test:

```powershell
python download_dataset.py --limit 200
```

## Run The Label App

```powershell
python server.py
```

Open:

```text
http://127.0.0.1:8877/
```

On Windows you can also run:

```powershell
.\iniciar_editor_peru.bat
```

## Annotation Flow

- `Guardar imagen actual` persists the current image correction to
  `data/corrected_labels/<image_id>.json`.
- `Aprobar + siguiente` saves and advances.
- `Guardar todo revisado / Exportar YOLO` exports reviewed images to
  `exports/yolo_latest/`.
- `Guardar CSV corregido y sobrescribir metadata.csv` backs up the original CSV
  under `data/backups/` and writes a corrected `data/raw/metadata.csv`.

## Train

Export YOLO from the app first, then run:

```powershell
python train_yolo.py --data exports/yolo_latest/data.yaml --model yolo11n.pt --epochs 50 --imgsz 1280
```

## Repository Hygiene

The following are ignored by design:

- `data/`
- `exports/`
- `runs/`
- `*.pt`, `*.onnx`, `*.engine`
- logs and Python caches
