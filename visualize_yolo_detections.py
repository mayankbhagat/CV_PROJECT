# src/visualize_yolo_detections.py

from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "D:/Cliniscan/models/yolo_best.pt"
IMG_DIR = Path("D:/Cliniscan/data/val/images")
OUT_DIR = Path("D:/Cliniscan/runs/yolo_visuals")

OUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH)

print("Running YOLO visualization on validation images...")

images = list(IMG_DIR.glob("*.jpg"))[:20]  # visualize only first 20

for img in images:
    results = model(img)
    for r in results:
        save_path = OUT_DIR / img.name
        r.save(filename=str(save_path))
        print("Saved:", save_path)

print("YOLO visualization complete.")
