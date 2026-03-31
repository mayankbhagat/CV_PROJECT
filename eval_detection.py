# src/eval_detection.py

from ultralytics import YOLO

MODEL_PATH = "D:/Cliniscan/models/yolo_best.pt"
DATA_YAML = "D:/Cliniscan/data/data.yaml"

model = YOLO(MODEL_PATH)

metrics = model.val(
    data=DATA_YAML,
    imgsz=1024,
    batch=4,
    device="cpu"
)

print("\n===== YOLO DETECTION METRICS =====")
print("mAP50:     ", metrics.box.map50)
print("mAP50-95:  ", metrics.box.map)
print("Precision: ", metrics.box.mp)
print("Recall:    ", metrics.box.mr)
