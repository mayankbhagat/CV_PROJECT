import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["opacity", "consolidation", "fibrosis", "mass", "other"]

YOLO_WEIGHTS = "models/yolo_best.pt"
CLASSIFIER_WEIGHTS = "models/efficientnet_best.pt"

IMAGES_DIR = "data/images"
OUT_DIR = "runs/full_pipeline"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------
# Classification model (must match train_classification.py)
# -------------------------
def get_classifier_model(num_classes: int):
    """
    Build EfficientNet-B0 exactly like in train_classification.py
    and load models/efficientnet_best.pt
    """
    print("Loading classifier (EfficientNet-B0)...")

    # Use torchvision EfficientNet-B0 (same as training)
    model = models.efficientnet_b0(weights=None)

    # Replace final layer for our 5 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Load trained weights
    state = torch.load(CLASSIFIER_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)  # strict=True is fine now

    model.to(DEVICE)
    model.eval()
    return model


# -------------------------
# YOLO detection model
# -------------------------
def get_yolo_model():
    print("Loading YOLO detector...")
    model = YOLO(YOLO_WEIGHTS)
    return model


# -------------------------
# Preprocess image for classifier
# -------------------------
def build_classifier_transform():
    # Match what we used during training (typical ImageNet-style)
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_image_for_classification(image_path: str, tfm):
    img = Image.open(image_path).convert("RGB")
    tensor = tfm(img).unsqueeze(0)  # [1,3,H,W]
    return tensor.to(DEVICE)


# -------------------------
# Run classifier
# -------------------------
def run_classifier(model, image_tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(image_tensor)  # [1, num_classes]
        probs = torch.sigmoid(logits)[0].cpu().numpy()  # [num_classes]

    class_probs = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
    return class_probs


# -------------------------
# Run YOLO detection
# -------------------------
def run_yolo_detector(model, image_path: str):
    results = model(image_path)[0]  # first (and only) image result
    detections = []

    if results.boxes is None:
        return detections

    boxes = results.boxes.xyxy.cpu().numpy()  # [N,4]
    scores = results.boxes.conf.cpu().numpy()  # [N]
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)

    for box, score, cid in zip(boxes, scores, cls_ids):
        detections.append(
            {
                "bbox": box,  # [x1, y1, x2, y2]
                "score": float(score),
                "class_id": int(cid),
                "class_name": results.names[int(cid)],
            }
        )

    return detections


# -------------------------
# Draw YOLO boxes on image
# -------------------------
def draw_detections(image_path: str, detections, save_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read image: {image_path}")
        return

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        cls_name = det["class_name"]

        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        label = f"{cls_name} {score:.2f}"
        cv2.putText(
            img,
            label,
            (p1[0], p1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(save_path, img)
    print(f"Saved combined YOLO visualization to: {save_path}")


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default="00150343289f317a0ad5629d5b7d9ef9.jpg",
        help="Image filename (in data/images)",
    )
    args = parser.parse_args()

    image_name = args.image
    image_path = str(Path(IMAGES_DIR) / image_name)

    print(f"\n=== Running full pipeline on: {image_name} ===\n")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1) Load models
    print("Loading models...")
    clf_model = get_classifier_model(num_classes=len(CLASS_NAMES))
    yolo_model = get_yolo_model()

    # 2) Classification
    print("\nRunning classification...")
    tfm = build_classifier_transform()
    inp = load_image_for_classification(image_path, tfm)
    class_probs = run_classifier(clf_model, inp)

    print("Classification probabilities:")
    for cls in CLASS_NAMES:
        print(f"  {cls:14s}: {class_probs[cls]:.3f}")

    # 3) Detection
    print("\nRunning YOLO detection...")
    detections = run_yolo_detector(yolo_model, image_path)

    if not detections:
        print("No detections found by YOLO.")
    else:
        print("YOLO detections:")
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            print(
                f"  {det['class_name']:12s} "
                f"({det['score']:.2f}) "
                f"bbox=[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
            )

    # 4) Save visualization with YOLO boxes
    out_img_path = str(Path(OUT_DIR) / f"pipeline_{image_name}")
    draw_detections(image_path, detections, out_img_path)

    print("\nPipeline complete.\n")


if __name__ == "__main__":
    main()
