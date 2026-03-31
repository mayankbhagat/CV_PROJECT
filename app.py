# app.py
import os
from pathlib import Path
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
import gradio as gr

# detector: ultralytics YOLO interface
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("ultralytics not available:", e)

# classifier: timm efficientnet
try:
    import timm
except Exception as e:
    timm = None
    print("timm not available:", e)

MODEL_DIR = Path("models")
CLASS_NAMES = ["opacity", "consolidation", "fibrosis", "mass", "other"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_classifier(num_classes=len(CLASS_NAMES), model_path=MODEL_DIR / "efficientnet_best.pt"):
    # build efficientnet-b0 and replace classifier head
    model = timm.create_model('tf_efficientnet_b0_ns', pretrained=False, num_classes=num_classes)  # adjust if you used different variant
    if model_path.exists():
        state = torch.load(str(model_path), map_location=DEVICE)
        # try to be flexible: if saved dict has 'model' or 'state_dict' keys:
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        # attempt to load, allow missing keys
        try:
            model.load_state_dict(state)
        except RuntimeError as e:
            print("Warning: strict load failed, trying non-strict load:", e)
            model.load_state_dict(state, strict=False)
    else:
        print("Classifier model file not found:", model_path)
    model.to(DEVICE).eval()
    return model

def get_detector(model_path=MODEL_DIR / "yolo_best.pt"):
    if YOLO is None:
        raise RuntimeError("ultralytics package required for YOLO detector.")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO weights not found at {model_path}")
    model = YOLO(str(model_path))  # ultralytics YOLO wrapper
    return model

# transforms for classifier
clf_tfm = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# load models on startup
CLASSIFIER = None
DETECTOR = None
try:
    if timm is not None:
        CLASSIFIER = get_classifier()
    if YOLO is not None:
        DETECTOR = get_detector()
except Exception as e:
    print("Model load error:", e)

def run_pipeline(pil_img: Image.Image):
    """
    Input: PIL Image
    Returns: dict with classification probabilities and combined image (PIL) with boxes
    """
    # ensure RGB
    img = pil_img.convert("RGB")
    # 1) classification
    probs = {c: None for c in CLASS_NAMES}
    if CLASSIFIER is not None:
        x = clf_tfm(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = CLASSIFIER(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs_tensor = torch.sigmoid(logits).squeeze().cpu().numpy()
            # If binary or multi-label: ensure shape matches
            for i, name in enumerate(CLASS_NAMES):
                probs[name] = float(probs_tensor[i]) if i < len(probs_tensor) else 0.0
    else:
        for name in CLASS_NAMES:
            probs[name] = 0.0

    # 2) detection (YOLO)
    out_img = img.copy()
    if DETECTOR is not None:
        # ultralytics returns predictions list
        results = DETECTOR(np.array(img), imgsz=1024, conf=0.25, iou=0.45)  # tune as needed
        # results could be a list with Boxes
        # draw boxes
        if results and len(results) > 0:
            res = results[0]
            boxes = getattr(res, 'boxes', None)
            if boxes is not None:
                draw = ImageDraw.Draw(out_img)
                # attempt to get COCO style names; if model trained on your classes, use them
                try:
                    model_names = DETECTOR.names
                except Exception:
                    model_names = {0: "obj"}
                for b in boxes:
                    xyxy = b.xyxy.cpu().numpy().astype(int).tolist()[0]  # ultralytics >=8 returns box objects; adapt as needed
                    x1, y1, x2, y2 = xyxy
                    conf = float(b.conf.cpu().numpy()[0]) if hasattr(b, 'conf') else 0.0
                    cls = int(b.cls.cpu().numpy()[0]) if hasattr(b, 'cls') else -1
                    label = model_names.get(cls, f"class{cls}") if cls>=0 else f"{conf:.2f}"
                    draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
                    draw.text((x1, y1-12), f"{label} {conf:.2f}", fill="red")
    else:
        out_img = img

    return probs, out_img

# Gradio interface
def predict_and_show(image):
    probs, vis = run_pipeline(image)
    # convert PIL to displayable
    return ({k: round(v,3) for k,v in probs.items()}, vis)

title = "CliniScan: Classifier + YOLO Detector"
description = "Upload chest X-ray -> get classification probabilities and YOLO detections. This runs on CPU in the Space."

demo = gr.Interface(
    fn=predict_and_show,
    inputs=gr.inputs.Image(type="pil", label="Upload image"),
    outputs=[gr.outputs.Label(label="Classification (probabilities)"), gr.outputs.Image(type="pil", label="Detections")],
    title=title,
    description=description,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
