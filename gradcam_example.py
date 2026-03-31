import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0

import matplotlib.pyplot as plt

# -------------------
# CONFIG
# -------------------
DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
VAL_CSV = DATA_DIR / "val_classification.csv"
MODEL_PATH = Path("models") / "efficientnet_best.pt"
PLOTS_DIR = Path("plots")

CLASS_NAMES = ["opacity", "consolidation", "fibrosis", "mass", "other"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------
# MODEL + TRANSFORMS
# -------------------
def get_model(num_classes: int = 5) -> nn.Module:
    """
    Recreate the SAME EfficientNet-B0 architecture used for training
    (torchvision) and load your fine-tuned weights.
    """
    # base model (no pretrained here; weights come from our checkpoint)
    model = efficientnet_b0(weights=None)

    # replace final classifier to match your training setup (5 outputs)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # load saved state dict
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()
    return model


def get_transforms():
    """
    Use ImageNet-style transforms (same kind as during training).
    """
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


# -------------------
# GRAD-CAM IMPLEMENTATION
# -------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # hooks
        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple; we want the gradients wrt activations
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int):
        """
        x: [1, 3, H, W]
        class_idx: which class to produce Grad-CAM for (0..4)
        """
        self.model.zero_grad()
        out = self.model(x)  # [1, num_classes]

        # multi-label: use raw logit for chosen class
        score = out[0, class_idx]

        score.backward(retain_graph=True)

        # gradients: [1, C, H', W']
        # activations: [1, C, H', W']
        grads = self.gradients  # d(score)/d(A)
        acts = self.activations

        # global-average-pool gradients over spatial dims -> [C,1,1]
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # weighted sum over channels -> [1, 1, H', W']
        cam = (weights * acts).sum(dim=1, keepdim=True)

        # relu
        cam = F.relu(cam)

        # normalize to [0,1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # [H', W']
        return cam[0, 0].cpu().numpy()

    def close(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


# -------------------
# UTILS
# -------------------
def choose_random_labeled_image():
    """
    Pick a random row from classification_val.csv that has at least
    one positive label.
    """
    df = pd.read_csv(VAL_CSV)

    label_cols = CLASS_NAMES
    has_any = df[label_cols].sum(axis=1) > 0
    df_pos = df[has_any].reset_index(drop=True)

    if df_pos.empty:
        raise RuntimeError("No positive-labeled images found in classification_val.csv")

    row = df_pos.sample(1, random_state=random.randint(0, 10_000)).iloc[0]

    img_name = row["image"]
    labels = {c: int(row[c]) for c in label_cols}
    return img_name, labels


def overlay_cam_on_image(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5):
    """
    img_rgb: HxWx3 uint8
    cam: H'xW' float [0,1], will be resized to HxW
    """
    h, w, _ = img_rgb.shape
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(alpha * heatmap + (1 - alpha) * img_rgb)
    return overlay, cam_resized


# -------------------
# MAIN
# -------------------
def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. pick an image
    img_name, labels = choose_random_labeled_image()
    print(f"Using image: {img_name}")
    print(f"True labels: {labels}")

    img_path = IMAGES_DIR / img_name
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # 2. load model (torchvision EfficientNet)
    model = get_model(num_classes=len(CLASS_NAMES))

    # hook into last conv layer in torchvision EfficientNet
    # model.features is a Sequential; last element is Conv2dNormActivation (conv + bn)
    target_layer = model.features[-1][0]  # the Conv2d inside the last block
    cam_extractor = GradCAM(model, target_layer)

    # 3. load image with OpenCV for better display, convert to RGB
    orig_bgr = cv2.imread(str(img_path))
    if orig_bgr is None:
        raise FileNotFoundError(f"Could not read image with cv2: {img_path}")
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    # 4. convert to PIL for transforms
    img_pil = Image.fromarray(orig_rgb)

    # 5. apply transforms -> tensor -> device
    tfm = get_transforms()
    inp = tfm(img_pil).unsqueeze(0).to(DEVICE)  # [1,3,H,W]
    inp.requires_grad_(True)

    # 6. forward pass to get probabilities
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    print("Predicted probabilities:")
    for c, p in zip(CLASS_NAMES, probs):
        print(f"  {c:14s}: {p:.3f}")

    # 7. pick a class that is actually positive for this image, if possible
    positive_classes = [i for i, c in enumerate(CLASS_NAMES) if labels[c] == 1]
    if positive_classes:
        class_idx = positive_classes[0]
    else:
        # fallback to the class with highest predicted prob
        class_idx = int(np.argmax(probs))
        print(
            "No positive ground-truth labels; using predicted max class "
            f"'{CLASS_NAMES[class_idx]}' for Grad-CAM."
        )

    print(f"Generating Grad-CAM for class: {CLASS_NAMES[class_idx]}")

    # 8. run Grad-CAM
    cam = cam_extractor(inp, class_idx)

    # 9. overlay
    overlay, cam_resized = overlay_cam_on_image(orig_rgb, cam, alpha=0.5)

    # 10. save figure with original + heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title(f"Grad-CAM mask ({CLASS_NAMES[class_idx]})")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    out_path = PLOTS_DIR / f"gradcam_{img_name.replace('.jpg', '')}_{CLASS_NAMES[class_idx]}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    cam_extractor.close()

    print(f"Saved Grad-CAM visualization to: {out_path}")


if __name__ == "__main__":
    main()
