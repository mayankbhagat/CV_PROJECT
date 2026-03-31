# src/preprocess.py
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import cv2
import pydicom
from tqdm import tqdm

def read_dicom(path):
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    # use rescale if present
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    arr = arr * slope + intercept
    return arr

def to_uint8(img):
    img = img - np.min(img)
    maxv = np.max(img)
    if maxv <= 0:
        return np.zeros_like(img, dtype=np.uint8)
    img = img / maxv
    img8 = (img * 255.0).astype(np.uint8)
    return img8

def clahe_equalize(img8, clipLimit=2.0, tileGridSize=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img8)

def denoise(img8):
    g = cv2.GaussianBlur(img8, (3,3), 0)
    m = cv2.medianBlur(g, 3)
    return m

def resize_and_pad(img, target_size=(1024,1024)):
    h,w = img.shape[:2]
    th, tw = target_size
    scale = min(th/h, tw/w)
    nh, nw = int(h*scale), int(w*scale)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (th - nh)//2
    left = (tw - nw)//2
    canvas = np.zeros((th, tw), dtype=img_resized.dtype)
    canvas[top:top+nh, left:left+nw] = img_resized
    return canvas, scale, left, top, nh, nw

def process_file(dcm_path, out_path, target=(1024,1024), apply_clahe=True, denoise_flag=True):
    arr = read_dicom(dcm_path)
    orig_h, orig_w = arr.shape[:2]
    img8 = to_uint8(arr)
    if apply_clahe:
        img8 = clahe_equalize(img8)
    if denoise_flag:
        img8 = denoise(img8)
    padded, scale, left, top, nh, nw = resize_and_pad(img8, target)
    rgb = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(str(out_path), rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return {
        'image_id': out_path.stem,
        'scale': scale,
        'left': left,
        'top': top,
        'target_h': target[0],
        'target_w': target[1],
        'orig_h': orig_h,
        'orig_w': orig_w,
        'resized_h': nh,
        'resized_w': nw
    }

def main(dicom_dir, out_dir, target_h=1024, target_w=1024, recursive=True):
    dicom_dir = Path(dicom_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list(dicom_dir.glob("**/*.dcm")) if recursive else list(dicom_dir.glob("*.dcm"))
    records = []
    for f in tqdm(files, desc="Processing DICOMs"):
        out_file = out_dir / (f.stem + ".jpg")
        try:
            rec = process_file(f, out_file, target=(target_h,target_w))
            records.append(rec)
        except Exception as e:
            print("Failed:", f, e)
    df = pd.DataFrame(records)
    map_csv = out_dir.parent / "mapping.csv"
    df.to_csv(map_csv, index=False)
    print(f"Saved mapping to {map_csv}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_h", type=int, default=1024)
    parser.add_argument("--target_w", type=int, default=1024)
    args = parser.parse_args()
    main(args.dicom_dir, args.out_dir, args.target_h, args.target_w)
