# src/convert_annotations.py
import argparse
from pathlib import Path
import pandas as pd
import cv2
from tqdm import tqdm
import os

def remap_box(orig_box, mapping_row):
    # orig_box in original DICOM pixel coords
    x_min, y_min, x_max, y_max = orig_box
    scale = mapping_row['scale']
    left = mapping_row['left']
    top = mapping_row['top']
    # After scaling the original coordinates:
    x_min_s = x_min * scale + left
    x_max_s = x_max * scale + left
    y_min_s = y_min * scale + top
    y_max_s = y_max * scale + top
    return x_min_s, y_min_s, x_max_s, y_max_s

def to_yolo_norm(x_min, y_min, x_max, y_max, img_w, img_h):
    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h
    return cx, cy, w, h

def main(csv_path, mapping_csv, images_dir, out_dir, classes_order=None, img_ext=".jpg"):
    df = pd.read_csv(csv_path)
    mapping = pd.read_csv(mapping_csv).set_index('image_id')
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # determine classes
    if classes_order is None:
        classes = sorted(df['class_name'].unique().tolist())
        class_to_idx = {c:i for i,c in enumerate(classes)}
    else:
        classes = classes_order
        class_to_idx = {c:i for i,c in enumerate(classes)}
    for image_id, rows in tqdm(df.groupby('image_id'), desc="Converting annotations"):
        if str(image_id) not in mapping.index:
            # image missing from processed images
            continue
        map_row = mapping.loc[str(image_id)]
        img_path = images_dir / (str(image_id) + img_ext)
        if not img_path.exists():
            continue
        import cv2
        img = cv2.imread(str(img_path))
        h,w = img.shape[:2]
        lines = []
        for _, r in rows.iterrows():
            orig = (r['x_min'], r['y_min'], r['x_max'], r['y_max'])
            x1,y1,x2,y2 = remap_box(orig, map_row)
            cx,cy,nw,nh = to_yolo_norm(x1,y1,x2,y2, w, h)
            # clip values to (0,1)
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(1e-6, min(1.0, nw))
            nh = max(1e-6, min(1.0, nh))
            cls_idx = class_to_idx[r['class_name']]
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if lines:
            with open(out_dir / (str(image_id) + ".txt"), "w") as f:
                f.write("\n".join(lines))
    # write classes file
    with open(out_dir / "classes.txt", "w") as f:
        for c in classes:
            f.write(c + "\n")
    print("Wrote YOLO labels to", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--img_ext", default=".jpg")
    args = parser.parse_args()
    main(args.csv, args.mapping, args.images, args.out, img_ext=args.img_ext)
