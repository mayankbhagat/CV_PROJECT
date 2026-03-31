# src/split_and_organize.py
import argparse
from pathlib import Path
import random
import shutil

def main(images_dir, labels_dir, out_dir, val_ratio=0.2, seed=42, copy=True):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted([p for p in images_dir.glob("*.jpg")] + [p for p in images_dir.glob("*.png")])
    print(f"Found {len(img_paths)} images in {images_dir}")
    random.seed(seed)
    random.shuffle(img_paths)
    n_val = int(len(img_paths) * val_ratio)
    val_imgs = set([p.name for p in img_paths[:n_val]])
    train_imgs = set([p.name for p in img_paths[n_val:]])
    # create folders
    for split in ["train","val"]:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    def copy_file(src, dst):
        if copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)
    # copy images & labels
    for p in img_paths[:n_val]:
        dst_img = out_dir / "val" / "images" / p.name
        copy_file(p, dst_img)
        lbl = labels_dir / (p.stem + ".txt")
        if lbl.exists():
            copy_file(lbl, out_dir / "val" / "labels" / (p.stem + ".txt"))
    for p in img_paths[n_val:]:
        dst_img = out_dir / "train" / "images" / p.name
        copy_file(p, dst_img)
        lbl = labels_dir / (p.stem + ".txt")
        if lbl.exists():
            copy_file(lbl, out_dir / "train" / "labels" / (p.stem + ".txt"))
    print("Done. Train images:", len(list((out_dir/"train"/"images").glob("*"))),
          "Val images:", len(list((out_dir/"val"/"images").glob("*"))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action='store_true', help="copy instead of move")
    args = parser.parse_args()
    # default to copy for safety
    main(args.images_dir, args.labels_dir, args.out_dir, val_ratio=args.val_ratio, seed=args.seed, copy=True)
