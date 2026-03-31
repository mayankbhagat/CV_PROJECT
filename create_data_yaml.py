# src/create_data_yaml.py
import yaml
from pathlib import Path

def create(data_root, train_sub="train/images", val_sub="val/images", names=None, out="data/data.yaml"):
    d = {
        'train': str(Path(data_root) / train_sub),
        'val': str(Path(data_root) / val_sub),
        'nc': len(names),
        'names': names
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(d, f)
    print("Wrote", out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--names", required=True, nargs="+")
    parser.add_argument("--out", default="data/data.yaml")
    args = parser.parse_args()
    create(args.data_root, names=args.names, out=args.out)
