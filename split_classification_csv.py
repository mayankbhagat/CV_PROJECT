import pandas as pd
from pathlib import Path

BASE = Path("D:/Cliniscan/data")
CSV = BASE / "classification_labels.csv"
TRAIN_IMAGES = BASE / "train" / "images"
VAL_IMAGES = BASE / "val" / "images"

df = pd.read_csv(CSV)

train_files = set([p.name for p in TRAIN_IMAGES.glob("*.jpg")])
val_files = set([p.name for p in VAL_IMAGES.glob("*.jpg")])

train_df = df[df['image'].isin(train_files)].reset_index(drop=True)
val_df = df[df['image'].isin(val_files)].reset_index(drop=True)

train_df.to_csv(BASE/"train_classification.csv", index=False)
val_df.to_csv(BASE/"val_classification.csv", index=False)

print("Train rows:", len(train_df))
print("Val rows:", len(val_df))
