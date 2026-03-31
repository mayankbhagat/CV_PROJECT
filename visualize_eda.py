# src/visualize_eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("D:/Cliniscan/data/classification_labels.csv")
OUT_DIR = Path("D:/Cliniscan/plots")
OUT_DIR.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(CSV_PATH)

CLASS_NAMES = ["opacity", "consolidation", "fibrosis", "mass", "other"]

# Total samples
total = len(df)

# Images with any label
df["any_label"] = df[CLASS_NAMES].sum(axis=1)
with_label = (df["any_label"] > 0).sum()
no_label = (df["any_label"] == 0).sum()

print(f"Total images: {total}")
print(f"Images with labels: {with_label}")
print(f"Images without labels: {no_label}")

# Per-class counts
class_counts = df[CLASS_NAMES].sum()

plt.figure(figsize=(8,5))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Classification Label Distribution")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(OUT_DIR / "classification_class_distribution.png")
plt.close()

print("Saved: plots/classification_class_distribution.png")

# Empty vs labeled plot
plt.figure(figsize=(5,4))
sns.barplot(x=["Labeled", "Unlabeled"], y=[with_label, no_label])
plt.title("Labeled vs Unlabeled Images")
plt.tight_layout()
plt.savefig(OUT_DIR / "labeled_vs_unlabeled.png")
plt.close()

print("Saved: plots/labeled_vs_unlabeled.png")
