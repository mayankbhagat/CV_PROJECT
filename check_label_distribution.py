# src/check_label_distribution.py
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

CSV = Path("D:/Cliniscan/data/classification_labels.csv")
OUT = Path("D:/Cliniscan/plots")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)
print("Total rows:", len(df))
# how many all-zero
label_cols = ["opacity","consolidation","fibrosis","mass","other"]
df['sum_labels'] = df[label_cols].sum(axis=1)
print("Images with zero labels:", int((df['sum_labels']==0).sum()))
print("Images with any label:", int((df['sum_labels']>0).sum()))
print("\nPer-class counts:")
print(df[label_cols].sum())

# plot
sns.set(style="whitegrid")
counts = df[label_cols].sum().sort_values(ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(x=counts.index, y=counts.values)
plt.title("Class counts (multi-label)")
plt.ylabel("Number of images")
plt.savefig(OUT/"class_counts.png", bbox_inches='tight')
print("Saved plot to", OUT/"class_counts.png")

