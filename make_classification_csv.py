import pandas as pd
from pathlib import Path

DATA = Path("D:/Cliniscan/data")
MAP = DATA/"mapping.csv"
ANN = DATA/"annotations_raw.csv"
OUT = DATA/"classification_labels.csv"

# final classes
CLASSES = ["opacity", "consolidation", "fibrosis", "mass", "other"]

# mapping from VinBigData class_name → your classes
MAP_RULES = {
    "Lung Opacity": "opacity",
    "Atelectasis": "opacity",

    "Pulmonary fibrosis": "fibrosis",
    "ILD": "fibrosis",

    "Nodule/Mass": "mass",

    "Aortic enlargement": "other",
    "Cardiomegaly": "other",
    "Pleural thickening": "other",

    "No finding": None       # ignore
}

def main():
    print("Loading files...")
    df_map = pd.read_csv(MAP)
    df_ann = pd.read_csv(ANN)
    
    # build filename column
    df_map["image"] = df_map["image_id"] + ".jpg"

    # map classes
    df_ann["mapped_class"] = df_ann["class_name"].map(MAP_RULES)

    # group annotations
    grouped = df_ann.groupby("image_id")["mapped_class"].apply(
        lambda x: [c for c in x.dropna().tolist()]
    ).reset_index()

    df = df_map.merge(grouped, on="image_id", how="left")

    df["mapped_class"] = df["mapped_class"].apply(lambda x: x if isinstance(x,list) else [])

    # build output
    out = pd.DataFrame()
    out["image"] = df["image"]

    for c in CLASSES:
        out[c] = df["mapped_class"].apply(lambda labels: 1 if c in labels else 0)

    out.to_csv(OUT, index=False)
    print("Saved:", OUT)
    print(out.head(10))

if __name__ == "__main__":
    main()
