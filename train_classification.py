# src/train_classification.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Config
DATA_CSV = Path("D:/Cliniscan/data/classification_labels.csv")
IMG_DIR = Path("D:/Cliniscan/data/images")
OUT_DIR = Path("D:/Cliniscan/models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_CLASSES = 5
BATCH = 16
EPOCHS = 12
IMG_SIZE = 384   # 224 or 384; try 224 then increase to 384 if GPU permits
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# transforms
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(6),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class MultiLabelDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.cols = df.columns.tolist()
        self.label_cols = [c for c in self.cols if c!='image']
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['image']
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return img, labels

def compute_metrics(y_true, y_prob, threshold=0.5):
    aucs = []
    f1s = []
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:,i], y_prob[:,i])
        except:
            auc = np.nan
        pred = (y_prob[:,i] >= threshold).astype(int)
        try:
            f1 = f1_score(y_true[:,i], pred)
        except:
            f1 = np.nan
        aucs.append(auc); f1s.append(f1)
    return aucs, f1s

def plot_metrics(history, out_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    df = pd.DataFrame(history)
    plt.figure(figsize=(8,4))
    plt.plot(df['train_loss'], label='train_loss')
    plt.plot(df['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig(out_dir/'loss_curve.png')
    plt.close()

def main():
    df = pd.read_csv(DATA_CSV)
    # quick shuffle + split
    dataset = MultiLabelDataset(df, IMG_DIR, transform=train_tf)
    n = len(dataset)
    val_n = int(0.15 * n)
    train_n = n - val_n
    train_set, val_set = random_split(dataset, [train_n, val_n])
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    # model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {'train_loss':[],'val_loss':[]}

    best_auc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        train_loss = running / len(train_loader.dataset)

        # validation
        model.eval()
        v_loss = 0.0
        preds = []
        trues = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels.to(DEVICE))
                v_loss += loss.item() * imgs.size(0)
                probs = torch.sigmoid(out).cpu().numpy()
                preds.append(probs)
                trues.append(labels.numpy())
        val_loss = v_loss / len(val_loader.dataset)
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        aucs, f1s = compute_metrics(trues, preds)
        mean_auc = np.nanmean(aucs)

        print(f"Epoch {epoch+1}/{EPOCHS} train_loss={train_loss:.4f} val_loss={val_loss:.4f} mean_auc={mean_auc:.4f}")
        print("AUCs:", ["{:.3f}".format(a) for a in aucs])
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # save best by mean AUC
        if mean_auc > best_auc:
            best_auc = mean_auc
            torch.save(model.state_dict(), OUT_DIR/'efficientnet_best.pt')
            print("Saved best model. mean_auc=", best_auc)

    plot_metrics(history, OUT_DIR)
    print("Training complete. best mean AUC:", best_auc)

if __name__ == "__main__":
    main()
