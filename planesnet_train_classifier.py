import argparse
import json
import os
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Optional, Tuple


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mkdir_runs(path):
    os.makedirs(path, exist_ok=True)


# --- Datasets ---

class PlanesNetJSON(Dataset):
    def __init__(self, json_path, augment=False):
        with open(json_path) as f:
            obj = json.load(f)
        self.data = obj["data"]
        self.labels = obj["labels"]
        self.scene_ids = obj.get("scene_ids")
        self.locations = obj.get("locations")
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        arr = np.array(self.data[idx], dtype=np.uint8)
        R = arr[:400].reshape(20, 20)
        G = arr[400:800].reshape(20, 20)
        B = arr[800:].reshape(20, 20)
        img = np.stack([R, G, B], axis=2)
        img = Image.fromarray(img)
        if self.augment:
            img = random_augment(img)
        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class PlanesNetImages(Dataset):
    def __init__(self, root_dir, augment=False):
        self.files = []
        for fn in os.listdir(root_dir):
            if fn.lower().endswith(".png"):
                self.files.append(os.path.join(root_dir, fn))
        if not self.files:
            raise FileNotFoundError("Aucun .png dans le dossier")
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        fn = os.path.basename(path)
        label_str = fn.split("_", 1)[0]
        try:
            y = int(label_str)
        except Exception:
            y = 1 if (label_str.lower().startswith("1") or ("plane" in label_str.lower() and "no-plane" not in label_str.lower())) else 0
        img = Image.open(path).convert("RGB")
        if self.augment:
            img = random_augment(img)
        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return x, torch.tensor(y, dtype=torch.long)


def random_augment(img):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < 0.5:
        img = img.rotate(random.choice([0, 90, 180, 270]))
    return img


# --- Modèle ---

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_splits(labels: List[int], train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    y = np.array(labels)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_size), random_state=seed)
    train_idx, not_train_idx = next(sss1.split(np.zeros_like(y), y))
    y_not = y[not_train_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size / (val_size + test_size), random_state=seed)
    val_rel, test_rel = next(sss2.split(np.zeros_like(y_not), y_not))
    val_idx = np.array(not_train_idx)[val_rel]
    test_idx = np.array(not_train_idx)[test_rel]
    return train_idx, val_idx, test_idx


def one_epoch_train(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def one_epoch_eval(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            probs = F.softmax(logits, dim=1)[:, 1]
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")
    return total_loss / total, correct / total, auc, all_probs, all_labels


def draw_confusion(cm, classes, out_path):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matrice de confusion")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("Vrai label")
    plt.xlabel("Prédit")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def draw_roc(fpr, tpr, out_path):
    fig = plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", type=str)
    src.add_argument("--images-dir", type=str)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="runs/best_model.pt")
    args = parser.parse_args()

    fix_seed(42)
    mkdir_runs("runs")

    if args.json:
        full_ds = PlanesNetJSON(args.json, augment=not args.evaluate)
        labels = [int(x) for x in full_ds.labels]
    else:
        full_ds = PlanesNetImages(args.images_dir, augment=not args.evaluate)
        labels = []
        for i in range(len(full_ds.files)):
            fn = os.path.basename(full_ds.files[i])
            lbl = fn.split("_", 1)[0]
            try:
                labels.append(int(lbl))
            except Exception:
                labels.append(1 if lbl.lower().startswith("1") or ("plane" in lbl.lower() and "no-plane" not in lbl.lower()) else 0)

    train_idx, val_idx, test_idx = make_splits(labels)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(
        PlanesNetJSON(args.json, augment=False) if args.json else PlanesNetImages(args.images_dir, augment=False),
        test_idx,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -1.0
    patience_left = args.patience

    if args.evaluate and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("Checkpoint chargé:", args.checkpoint)

    if not args.evaluate:
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = one_epoch_train(model, train_loader, optimizer, device)
            va_loss, va_acc, va_auc, _, _ = one_epoch_eval(model, val_loader, device)
            print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc:.4f}")

            if va_auc > best_auc:
                best_auc = va_auc
                torch.save(model.state_dict(), args.checkpoint)
                print(f"  -> nouveau best AUC {best_auc:.4f}, sauvegardé")
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("Early stopping.")
                    break

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("Évaluation du meilleur checkpoint:", args.checkpoint)

    te_loss, te_acc, te_auc, te_probs, te_labels = one_epoch_eval(model, test_loader, device)
    te_preds = (te_probs >= 0.5).astype(int)
    print(f"Test loss {te_loss:.4f} acc {te_acc:.4f} auc {te_auc:.4f}")
    print(classification_report(te_labels, te_preds, target_names=["no-plane", "plane"]))

    cm = confusion_matrix(te_labels, te_preds)
    draw_confusion(cm, ["no-plane", "plane"], "runs/confusion_matrix.png")
    fpr, tpr, _ = roc_curve(te_labels, te_probs)
    draw_roc(fpr, tpr, "runs/roc_curve.png")

    with open("runs/metrics.json", "w") as f:
        json.dump({
            "test_loss": float(te_loss),
            "test_acc": float(te_acc),
            "test_auc": float(te_auc),
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)

    print("Résultats dans ./runs : best_model.pt, metrics.json, confusion_matrix.png, roc_curve.png")


if __name__ == "__main__":
    main()
