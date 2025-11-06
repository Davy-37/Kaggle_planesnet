#!/usr/bin/env python3
"""
PlanesNet (Planes in Satellite Imagery) - Binary Classification Trainer

Supports training from either the JSON dump (planesnet.json) or the extracted
PNG directory (planesnet.zip -> images). Uses a small CNN in PyTorch with
train/val/test split, basic augmentations, early stopping, and metrics.

Usage examples
--------------
# Train from JSON
python planesnet_train_classifier.py --json /path/to/planesnet.json --epochs 15 --batch-size 256

# Train from images directory (filenames: {label}_{sceneid}_{lon}_{lat}.png)
python planesnet_train_classifier.py --images-dir /path/to/planesnet_images --epochs 15

# Evaluate a saved model on the test split
python planesnet_train_classifier.py --json /path/to/planesnet.json --evaluate --checkpoint runs/best_model.pt

Notes
-----
- Labels: 1 = plane, 0 = no-plane
- Images are 20x20 RGB at 3m/pixel. We keep them small and use light augmentations.
- This script saves:
  - runs/best_model.pt (best validation ROC-AUC)
  - runs/metrics.json (final metrics)
  - runs/confusion_matrix.png
  - runs/roc_curve.png
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt

# ----------------------
# Utilities
# ----------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ----------------------
# Dataset implementations
# ----------------------

class PlanesNetJSON(Dataset):
    def __init__(self, json_path: str, augment: bool = False):
        with open(json_path, 'r') as f:
            obj = json.load(f)
        self.data = obj['data']
        self.labels = obj['labels']
        self.scene_ids = obj.get('scene_ids', None)
        self.locations = obj.get('locations', None)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        arr = np.array(self.data[idx], dtype=np.uint8)
        # Split channels: first 400 R, next 400 G, last 400 B
        R = arr[:400].reshape(20, 20)
        G = arr[400:800].reshape(20, 20)
        B = arr[800:].reshape(20, 20)
        img = np.stack([R, G, B], axis=2)  # HWC
        img = Image.fromarray(img)

        if self.augment:
            img = random_augment(img)

        # To tensor (C,H,W) in [0,1]
        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class PlanesNetImages(Dataset):
    def __init__(self, root_dir: str, augment: bool = False):
        self.files = []
        for fn in os.listdir(root_dir):
            if fn.lower().endswith('.png'):
                self.files.append(os.path.join(root_dir, fn))
        if not self.files:
            raise FileNotFoundError("No .png files found in images directory")
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        fn = os.path.basename(path)
        # expected: {label}_{sceneid}_{lon}_{lat}.png
        label_str = fn.split('_', 1)[0]
        try:
            y = int(label_str)
        except Exception:
            # fallback if filenames are like 'plane'/'no-plane'
            y = 1 if label_str.lower().startswith('1') or 'plane' in label_str.lower() and not 'no-plane' in label_str.lower() else 0
        img = Image.open(path).convert('RGB')

        if self.augment:
            img = random_augment(img)

        x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# ----------------------
# Augmentations (kept light for 20x20)
# ----------------------

def random_augment(img: Image.Image) -> Image.Image:
    # For tiny images, keep augmentations minimal
    # random flips and small rotations
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < 0.5:
        angle = random.choice([0, 90, 180, 270])
        img = img.rotate(angle)
    return img


# ----------------------
# Model
# ----------------------

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)  # 20x20 -> 10x10 -> 5x5
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))  # shape: (B,64,10,10) then pool -> (B,64,5,5)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ----------------------
# Training / Evaluation
# ----------------------

@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 5  # early stopping patience
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_splits(labels: List[int], train_size=0.7, val_size=0.15, test_size=0.15, seed=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    y = np.array(labels)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_size), random_state=seed)
    train_idx, not_train_idx = next(sss1.split(np.zeros_like(y), y))

    y_not_train = y[not_train_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size / (val_size + test_size), random_state=seed)
    val_rel_idx, test_rel_idx = next(sss2.split(np.zeros_like(y_not_train), y_not_train))
    val_idx = np.array(not_train_idx)[val_rel_idx]
    test_idx = np.array(not_train_idx)[test_rel_idx]
    return train_idx, val_idx, test_idx


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            probs = F.softmax(logits, dim=1)[:, 1]
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float('nan')
    return total_loss / total, correct / total, auc, all_probs, all_labels


def plot_confusion_matrix(cm, classes, out_path):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_roc(fpr, tpr, out_path):
    fig = plt.figure()
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


# ----------------------
# Main
# ----------------------

def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--json', type=str, help='Path to planesnet.json')
    src.add_argument('--images-dir', type=str, help='Directory with PNGs')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--evaluate', action='store_true', help='Only evaluate a saved checkpoint on test split')
    parser.add_argument('--checkpoint', type=str, default='runs/best_model.pt')
    args = parser.parse_args()

    seed_everything(42)
    ensure_dir('runs')

    # Build dataset
    if args.json:
        full_ds = PlanesNetJSON(args.json, augment=not args.evaluate)
        labels = [int(x) for x in full_ds.labels]
    else:
        full_ds = PlanesNetImages(args.images_dir, augment=not args.evaluate)
        # Build labels by reading once
        labels = []
        for i in range(len(full_ds.files)):
            fn = os.path.basename(full_ds.files[i])
            lbl = fn.split('_', 1)[0]
            try:
                labels.append(int(lbl))
            except Exception:
                labels.append(1 if lbl.lower().startswith('1') or 'plane' in lbl.lower() and not 'no-plane' in lbl.lower() else 0)

    train_idx, val_idx, test_idx = make_splits(labels)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(PlanesNetJSON(args.json, augment=False) if args.json else PlanesNetImages(args.images_dir, augment=False), test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -1.0
    patience_left = args.patience

    if args.evaluate and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")

    if not args.evaluate:
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
            va_loss, va_acc, va_auc, _, _ = eval_epoch(model, val_loader, device)
            print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc:.4f}")

            if va_auc > best_auc:
                best_auc = va_auc
                torch.save(model.state_dict(), args.checkpoint)
                print(f"  ↳ New best AUC {best_auc:.4f}. Saved to {args.checkpoint}")
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("Early stopping triggered.")
                    break

    # Load best for evaluation
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Evaluating best checkpoint: {args.checkpoint}")

    te_loss, te_acc, te_auc, te_probs, te_labels = eval_epoch(model, test_loader, device)
    te_preds = (te_probs >= 0.5).astype(int)
    print(f"Test loss {te_loss:.4f} acc {te_acc:.4f} auc {te_auc:.4f}")
    print(classification_report(te_labels, te_preds, target_names=['no-plane', 'plane']))

    # Confusion matrix & ROC plots
    cm = confusion_matrix(te_labels, te_preds)
    plot_confusion_matrix(cm, classes=['no-plane', 'plane'], out_path='runs/confusion_matrix.png')

    fpr, tpr, _ = roc_curve(te_labels, te_probs)
    plot_roc(fpr, tpr, out_path='runs/roc_curve.png')

    # Save metrics
    with open('runs/metrics.json', 'w') as f:
        json.dump({
            'test_loss': float(te_loss),
            'test_acc': float(te_acc),
            'test_auc': float(te_auc),
            'confusion_matrix': cm.tolist()
        }, f, indent=2)

    print("Artifacts saved in ./runs : best_model.pt, metrics.json, confusion_matrix.png, roc_curve.png")


if __name__ == '__main__':
    main()
