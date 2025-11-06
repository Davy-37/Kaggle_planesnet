#!/usr/bin/env python3
"""
Neural Networks on PlanesNet (JSON or Images)

Models:
- MLP (on flattened 1200-dim vectors)
- SmallCNN (native 20x20)
- ResNet18 (from torchvision; images upscaled to 64x64)

Features:
- Train/val/test stratified split
- Light augmentations for CNNs
- Mixed precision (AMP) optional
- Cosine LR schedule w/ warmup, early stopping, checkpointing
- Class weighting (optional), label smoothing
- Metrics: Accuracy, ROC-AUC, PR-AUC, confusion matrix, classification report
- Artifacts: best_model.pt, metrics.json, confusion_matrix.png, roc_curve.png, pr_curve.png, curves.png

Usage examples
--------------
# Train a small CNN directly on 20x20
python planesnet_nn.py --json Data/planesnet/planesnet.json --model cnn --epochs 25 --batch-size 256

# Train an MLP on flattened vectors
python planesnet_nn.py --json Data/planesnet/planesnet.json --model mlp --epochs 20

# Train ResNet18 (images upscaled to 64) from JSON
python planesnet_nn.py --json Data/planesnet/planesnet.json --model resnet18 --epochs 30 --batch-size 256 --amp

# Evaluate a saved checkpoint
python planesnet_nn.py --json Data/planesnet/planesnet.json --evaluate --checkpoint runs_nn/best_model.pt

# From an images directory
python planesnet_nn.py --images-dir /path/to/images --model cnn
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve
)
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

try:
    import torchvision
    from torchvision import transforms
except Exception:
    torchvision = None
    transforms = None


# ---------------------- Utils ----------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ---------------------- Datasets ----------------------

class PlanesNetJSON(Dataset):
    def __init__(self, json_path: str, augment: bool = False, upscale_for_torchvision: bool = False):
        with open(json_path, 'r') as f:
            obj = json.load(f)
        self.data = obj['data']
        self.labels = obj['labels']
        self.augment = augment
        self.upscale = upscale_for_torchvision
        # transforms for torchvision backbones (expect 64x64)
        self.tfm = None
        if self.upscale and transforms is not None:
            self.tfm = transforms.Compose([
                transforms.Resize((64,64), interpolation=Image.BILINEAR),
                transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
                transforms.RandomVerticalFlip() if augment else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        arr = np.array(self.data[idx], dtype=np.uint8)
        R = arr[:400].reshape(20, 20)
        G = arr[400:800].reshape(20, 20)
        B = arr[800:].reshape(20, 20)
        img = np.stack([R, G, B], axis=2)
        pil = Image.fromarray(img)

        if self.upscale and self.tfm is not None:
            x = self.tfm(pil)
        else:
            if self.augment:
                pil = random_augment(pil)
            x = torch.from_numpy(np.array(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


class PlanesNetImages(Dataset):
    def __init__(self, root_dir: str, augment: bool = False, upscale_for_torchvision: bool = False):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith('.png')]
        if not self.files:
            raise FileNotFoundError('No .png files found')
        self.augment = augment
        self.upscale = upscale_for_torchvision
        self.tfm = None
        if self.upscale and transforms is not None:
            self.tfm = transforms.Compose([
                transforms.Resize((64,64), interpolation=Image.BILINEAR),
                transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
                transforms.RandomVerticalFlip() if augment else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        fn = os.path.basename(path)
        label_str = fn.split('_', 1)[0]
        try:
            y = int(label_str)
        except Exception:
            y = 1 if label_str.lower().startswith('1') or ('plane' in label_str.lower() and 'no-plane' not in label_str.lower()) else 0
        pil = Image.open(path).convert('RGB')
        if self.upscale and self.tfm is not None:
            x = self.tfm(pil)
        else:
            if self.augment:
                pil = random_augment(pil)
            x = torch.from_numpy(np.array(pil, dtype=np.float32) / 255.0).permute(2, 0, 1)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def random_augment(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < 0.5:
        angle = random.choice([0, 90, 180, 270])
        img = img.rotate(angle)
    return img


# ---------------------- Models ----------------------

class MLP(nn.Module):
    def __init__(self, in_dim=1200, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 2)
        )
    def forward(self, x):
        # x: (B,3,20,20) or flattened
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 20->10
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 10->5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128*5*5, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_resnet18(num_classes=2):
    if torchvision is None:
        raise RuntimeError('torchvision not available; cannot build resnet18')
    m = torchvision.models.resnet18(weights=None)
    # Input will be 64x64; adjust first conv if needed (keep default 7x7 stride 2 acceptable)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# ---------------------- Training/Eval ----------------------

@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 7
    label_smoothing: float = 0.0
    amp: bool = False


def make_splits(labels: List[int], train=0.7, val=0.15, test=0.15, seed=42):
    y = np.array(labels)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1-train), random_state=seed)
    tr, not_tr = next(sss1.split(np.zeros_like(y), y))
    y_not = y[not_tr]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test/(val+test), random_state=seed)
    va_rel, te_rel = next(sss2.split(np.zeros_like(y_not), y_not))
    va = np.array(not_tr)[va_rel]
    te = np.array(not_tr)[te_rel]
    return tr, va, te


def get_class_weights(labels: List[int]):
    labels = np.array(labels)
    pos = (labels==1).sum()
    neg = (labels==0).sum()
    if pos == 0 or neg == 0:
        return None
    # inverse frequency
    w0 = 0.5 * (pos+neg) / (2*neg)
    w1 = 0.5 * (pos+neg) / (2*pos)
    return torch.tensor([w0, w1], dtype=torch.float32)


def train_one_epoch(model, loader, opt, device, loss_fn, scaler=None):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        if scaler is None:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        else:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        total_loss += loss.item()*x.size(0)
        preds = logits.argmax(1)
        correct += (preds==y).sum().item()
        total += x.size(0)
    return total_loss/total, correct/total


def eval_epoch(model, loader, device, loss_fn):
    model.eval()
    total, correct, total_loss = 0, 0, 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            prob = F.softmax(logits, dim=1)[:,1]
            total_loss += loss.item()*x.size(0)
            preds = logits.argmax(1)
            correct += (preds==y).sum().item()
            total += x.size(0)
            all_probs.append(prob.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(labels, probs)
        ap = average_precision_score(labels, probs)
    except Exception:
        auc, ap = float('nan'), float('nan')
    return total_loss/total, correct/total, auc, ap, probs, labels


def plot_confusion_matrix(cm, classes, out_path):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    thresh = cm.max()/2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i,j], 'd'), ha='center', va='center',
                     color='white' if cm[i,j]>thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_curves(hist, out_path):
    fig = plt.figure()
    epochs = np.arange(1, len(hist['train_loss'])+1)
    plt.plot(epochs, hist['train_loss'], label='train loss')
    plt.plot(epochs, hist['val_loss'], label='val loss')
    plt.plot(epochs, hist['train_acc'], label='train acc')
    plt.plot(epochs, hist['val_acc'], label='val acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Training curves')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--json', type=str)
    g.add_argument('--images-dir', type=str)
    p.add_argument('--model', type=str, default='cnn', choices=['mlp','cnn','resnet18'])
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--class-weight', action='store_true', help='use inverse-frequency class weights')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--checkpoint', type=str, default='runs_nn/best_model.pt')
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--num-workers', type=int, default=2)
    args = p.parse_args()

    seed_everything(42)
    ensure_dir('runs_nn')

    # Build dataset
    upscale = (args.model == 'resnet18')
    if args.json:
        full = PlanesNetJSON(args.json, augment=not args.evaluate, upscale_for_torchvision=upscale)
        labels = [int(x) for x in full.labels]
    else:
        full = PlanesNetImages(args.images_dir, augment=not args.evaluate, upscale_for_torchvision=upscale)
        labels = []
        for fp in full.files:
            lbl = os.path.basename(fp).split('_',1)[0]
            try: labels.append(int(lbl))
            except: labels.append(1 if lbl.lower().startswith('1') or ('plane' in lbl.lower() and 'no-plane' not in lbl.lower()) else 0)

    tr_idx, va_idx, te_idx = make_splits(labels)
    train_ds = Subset(full, tr_idx)
    val_ds   = Subset(full, va_idx)
    test_ds  = Subset(PlanesNetJSON(args.json, augment=False, upscale_for_torchvision=upscale) if args.json else PlanesNetImages(args.images_dir, augment=False, upscale_for_torchvision=upscale), te_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    if args.model == 'mlp':
        model = MLP()
    elif args.model == 'cnn':
        model = SmallCNN()
    elif args.model == 'resnet18':
        model = build_resnet18()
    else:
        raise ValueError('unknown model')
    model = model.to(device)

    # Loss
    if args.class_weight:
        cw = get_class_weights(labels)
        loss_fn = nn.CrossEntropyLoss(weight=cw.to(device) if cw is not None else None, label_smoothing=args.label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optim & sched
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Evaluate-only
    if args.evaluate and os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f'Loaded checkpoint: {args.checkpoint}')

    best_auc = -1.0
    patience = args.patience
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

    if not args.evaluate:
        for epoch in range(1, args.epochs+1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, loss_fn, scaler=scaler if args.amp else None)
            va_loss, va_acc, va_auc, va_ap, _, _ = eval_epoch(model, val_loader, device, loss_fn)
            sched.step()

            history['train_loss'].append(tr_loss)
            history['val_loss'].append(va_loss)
            history['train_acc'].append(tr_acc)
            history['val_acc'].append(va_acc)

            print(f'Epoch {epoch:02d}  train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc:.4f} ap {va_ap:.4f}')

            if va_auc > best_auc:
                best_auc = va_auc
                torch.save(model.state_dict(), args.checkpoint)
                print(f'  ↳ New best AUC {best_auc:.4f}. Saved {args.checkpoint}')
                patience = args.patience
            else:
                patience -= 1
                if patience <= 0:
                    print('Early stopping.')
                    break
        # plot curves
        plot_curves(history, 'runs_nn/curves.png')

    # Load best
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f'Evaluating best checkpoint: {args.checkpoint}')

    te_loss, te_acc, te_auc, te_ap, te_probs, te_labels = eval_epoch(model, test_loader, device, loss_fn)
    te_preds = (te_probs >= 0.5).astype(int)
    print(f'Test loss {te_loss:.4f} acc {te_acc:.4f} auc {te_auc:.4f} ap {te_ap:.4f}')
    print(classification_report(te_labels, te_preds, target_names=['no-plane','plane']))

    # Plots & metrics
    cm = confusion_matrix(te_labels, te_preds)
    plot_confusion_matrix(cm, ['no-plane','plane'], 'runs_nn/confusion_matrix.png')

    fpr, tpr, _ = roc_curve(te_labels, te_probs)
    fig = plt.figure(); plt.plot(fpr,tpr,label='ROC'); plt.plot([0,1],[0,1],'--'); plt.legend(); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); fig.savefig('runs_nn/roc_curve.png',dpi=150); plt.close(fig)

    prec, rec, _ = precision_recall_curve(te_labels, te_probs)
    fig = plt.figure(); plt.plot(rec,prec); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve'); fig.savefig('runs_nn/pr_curve.png',dpi=150); plt.close(fig)

    import json as _json
    with open('runs_nn/metrics.json','w') as f:
        _json.dump({
            'test_loss': float(te_loss),
            'test_acc': float(te_acc),
            'test_auc': float(te_auc),
            'test_ap': float(te_ap),
            'confusion_matrix': cm.tolist()
        }, f, indent=2)

    print('Artifacts in ./runs_nn : best_model.pt, metrics.json, confusion_matrix.png, roc_curve.png, pr_curve.png, curves.png')


if __name__ == '__main__':
    main()
