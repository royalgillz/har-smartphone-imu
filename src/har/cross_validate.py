"""
Grouped k-fold cross-validation for a trustworthy accuracy estimate.

Uses StratifiedGroupKFold so each recording stays wholly within one fold (no window
leakage) while keeping class balance across folds as well as possible. Reports per-fold
and mean +/- std accuracy and macro-F1. Trains one model per fold, so this is slow on
CPU -- run it in the background.

Run:  python -m src.har.cross_validate [--folds 5] [--epochs 20]
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from . import config
from .augment import augment_batch
from .models.cnn_lstm import CNNLSTM


def train_fold(Xtr_raw, ytr, Xte_raw, num_classes, device, epochs, seed):
    means = Xtr_raw.mean(axis=(0, 1), keepdims=True)
    stds = Xtr_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    channel_std = stds.reshape(-1)

    present = np.unique(ytr)
    w = compute_class_weight("balanced", classes=present, y=ytr)
    weight = np.ones(num_classes, dtype=np.float32)
    weight[present] = w

    model = CNNLSTM(input_features=Xtr_raw.shape[2], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    rng = np.random.default_rng(seed)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    n = len(Xtr_raw)

    model.train()
    for ep in range(epochs):
        perm = np.random.default_rng(seed + ep + 1).permutation(n)
        for i in range(0, n, config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            xb_raw = Xtr_raw[idx]
            if config.USE_AUGMENTATION:
                xb_raw = augment_batch(xb_raw, channel_std, rng)
            xb = torch.tensor((xb_raw - means) / stds, dtype=torch.float32).to(device)
            yb = ytr_t[idx].to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    Xte = (Xte_raw - means) / stds
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 256):
            xb = torch.tensor(Xte[i:i + 256], dtype=torch.float32).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    d = config.WINDOWED_DIR
    X = np.load(d / "X.npy")
    y = np.load(d / "y.npy")
    groups = np.load(d / "groups.npy")
    labels = list(np.load(d / "labels.npy"))
    num_classes = len(labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                random_state=config.RANDOM_SEED)
    print(f"{args.folds}-fold CV on {len(X)} windows / {len(np.unique(groups))} "
          f"recordings, {args.epochs} epochs/fold, device={device}, "
          f"augment={config.USE_AUGMENTATION}\n")

    accs, f1s = [], []
    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups), start=1):
        preds = train_fold(X[tr], y[tr], X[te], num_classes, device,
                           args.epochs, config.RANDOM_SEED + fold)
        acc = accuracy_score(y[te], preds)
        f1 = f1_score(y[te], preds, labels=np.arange(num_classes),
                      average="macro", zero_division=0)
        accs.append(acc)
        f1s.append(f1)
        print(f"  fold {fold}/{args.folds}: acc={acc:.3f}  macro-F1={f1:.3f}  "
              f"(test {len(te)} windows from "
              f"{len(np.unique(groups[te]))} recordings)")

    accs, f1s = np.array(accs), np.array(f1s)
    print(f"\n=== {args.folds}-fold cross-validation ===")
    print(f"accuracy : {accs.mean():.3f} +/- {accs.std():.3f}")
    print(f"macro-F1 : {f1s.mean():.3f} +/- {f1s.std():.3f}")


if __name__ == "__main__":
    main()
