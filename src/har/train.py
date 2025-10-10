"""
Train and evaluate the CNN-LSTM on windowed HAR data.

Correctness properties:
  * Split is by RECORDING (train/val/test), not by window, so overlapping windows
    from the same recording never straddle splits (the old random split inflated
    accuracy). Per class: >=3 recordings -> train/val/test; ==2 -> train/test;
    ==1 -> train only (cannot be honestly evaluated yet).
  * Normalisation is fit on TRAIN only and saved to norm_stats.npz for inference.
  * Class-weighted cross-entropy compensates for heavy class imbalance.
  * The model with the best validation accuracy is kept (falls back to last epoch
    when there is no validation set).
  * Saves dataset + evaluation plots to docs/plots/.

Run:  python -m src.har.train
"""
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from . import config, viz
from .augment import augment_batch
from .models.cnn_lstm import CNNLSTM


def load_windows():
    d = config.WINDOWED_DIR
    return (np.load(d / "X.npy"), np.load(d / "y.npy"),
            np.load(d / "groups.npy"), np.load(d / "labels.npy"))


def split_by_recording(y, groups, seed):
    """Assign whole recordings to train/val/test so no window leaks across splits."""
    rng = np.random.default_rng(seed)
    group_label = {g: int(y[groups == g][0]) for g in np.unique(groups)}
    by_class = defaultdict(list)
    for g, cls in group_label.items():
        by_class[cls].append(g)

    train_g, val_g, test_g = [], [], []
    for gs in by_class.values():
        gs = list(gs)
        rng.shuffle(gs)
        n = len(gs)
        if n == 1:
            train_g += gs
        elif n == 2:
            test_g += gs[:1]
            train_g += gs[1:]
        else:
            n_test = max(1, round(0.15 * n))
            n_val = max(1, round(0.15 * n))
            test_g += gs[:n_test]
            val_g += gs[n_test:n_test + n_val]
            train_g += gs[n_test + n_val:]

    masks = tuple(np.isin(groups, g) for g in (train_g, val_g, test_g))
    return masks


def accuracy(model, X, y, device, batch=256):
    if len(X) == 0:
        return None
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return float((np.concatenate(preds) == y).mean())


def main():
    X, y, groups, labels = load_windows()
    labels = list(labels)
    num_classes = len(labels)
    print(f"Loaded {X.shape[0]} windows | shape {X.shape[1:]} | "
          f"{num_classes} classes | {len(np.unique(groups))} recordings")

    tr, va, te = split_by_recording(y, groups, config.RANDOM_SEED)
    Xtr_raw, ytr = X[tr], y[tr]
    Xva_raw, yva = X[va], y[va]
    Xte_raw, yte = X[te], y[te]
    print(f"Recordings -> train {len(np.unique(groups[tr]))}, "
          f"val {len(np.unique(groups[va]))}, test {len(np.unique(groups[te]))}")
    print(f"Windows    -> train {len(Xtr_raw)}, val {len(Xva_raw)}, test {len(Xte_raw)}")

    untestable = [labels[i] for i in range(num_classes) if i not in set(yte.tolist())]
    if untestable:
        print(f"[note] no test data for: {untestable}")

    # normalise on TRAIN only, save stats for inference
    means = Xtr_raw.mean(axis=(0, 1), keepdims=True)
    stds = Xtr_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    channel_std = stds.reshape(-1)  # (F,) for jitter scaling
    Xva = (Xva_raw - means) / stds
    Xte = (Xte_raw - means) / stds
    config.WINDOWED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(config.NORM_STATS_PATH, means=means, stds=stds,
             feature_cols=np.array(config.FEATURE_COLUMNS, dtype=object),
             labels=np.array(labels))
    print(f"Saved normalisation stats -> "
          f"{config.NORM_STATS_PATH.relative_to(config.REPO_ROOT)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    present = np.unique(ytr)
    w = compute_class_weight("balanced", classes=present, y=ytr)
    weight = np.ones(num_classes, dtype=np.float32)
    weight[present] = w
    class_weight = torch.tensor(weight, device=device)

    model = CNNLSTM(input_features=X.shape[2], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    aug_rng = np.random.default_rng(config.RANDOM_SEED)
    print(f"\nTraining on {device} for {config.EPOCHS} epochs"
          f"{' (augmented)' if config.USE_AUGMENTATION else ''}...")
    history, best_val, best_state = [], -1.0, None
    n = len(Xtr_raw)
    for epoch in range(config.EPOCHS):
        model.train()
        perm = torch.randperm(n).numpy()
        total = 0.0
        for i in range(0, n, config.BATCH_SIZE):
            idx = perm[i:i + config.BATCH_SIZE]
            xb_raw = Xtr_raw[idx]
            if config.USE_AUGMENTATION:
                xb_raw = augment_batch(xb_raw, channel_std, aug_rng)
            xb = torch.tensor((xb_raw - means) / stds, dtype=torch.float32).to(device)
            yb = ytr_t[idx].to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        val_acc = accuracy(model, Xva, yva, device)
        history.append({"epoch": epoch + 1, "train_loss": total, "val_acc": val_acc})
        msg = f"  epoch {epoch + 1:2d}/{config.EPOCHS}  loss={total:.4f}"
        if val_acc is not None:
            msg += f"  val_acc={val_acc:.3f}"
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val_acc={best_val:.3f})")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"Saved model -> {config.MODEL_PATH.relative_to(config.REPO_ROOT)}")

    # ---- evaluation ----
    viz.plot_class_distribution(labels, y)
    viz.plot_training_curves(history)

    if len(Xte) == 0:
        print("\n[warn] empty test set -- nothing to evaluate.")
        return

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(Xte, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()

    test_acc = float((preds == yte).mean())
    rep = classification_report(yte, preds, labels=np.arange(num_classes),
                                target_names=labels, zero_division=0)
    rep_d = classification_report(yte, preds, labels=np.arange(num_classes),
                                  target_names=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(yte, preds, labels=np.arange(num_classes))

    print(f"\n=== Held-out TEST (split by recording) ===")
    print(f"accuracy={test_acc:.3f}  macro-F1={rep_d['macro avg']['f1-score']:.3f}  "
          f"weighted-F1={rep_d['weighted avg']['f1-score']:.3f}")
    print(rep)

    viz.plot_confusion_matrix(cm, labels)
    viz.plot_per_class_f1(rep_d, labels)


if __name__ == "__main__":
    main()
