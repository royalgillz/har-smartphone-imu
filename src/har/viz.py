"""Plotting helpers for dataset and evaluation figures (saved to docs/plots)."""
import matplotlib
matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np

from . import config

PLOTS_DIR = config.REPO_ROOT / "docs" / "plots"


def _save(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot -> {path.relative_to(config.REPO_ROOT)}")
    return path


def plot_class_distribution(labels, y, name="class_distribution.png"):
    counts = np.bincount(y, minlength=len(labels))
    order = np.argsort(counts)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([labels[i] for i in order], counts[order], color="#4477aa")
    ax.set_xlabel("windows")
    ax.set_title("Windows per class")
    for i, v in enumerate(counts[order]):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    _save(fig, name)


def plot_training_curves(history, name="training_curves.png"):
    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ep, [h["train_loss"] for h in history], "o-", color="#cc3311",
             label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="#cc3311")
    ax2 = ax1.twinx()
    val = [h.get("val_acc") for h in history]
    if any(v is not None for v in val):
        ax2.plot(ep, val, "s-", color="#009988", label="val accuracy")
        ax2.set_ylabel("val accuracy", color="#009988")
        ax2.set_ylim(0, 1)
    ax1.set_title("Training loss & validation accuracy")
    _save(fig, name)


def plot_confusion_matrix(cm, labels, name="confusion_matrix.png", normalize=True):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_disp = np.where(row > 0, cm / row, 0.0)
    else:
        cm_disp = cm
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_disp, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    thresh = 0.5 if normalize else cm_disp.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, f"{cm_disp[i, j]:.2f}" if normalize else int(cm[i, j]),
                        ha="center", va="center", fontsize=6,
                        color="white" if cm_disp[i, j] > thresh else "black")
    _save(fig, name)


def plot_per_class_f1(report_dict, labels, name="per_class_f1.png"):
    f1 = [report_dict.get(lbl, {}).get("f1-score", 0.0) for lbl in labels]
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#bbbbbb" if f1[i] == 0 else "#ee7733" for i in order]
    ax.barh([labels[i] for i in order], [f1[i] for i in order], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-score (test)")
    ax.set_title("Per-class F1 (grey = no test samples)")
    for i, idx in enumerate(order):
        ax.text(f1[idx], i, f" {f1[idx]:.2f}", va="center", fontsize=8)
    _save(fig, name)
