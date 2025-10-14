"""
Predict the activity for recorded sensor zip(s).

Uses the exact same feature order and normalisation statistics saved at training
time (norm_stats.npz), so inference cannot silently drift from training.

Run on every zip in data/test_zips:
    python -m src.har.predict

Run on one zip (or an already-extracted folder):
    python -m src.har.predict path/to/recording.zip
"""
import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .models.cnn_lstm import CNNLSTM
from .preprocess.merge_sensors import merge_sensor_folder
from .preprocess.window_data import window_one_file


def load_stats():
    if not config.NORM_STATS_PATH.exists():
        raise SystemExit(f"Missing {config.NORM_STATS_PATH}. Train the model first.")
    data = np.load(config.NORM_STATS_PATH, allow_pickle=True)
    return data["means"], data["stds"], list(data["labels"])


def load_model(num_classes, device):
    model = CNNLSTM(input_features=config.NUM_FEATURES, num_classes=num_classes)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    return model.to(device).eval()


def find_sensor_folder(root):
    """Locate the folder that actually contains the sensor CSVs inside an extract."""
    root = Path(root)
    items = list(root.iterdir())
    if any(p.suffix.lower() in (".csv", ".xlsx") for p in items):
        return root
    subdirs = [p for p in items if p.is_dir()]
    if len(subdirs) == 1:
        return find_sensor_folder(subdirs[0])
    return root


def extract_zip(zip_path):
    zip_path = Path(zip_path)
    out = config.TEST_EXTRACT_DIR / zip_path.stem
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out)
    return find_sensor_folder(out)


def window_predictions(sensor_folder, model, means, stds, device):
    """Merge + window a recording and return per-window (preds, probs), or None."""
    merged = merge_sensor_folder(sensor_folder)
    if merged is None:
        return None
    windows, _, _ = window_one_file(merged, label=0, group_id=0)
    if not windows:
        return None

    X = np.asarray(windows, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = (X - means) / stds

    with torch.no_grad():
        probs = F.softmax(model(torch.tensor(X, device=device)), dim=-1).cpu().numpy()
    return probs.argmax(axis=1), probs


def predict_folder(sensor_folder, model, means, stds, labels, device):
    out = window_predictions(sensor_folder, model, means, stds, device)
    if out is None:
        return None
    preds, probs = out

    counts = np.bincount(preds, minlength=len(labels))
    winner = int(counts.argmax())
    return {
        "label": str(labels[winner]),
        "n_windows": len(preds),
        "confidence": float(probs[:, winner].mean()),
        "counts": {str(labels[i]): int(c) for i, c in enumerate(counts) if c},
    }


def report(name, result):
    print(f"\n== {name} ==")
    if result is None:
        print("   no usable windows (too little / no sensor data)")
        return
    print(f"   PREDICTION: {result['label']}  "
          f"(mean conf {result['confidence']:.2f}, {result['n_windows']} windows)")
    print(f"   votes: {result['counts']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", help="zip file or extracted folder; "
                    "default = every zip in data/test_zips")
    args = ap.parse_args()

    means, stds, labels = load_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(len(labels), device)
    print(f"Loaded model ({len(labels)} classes) on {device}")

    if args.target:
        target = Path(args.target)
        folder = extract_zip(target) if target.suffix.lower() == ".zip" else target
        report(target.name, predict_folder(folder, model, means, stds, labels, device))
        return

    zips = sorted(config.TEST_ZIP_DIR.glob("*.zip"))
    if not zips:
        print(f"No .zip files in {config.TEST_ZIP_DIR}")
        return
    for zp in zips:
        folder = extract_zip(zp)
        report(zp.name, predict_folder(folder, model, means, stds, labels, device))


if __name__ == "__main__":
    main()
