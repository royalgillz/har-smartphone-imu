"""
Turn merged per-recording CSVs into fixed-length sliding windows.

Input : data/processed/merged/<Activity>/<Sample>.csv
Output: data/processed/windowed/{X.npy, y.npy, groups.npy, labels.npy}

Design choices that fix the old pipeline:
  * Features come from config.FEATURE_COLUMNS (explicit, no timestamps, includes
    orientation quaternion + barometer) -- not suffix matching.
  * Labels come from the activity sub-folder name -- not filename.split('_')[0].
  * Every window records the recording it came from (groups.npy) so training can
    split by recording and avoid leaking overlapping windows across train/test.
  * Windows are saved RAW (only NaN/Inf cleaned). Normalisation is fit on the
    training split inside train.py, so test statistics never leak into scaling.

Run:  python -m src.har.preprocess.window_data
"""
import numpy as np
import pandas as pd

from .. import config


def window_one_file(df, label, group_id):
    """Slice one recording into overlapping windows of fixed feature order."""
    # Guarantee every feature exists and is in the canonical order.
    for col in config.FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    data = df[config.FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    windows, labels, groups = [], [], []
    for start in range(0, len(data) - config.WINDOW_SIZE + 1, config.STEP):
        windows.append(data[start:start + config.WINDOW_SIZE])
        labels.append(label)
        groups.append(group_id)
    return windows, labels, groups


def main():
    if not config.MERGED_DIR.is_dir():
        raise SystemExit(f"Merged dir not found: {config.MERGED_DIR}. "
                         "Run merge_sensors first.")

    activities = sorted(p.name for p in config.MERGED_DIR.iterdir() if p.is_dir())
    if not activities:
        raise SystemExit(f"No activity sub-folders in {config.MERGED_DIR}.")
    act_to_idx = {act: i for i, act in enumerate(activities)}
    print(f"Activities ({len(activities)}): {activities}")
    print(f"Features ({config.NUM_FEATURES}): {config.FEATURE_COLUMNS}")

    X_all, y_all, groups_all = [], [], []
    group_id = 0  # unique id per recording

    for activity in activities:
        files = sorted((config.MERGED_DIR / activity).glob("*.csv"))
        print(f"\n{activity}: {len(files)} recordings")
        for path in files:
            df = pd.read_csv(path)
            w, lab, grp = window_one_file(df, act_to_idx[activity], group_id)
            X_all.extend(w)
            y_all.extend(lab)
            groups_all.extend(grp)
            print(f"  {path.name}: {len(w)} windows (group {group_id})")
            group_id += 1

    X = np.asarray(X_all, dtype=np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    groups = np.asarray(groups_all, dtype=np.int64)

    # Clean NaN/Inf (e.g. a sensor entirely missing for some recording).
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    config.WINDOWED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.WINDOWED_DIR / "X.npy", X)
    np.save(config.WINDOWED_DIR / "y.npy", y)
    np.save(config.WINDOWED_DIR / "groups.npy", groups)
    np.save(config.WINDOWED_DIR / "labels.npy", np.array(activities))

    print(f"\nSaved {X.shape[0]} windows of shape "
          f"{X.shape[1:]} | {len(activities)} classes | "
          f"{len(np.unique(groups))} recordings")
    counts = np.bincount(y, minlength=len(activities))
    print("Windows per class:")
    for i, act in enumerate(activities):
        print(f"  {act}: {counts[i]}")


if __name__ == "__main__":
    main()
