"""
Merge per-sensor CSV files of each recording into one timestamp-aligned CSV.

Input : data/raw/<Activity>/<Sample>/<Sensor>.csv
Output: data/processed/merged/<Activity>/<Sample>.csv

The output is organised as one sub-folder per activity so the class label can be
read directly from the directory name later -- no fragile filename parsing.

Run:  python -m src.har.preprocess.merge_sensors
"""
import pandas as pd

from .. import config


def load_sensor_file(path):
    """Read a sensor file, returning an empty DataFrame (with a warning) on error."""
    try:
        if str(path).lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_excel(path)
    except Exception as e:  # noqa: BLE001 - report instead of silently swallowing
        print(f"    [warn] could not read {path}: {e}")
        return pd.DataFrame()


def extract_timestamp(df):
    """
    Keep a single numeric `seconds_elapsed` timestamp column and drop every other
    time-like column. Returns (clean_df, ts_column_name) or (None, None).
    """
    df = df.copy()

    sec_cols = [c for c in df.columns if "seconds" in c.lower()]
    if sec_cols:
        ts = sec_cols[0]
    else:
        ts_candidates = [c for c in df.columns if "time" in c.lower()]
        if not ts_candidates:
            return None, None
        ts = ts_candidates[0]

    df[ts] = pd.to_numeric(df[ts], errors="coerce")
    df = df.dropna(subset=[ts]).sort_values(ts).reset_index(drop=True)

    # Remove all other time-like columns so they never leak in as features.
    for col in list(df.columns):
        if col != ts and ("time" in col.lower() or "second" in col.lower()):
            df = df.drop(columns=[col], errors="ignore")

    return df, ts


def merge_asof_safe(left, right, left_key, right_key, tolerance):
    left = left.dropna(subset=[left_key]).sort_values(left_key)
    right = right.dropna(subset=[right_key]).sort_values(right_key)
    return pd.merge_asof(
        left, right,
        left_on=left_key, right_on=right_key,
        direction="nearest", tolerance=tolerance,
    )


def merge_sensor_folder(folder):
    """
    Merge every sensor CSV/XLSX found directly in `folder` into one aligned
    DataFrame. Shared by the training pipeline and zip-based inference so they
    use identical merge logic. Returns the merged DataFrame or None.
    """
    from pathlib import Path
    folder = Path(folder)

    merged = None
    merged_key = None

    for sensor in config.SENSORS:
        path = None
        for ext in ("csv", "xlsx"):
            candidate = folder / f"{sensor}.{ext}"
            if candidate.exists():
                path = candidate
                break
        if path is None:
            print(f"    [skip] missing sensor: {sensor}")
            continue

        df = load_sensor_file(path)
        if df.empty:
            print(f"    [skip] empty: {path.name}")
            continue

        df, ts = extract_timestamp(df)
        if df is None:
            print(f"    [skip] no timestamp in {path.name}")
            continue

        df = df.add_prefix(sensor + "_")
        df = df.rename(columns={f"{sensor}_{ts}": f"{sensor}_ts"})

        if merged is None:
            merged = df.copy()
            merged_key = f"{sensor}_ts"
        else:
            merged = merge_asof_safe(merged, df, merged_key, f"{sensor}_ts",
                                     config.MERGE_TOLERANCE_S)

    if merged is None:
        return None

    # Fill gaps left by slower / offset streams (e.g. barometer) so their signal
    # survives instead of becoming zeros after the nearest-timestamp join.
    return merged.ffill().bfill()


def merge_one_sample(activity, sample):
    """Merge all sensors for one raw recording and return the merged DataFrame."""
    folder = config.RAW_DIR / activity / sample
    print(f"\n  Merging {folder.relative_to(config.REPO_ROOT)}")
    merged = merge_sensor_folder(folder)
    if merged is None:
        print(f"    [warn] nothing merged for {activity}/{sample}")
    return merged


def main():
    if not config.RAW_DIR.is_dir():
        raise SystemExit(f"Raw data dir not found: {config.RAW_DIR}")

    n_written = 0
    for activity_dir in sorted(p for p in config.RAW_DIR.iterdir() if p.is_dir()):
        activity = activity_dir.name
        for sample_dir in sorted(p for p in activity_dir.iterdir() if p.is_dir()):
            sample = sample_dir.name
            merged = merge_one_sample(activity, sample)
            if merged is None:
                continue
            out_dir = config.MERGED_DIR / activity
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{sample}.csv"
            merged.to_csv(out_path, index=False)
            print(f"    saved -> {out_path.relative_to(config.REPO_ROOT)}")
            n_written += 1

    print(f"\nDone. Wrote {n_written} merged recordings to "
          f"{config.MERGED_DIR.relative_to(config.REPO_ROOT)}")


if __name__ == "__main__":
    main()
