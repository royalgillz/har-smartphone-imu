"""
Ingest the original phyphox export zips in data/original_recordings/ into the
data/raw/<Activity>/<Sample>/ layout, so recordings that were never extracted
become available for training.

Deduplication: each phyphox export has a unique "recording time" in Metadata.csv.
A zip whose recording time already exists under its target activity is skipped, so
re-running is safe and never creates duplicates.

Run:  python -m src.har.preprocess.ingest_zips
"""
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from .. import config

# Map a zip-filename prefix to the target activity folder in data/raw.
# Order matters: the first prefix that the filename starts with wins.
PREFIX_TO_ACTIVITY = {
    "Elevator_down": "Elevator_down",
    "Elevator_up": "Elevator_up",
    "Normal_walking": "Normal_walking",
    "Round_Cycling": "Round_Cycling",
    "Running_Hand_still": "Running_Hand_still",
    "Running_pocket": "Running_pocket",
    "Straight_cycling": "Straight_cycling",
    "Stair_Down": "Stairs_down",
    "Stairs_Down": "Stairs_down",
    "stair down": "Stairs_down",
}

ORIG_DIR = config.DATA_DIR / "original_recordings"


def recording_time(folder):
    """Read the unique recording-time key from a sensor folder's Metadata.csv."""
    meta = folder / "Metadata.csv"
    if not meta.exists():
        return None
    try:
        df = pd.read_csv(meta)
        for col in df.columns:
            if col.strip().lower() == "recording time":
                return str(df[col].iloc[0])
    except Exception:
        return None
    return None


def existing_recording_times(activity_dir):
    times = {}
    if not activity_dir.is_dir():
        return times
    for sample in activity_dir.iterdir():
        if sample.is_dir():
            t = recording_time(sample)
            if t:
                times[t] = sample.name
    return times


def find_sensor_root(root):
    root = Path(root)
    items = list(root.iterdir())
    if any(p.suffix.lower() in (".csv", ".xlsx") for p in items):
        return root
    subdirs = [p for p in items if p.is_dir()]
    if len(subdirs) == 1:
        return find_sensor_root(subdirs[0])
    return root


def match_activity(zip_name):
    for prefix, activity in PREFIX_TO_ACTIVITY.items():
        if zip_name.startswith(prefix):
            return activity
    return None


def next_sample_index(activity_dir, activity):
    n = 0
    if activity_dir.is_dir():
        n = sum(1 for p in activity_dir.iterdir() if p.is_dir())
    return n + 1


def main():
    if not ORIG_DIR.is_dir():
        raise SystemExit(f"Not found: {ORIG_DIR}")

    zips = sorted(ORIG_DIR.glob("*.zip"))
    print(f"Found {len(zips)} zips in {ORIG_DIR.relative_to(config.REPO_ROOT)}")

    added, skipped, ignored = 0, 0, []
    for zp in zips:
        activity = match_activity(zp.name)
        if activity is None:
            ignored.append(zp.name)
            continue

        activity_dir = config.RAW_DIR / activity
        existing = existing_recording_times(activity_dir)

        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(tmp)
            src = find_sensor_root(tmp)
            rtime = recording_time(src)

            if rtime and rtime in existing:
                print(f"  skip (already present as {existing[rtime]}): {zp.name}")
                skipped += 1
                continue

            idx = next_sample_index(activity_dir, activity)
            dest = activity_dir / f"{activity}_{idx}"
            dest.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, dest / f.name)
            print(f"  added -> {dest.relative_to(config.REPO_ROOT)}  (from {zp.name})")
            added += 1

    print(f"\nAdded {added}, skipped {skipped} duplicates.")
    if ignored:
        print(f"Ignored (no activity mapping): {ignored}")


if __name__ == "__main__":
    main()
