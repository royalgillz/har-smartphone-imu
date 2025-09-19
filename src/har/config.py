"""
Central configuration for the HAR pipeline.

Single source of truth for paths, the sensor list, the explicit feature set, and
windowing parameters. Every other module imports from here so that training and
inference can never silently disagree about feature order or window size.

Paths are resolved from this file's location, so scripts work regardless of the
current working directory.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths  (src/har/config.py -> parents[2] == repo root)
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                       # data/raw/<Activity>/<Sample>/*.csv
PROCESSED_DIR = DATA_DIR / "processed"
MERGED_DIR = PROCESSED_DIR / "merged"           # merged/<Activity>/<Sample>.csv
WINDOWED_DIR = PROCESSED_DIR / "windowed"       # X.npy, y.npy, groups.npy, labels.npy
TEST_ZIP_DIR = DATA_DIR / "test_zips"
TEST_EXTRACT_DIR = DATA_DIR / "_test_extracted"  # scratch dir for unzipped test data

MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "cnn_lstm_har.pth"
NORM_STATS_PATH = WINDOWED_DIR / "norm_stats.npz"

# --------------------------------------------------------------------------
# Sensors to read from each recording (phyphox export file names, no extension)
# --------------------------------------------------------------------------
SENSORS = [
    "Accelerometer",
    "Gyroscope",
    "Gravity",
    "TotalAcceleration",
    "Orientation",
    "Barometer",
]

# Sensors whose readings change slowly and/or are sampled sparsely. After the
# nearest-timestamp merge they are forward/back-filled instead of zero-filled,
# so e.g. the barometer (event-based, far below 100 Hz) is not wiped out.
SLOW_FILL_SENSORS = ["Barometer", "Orientation", "Gravity"]

# --------------------------------------------------------------------------
# Explicit feature set (merged column names, in fixed order).
#
# Why explicit: the old pipeline derived features by suffix-matching *_x/_y/_z/_ts,
# which (a) fed 8 raw timestamps to the model and (b) silently dropped the
# Orientation quaternion (Orientation_qx etc. ends in "qx", not "_x") and the
# whole Barometer. This list is the real motion signal only -- no timestamps,
# no redundant uncalibrated duplicates.
# --------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Accelerometer_x", "Accelerometer_y", "Accelerometer_z",
    "Gyroscope_x", "Gyroscope_y", "Gyroscope_z",
    "Gravity_x", "Gravity_y", "Gravity_z",
    "TotalAcceleration_x", "TotalAcceleration_y", "TotalAcceleration_z",
    "Orientation_qw", "Orientation_qx", "Orientation_qy", "Orientation_qz",
    "Barometer_relativeAltitude", "Barometer_pressure",
]
NUM_FEATURES = len(FEATURE_COLUMNS)  # 18

# --------------------------------------------------------------------------
# Merging / windowing
# --------------------------------------------------------------------------
# Tolerance (seconds) for nearest-timestamp join of the 100 Hz streams.
MERGE_TOLERANCE_S = 0.05

# ~100 Hz sampling -> 192 samples ~= 1.92 s window, 64-sample (~0.64 s) stride.
SAMPLE_RATE_HZ = 100
WINDOW_SIZE = 192
STEP = 64

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2          # fraction of *recordings* (not windows) held out
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# --------------------------------------------------------------------------
# Data augmentation (training batches only; see augment.py)
# --------------------------------------------------------------------------
USE_AUGMENTATION = True
AUG = {
    "rotate_p": 0.5, "rotate_std": 0.20,    # ~11 deg std random 3D rotation
    "scale_p": 0.5, "scale_std": 0.10,      # per-channel amplitude scaling
    "jitter_p": 0.5, "jitter_std": 0.05,    # noise as fraction of channel std
    "magwarp_p": 0.3, "magwarp_std": 0.10, "magwarp_knots": 4,
}
