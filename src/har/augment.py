"""
On-the-fly data augmentation for IMU windows (applied to TRAIN batches only).

Augmentations are applied to RAW (un-normalized) windows so they stay physically
meaningful, then the caller normalizes. Transforms:
  * rotation  - small random 3D rotation of each (x,y,z) sensor triplet, the most
                effective IMU augmentation (simulates holding the phone differently)
  * scaling   - per-channel amplitude scaling (stronger/weaker motion)
  * jitter    - additive Gaussian noise scaled per channel (sensor noise)
  * magwarp   - smooth multiplicative warp along time (slow drift / cadence change)

The orientation quaternion and barometer channels are left out of rotation/scaling
(rotating them arbitrarily is not meaningful); they still receive jitter.
"""
import numpy as np

from . import config


def _triplet_index_groups():
    """Column-index triplets for the sensors whose x/y/z can be rotated together."""
    cols = config.FEATURE_COLUMNS
    groups = []
    for prefix in ("Accelerometer", "Gyroscope", "Gravity", "TotalAcceleration"):
        names = [f"{prefix}_{a}" for a in ("x", "y", "z")]
        if all(n in cols for n in names):
            groups.append([cols.index(n) for n in names])
    return groups


TRIPLETS = _triplet_index_groups()


def _random_rotation(rng, angle_std):
    """Small random rotation matrix via axis-angle (Rodrigues)."""
    angle = rng.normal(0.0, angle_std)
    axis = rng.normal(size=3)
    axis /= (np.linalg.norm(axis) + 1e-8)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R.astype(np.float32)


def _magnitude_warp(rng, T, n_knots, std):
    """Smooth length-T multiplicative curve centered on 1.0."""
    knots = rng.normal(1.0, std, size=n_knots)
    xp = np.linspace(0, T - 1, n_knots)
    return np.interp(np.arange(T), xp, knots).astype(np.float32)


def augment_batch(X, channel_std, rng, aug=None):
    """
    X: (B, T, F) raw float32. Returns an augmented copy (input is not modified).
    channel_std: (F,) per-channel std of the training set, used to scale jitter.
    """
    aug = aug or config.AUG
    X = X.copy()
    B, T, F = X.shape

    for b in range(B):
        if TRIPLETS and rng.random() < aug["rotate_p"]:
            for idx in TRIPLETS:
                R = _random_rotation(rng, aug["rotate_std"])
                X[b][:, idx] = X[b][:, idx] @ R.T

        if rng.random() < aug["scale_p"]:
            s = rng.normal(1.0, aug["scale_std"], size=F).astype(np.float32)
            X[b] *= s

        if rng.random() < aug["magwarp_p"]:
            curve = _magnitude_warp(rng, T, aug["magwarp_knots"], aug["magwarp_std"])
            X[b] *= curve[:, None]

        if rng.random() < aug["jitter_p"]:
            noise = rng.normal(0.0, aug["jitter_std"], size=(T, F)).astype(np.float32)
            X[b] += noise * channel_std

    return X
