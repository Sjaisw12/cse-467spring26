"""
global_dp.py

Global DP utilities for server-side aggregation.
"""

from __future__ import annotations

import numpy as np


def l2_clip(vector: np.ndarray, clip_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.copy()

    scale = min(1.0, clip_norm / (norm + 1e-12))
    return vector * scale


def clip_client_updates(client_updates, clip_norm: float):
    return [l2_clip(update, clip_norm) for update in client_updates]


def add_gaussian_noise(vector: np.ndarray, noise_scale: float) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=vector.shape)
    return vector + noise