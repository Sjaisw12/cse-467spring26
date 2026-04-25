"""
server.py

Federated server:
- receives client updates
- applies FedAvg
- applies Global DP
"""

from __future__ import annotations

from typing import List, Dict, Any
import numpy as np

from global_dp import clip_client_updates, add_gaussian_noise


class FederatedServer:
    def __init__(
        self,
        num_features: int,
        global_clip_norm: float = 1.0,
        global_noise_scale: float = 0.05,
        global_learning_rate: float = 1.0,
    ):
        self.global_weights = np.zeros(num_features, dtype=np.float64)
        self.global_clip_norm = global_clip_norm
        self.global_noise_scale = global_noise_scale
        self.global_learning_rate = global_learning_rate

    def aggregate(self, client_payloads: List[Dict[str, Any]]) -> np.ndarray:
        if not client_payloads:
            raise ValueError("No client payloads received by server.")

        updates = [payload["weights"] for payload in client_payloads]

        sample_counts = np.array(
            [payload["num_samples"] for payload in client_payloads],
            dtype=np.float64
        )

        total_samples = sample_counts.sum()

        if total_samples <= 0:
            raise ValueError("Total client sample count must be > 0.")

        clipped_updates = clip_client_updates(updates, self.global_clip_norm)

        weighted_update = np.zeros_like(self.global_weights)

        for upd, count in zip(clipped_updates, sample_counts):
            weighted_update += (count / total_samples) * upd

        private_update = add_gaussian_noise(
            weighted_update,
            self.global_noise_scale
        )

        self.global_weights = (
            self.global_weights
            + self.global_learning_rate * private_update
        )

        return self.global_weights.copy()

    def get_global_weights(self) -> np.ndarray:
        return self.global_weights.copy()