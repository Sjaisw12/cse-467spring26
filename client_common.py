"""
client_common.py

Reusable logic for all federated clients.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from fl_utils import load_client_dataset


def split_weight_vector(weight_vector: np.ndarray, num_features: int):
    coef = weight_vector[:num_features].reshape(1, -1)
    intercept = np.array([weight_vector[num_features]])
    return coef, intercept


def combine_model_weights(model: SGDClassifier) -> np.ndarray:
    coef = model.coef_.reshape(-1)
    intercept = model.intercept_.reshape(-1)
    return np.concatenate([coef, intercept])


def clip_vector(vec: np.ndarray, clip_norm: float) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.copy()
    scale = min(1.0, clip_norm / (norm + 1e-12))
    return vec * scale


def add_local_dp(update: np.ndarray, clip_norm: float, noise_scale: float) -> np.ndarray:
    clipped = clip_vector(update, clip_norm)
    noise = np.random.normal(loc=0.0, scale=noise_scale, size=update.shape)
    return clipped + noise


def run_client(
    client_id: int,
    data_path: str,
    feature_columns,
    global_weights: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    local_epochs: int = 3,
    local_clip_norm: float = 1.0,
    local_noise_scale: float = 0.05,
) -> Dict[str, Any]:
    X, y, sensitive_values, target_column, sensitive_column = load_client_dataset(
        data_path=data_path,
        feature_columns=feature_columns,
    )

    stratify_y = y if len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y
    )

    num_features = X_train.shape[1]

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        max_iter=1,
        tol=None,
        random_state=random_state,
        warm_start=True,
    )

    # Initialize model once
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))

    # Set current global weights into local model
    global_coef, global_intercept = split_weight_vector(global_weights, num_features)
    model.coef_ = global_coef.copy()
    model.intercept_ = global_intercept.copy()

    # Local training
    for _ in range(local_epochs):
        model.partial_fit(X_train, y_train)

    local_weights = combine_model_weights(model)

    # Update relative to current global model
    raw_update = local_weights - global_weights

    # Local DP on client update
    private_update = add_local_dp(
        update=raw_update,
        clip_norm=local_clip_norm,
        noise_scale=local_noise_scale,
    )

    # Local eval
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[Client {client_id}] train={len(X_train)} test={len(X_test)} acc={acc:.4f}")

    return {
        "client_id": client_id,
        "update": private_update,
        "local_accuracy": float(acc),
        "num_train_samples": int(len(X_train)),
        "num_test_samples": int(len(X_test)),
        "target_column": target_column,
        "sensitive_column": sensitive_column,
    }