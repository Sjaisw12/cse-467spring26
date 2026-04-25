"""
CLIENT NODE (CLIENT 5)

Same as previous clients, fairness is computed using RACE again.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "client_5.xlsx"

TEST_SIZE = 0.2
RANDOM_STATE = 42

CLIP_NORM = 1.0
NOISE_SCALE = 0.1


def run_client5(feature_columns, global_weights=None):

    df = pd.read_excel(DATA_PATH)

    y = df["income_group"]
    y = y.map({
        "low": 0,
        "medium": 1,
        "high": 2
    })

    X = df.drop("income_group", axis=1)
    X = pd.get_dummies(X)

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    X_test_original = X_test.copy()

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    weights = model.coef_
    bias = model.intercept_

    norm = np.linalg.norm(weights)

    if norm > CLIP_NORM:
        weights = weights * (CLIP_NORM / norm)

    noise = np.random.normal(0, NOISE_SCALE, weights.shape)
    weights_noisy = weights + noise

    model.coef_ = weights_noisy
    model.intercept_ = bias

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Client 5 Overall Accuracy:", accuracy)

    print("\n=== Fairness Metrics (Race) ===")

    white_col = None
    black_col = None
    dp_gap = None

    for col in X.columns:
        if "race" in col.lower() and "white" in col.lower():
            white_col = col
        if "race" in col.lower() and "black" in col.lower():
            black_col = col

    if white_col is None or black_col is None:
        print("⚠️ Race columns not found")
    else:
        white_idx = X_test_original[white_col] == 1
        black_idx = X_test_original[black_col] == 1

        white_acc = accuracy_score(y_test[white_idx], y_pred[white_idx]) if white_idx.sum() > 0 else None
        black_acc = accuracy_score(y_test[black_idx], y_pred[black_idx]) if black_idx.sum() > 0 else None

        white_rate = np.mean(y_pred[white_idx] == 2) if white_idx.sum() > 0 else None
        black_rate = np.mean(y_pred[black_idx] == 2) if black_idx.sum() > 0 else None

        print("White Accuracy:", white_acc)
        print("Black Accuracy:", black_acc)

        print("\nDemographic Parity (High Income Rate):")
        print("White Positive Rate:", white_rate)
        print("Black Positive Rate:", black_rate)

        if white_rate is not None and black_rate is not None:
            dp_gap = abs(white_rate - black_rate)
            print("Demographic Parity Gap:", dp_gap)

    client_output = {
        "weights": model.coef_,
        "bias": model.intercept_,
        "num_samples": len(X_train),
    }

    output_dir = Path(__file__).resolve().parent.parent / "client_output"
    output_dir.mkdir(exist_ok=True)

    np.savez(
        output_dir / "client_5_output.npz",
        weights=client_output["weights"],
        bias=client_output["bias"],
        num_samples=client_output["num_samples"],
        accuracy=accuracy,
        dp_gap=dp_gap if dp_gap is not None else -1,
    )

    print(f"Client 5 output saved to {output_dir / 'client_5_output.npz'}")

    print("\nModel Weights Shape:", model.coef_.shape)
    print("Number of Training Samples:", len(X_train))

    return {
        "weights": model.coef_.flatten(),
        "bias": model.intercept_,
        "num_samples": len(X_train),
        "local_accuracy": accuracy,
        "dp_gap": dp_gap,
    }