"""
CLIENT NODE (CLIENT 3)

Same as client2, but fairness is computed using AGE GROUP.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "client_3.xlsx"

TEST_SIZE = 0.2
RANDOM_STATE = 42

CLIP_NORM = 1.0
NOISE_SCALE = 0.1


def run_client3(feature_columns, global_weights=None):

    # ================================
    # STEP 1: LOAD DATA
    # ================================
    df = pd.read_excel(DATA_PATH)

    # ================================
    # STEP 2: PREPARE FEATURES + LABEL
    # ================================
    y = df["income_group"]

    y = y.map({
        "low": 0,
        "medium": 1,
        "high": 2
    })

    X = df.drop("income_group", axis=1)
    X = pd.get_dummies(X)

    # align with global schema
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_columns]

    # ================================
    # STEP 3: TRAIN / TEST SPLIT
    # ================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    X_test_original = X_test.copy()

    # ================================
    # STEP 4: SCALE DATA
    # ================================
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    # ================================
    # STEP 5: TRAIN MODEL
    # ================================
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # ================================
    # STEP 6: LOCAL DIFFERENTIAL PRIVACY
    # ================================
    weights = model.coef_
    bias = model.intercept_

    norm = np.linalg.norm(weights)

    if norm > CLIP_NORM:
        weights = weights * (CLIP_NORM / norm)

    noise = np.random.normal(0, NOISE_SCALE, weights.shape)
    weights_noisy = weights + noise

    model.coef_ = weights_noisy
    model.intercept_ = bias

    # ================================
    # STEP 7: LOCAL TESTING
    # ================================
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Client 3 Overall Accuracy:", accuracy)

    # ================================
    # STEP 7.1: FAIRNESS METRICS (AGE GROUP)
    # ================================
    print("\n=== Fairness Metrics (Age Group) ===")

    age_cols = [col for col in X.columns if "age_group" in col.lower()]

    dp_gap = None

    if len(age_cols) == 0:
        print("⚠️ Age group columns not found")
    else:
        rates = []

        for col in age_cols:
            idx = X_test_original[col] == 1

            if idx.sum() == 0:
                continue

            acc = accuracy_score(y_test[idx], y_pred[idx])
            rate = np.mean(y_pred[idx] == 2)

            rates.append(rate)

            print(f"{col} Accuracy:", acc)
            print(f"{col} High Income Rate:", rate)

        if len(rates) > 1:
            dp_gap = max(rates) - min(rates)
            print("\nDemographic Parity Gap:", dp_gap)

    # ================================
    # STEP 8: OUTPUT FOR SERVER
    # ================================
    client_output = {
        "weights": model.coef_,
        "bias": model.intercept_,
        "num_samples": len(X_train),
    }

    # ================================
    # STEP 8: SAVE CLIENT OUTPUT
    # ================================
    output_dir = Path(__file__).resolve().parent.parent / "client_output"
    output_dir.mkdir(exist_ok=True)

    np.savez(
        output_dir / "client_3_output.npz",
        weights=client_output["weights"],
        bias=client_output["bias"],
        num_samples=client_output["num_samples"],
        accuracy=accuracy,
        dp_gap=dp_gap if dp_gap is not None else -1,
    )

    print(f"Client 3 output saved to {output_dir / 'client_3_output.npz'}")

    # ================================
    # DEBUG
    # ================================
    print("\nModel Weights Shape:", model.coef_.shape)
    print("Number of Training Samples:", len(X_train))

    return {
        "weights": model.coef_.flatten(),
        "bias": model.intercept_,
        "num_samples": len(X_train),
        "local_accuracy": accuracy,
        "dp_gap": dp_gap,
    }