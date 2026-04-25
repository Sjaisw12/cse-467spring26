"""
CLIENT NODE (CLIENT 1)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ================================
# CONFIGURATION
# ================================

DATA_PATH = Path(__file__).resolve().parent / "client_1.xlsx"

TEST_SIZE = 0.2
RANDOM_STATE = 42

CLIP_NORM = 1.0
NOISE_SCALE = 0.1

def run_client1(feature_columns, global_weights=None):
    from pathlib import Path

    # ================================
    # STEP 1: LOAD DATA
    # ================================
    DATA_PATH = Path(__file__).resolve().parent / "client_1.xlsx"
    df = pd.read_excel(DATA_PATH)

    # ================================
    # STEP 2: PREPARE FEATURES + LABEL
    # ================================
    y = df["income_group"]
    X = df.drop("income_group", axis=1)

    X = pd.get_dummies(X)

    # align schema with global features
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_columns]

    # ================================
    # STEP 3: TRAIN / TEST SPLIT
    # ================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ================================
    # STEP 4: TRAIN MODEL
    # ================================
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # ================================
    # STEP 5: APPLY LOCAL DP
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
    # STEP 6: LOCAL TESTING
    # ================================
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Client 1 Overall Accuracy:", accuracy)

    # ================================
    # STEP 6.1: FAIRNESS METRICS
    # ================================
    gender_col = None
    for col in X.columns:
        if "gender" in col.lower():
            gender_col = col
            break

    dp_gap = None

    if gender_col:
        male_idx = X_test[gender_col] == 1
        female_idx = X_test[gender_col] == 0

        male_acc = accuracy_score(y_test[male_idx], y_pred[male_idx]) if male_idx.sum() > 0 else None
        female_acc = accuracy_score(y_test[female_idx], y_pred[female_idx]) if female_idx.sum() > 0 else None

        male_pos = np.mean(y_pred[male_idx]) if male_idx.sum() > 0 else None
        female_pos = np.mean(y_pred[female_idx]) if female_idx.sum() > 0 else None

        print("\n=== Fairness Metrics (Gender) ===")
        print("Male Accuracy:", male_acc)
        print("Female Accuracy:", female_acc)

        print("\nDemographic Parity:")
        print("Male Positive Rate:", male_pos)
        print("Female Positive Rate:", female_pos)

        if male_pos is not None and female_pos is not None:
            dp_gap = abs(male_pos - female_pos)
            print("Demographic Parity Gap:", dp_gap)
    else:
        print("⚠️ Gender column not found")

    # ================================
    # STEP 8: SAVE CLIENT OUTPUT (KEPT ✅)
    # ================================
    output_dir = Path(__file__).resolve().parent.parent / "client_output"
    output_dir.mkdir(exist_ok=True)

    np.savez(
        output_dir / "client_1_output.npz",
        weights=model.coef_.flatten(),
        bias=model.intercept_,
        num_samples=len(X_train),
        accuracy=accuracy,
        dp_gap=dp_gap if dp_gap is not None else -1
    )

    print(f"Client 1 output saved to {output_dir / 'client_1_output.npz'}")

    print("\nModel Weights Shape:", model.coef_.shape)
    print("Number of Training Samples:", len(X_train))

    # ================================
    # RETURN TO SERVER
    # ================================
    return {
        "weights": model.coef_.flatten(),
        "bias": model.intercept_,
        "num_samples": len(X_train),
        "local_accuracy": accuracy,
        "dp_gap": dp_gap,
    }
    
    