"""
CLIENT NODE (CLIENT 4)

Same as client3, but fairness is computed using GENDER again.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "client_4.xlsx"

TEST_SIZE = 0.2
RANDOM_STATE = 42

CLIP_NORM = 1.0
NOISE_SCALE = 0.0

def run_client4(feature_columns, global_weights=None):

    # STEP 1: LOAD DATA
    df = pd.read_excel(DATA_PATH)

    # STEP 2: PREPARE FEATURES + LABEL
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

    # STEP 3: TRAIN / TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    X_test_original = X_test.copy()

    # STEP 4: SCALE DATA
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

    # STEP 5: TRAIN MODEL
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # STEP 6: LOCAL DIFFERENTIAL PRIVACY
    weights = model.coef_
    bias = model.intercept_

    norm = np.linalg.norm(weights)

    if norm > CLIP_NORM:
        weights = weights * (CLIP_NORM / norm)

    noise = np.random.normal(0, NOISE_SCALE, weights.shape)
    weights_noisy = weights + noise

    model.coef_ = weights_noisy
    model.intercept_ = bias

    # STEP 7: LOCAL TESTING
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Client 4 Overall Accuracy:", accuracy)

    # STEP 7.1: FAIRNESS METRICS (GENDER)
    print("\n=== Fairness Metrics (Gender) ===")

    gender_col = None

    for col in X.columns:
        if "gender" in col.lower():
            gender_col = col
            break

    dp_gap = None

    if gender_col is None:
        print("⚠️ Gender column not found")
    else:
        male_idx = X_test_original[gender_col] == 1
        female_idx = X_test_original[gender_col] == 0

        male_acc = accuracy_score(y_test[male_idx], y_pred[male_idx]) if male_idx.sum() > 0 else None
        female_acc = accuracy_score(y_test[female_idx], y_pred[female_idx]) if female_idx.sum() > 0 else None

        male_rate = np.mean(y_pred[male_idx] == 2) if male_idx.sum() > 0 else None
        female_rate = np.mean(y_pred[female_idx] == 2) if female_idx.sum() > 0 else None

        print("Male Accuracy:", male_acc)
        print("Female Accuracy:", female_acc)

        print("\nDemographic Parity (High Income Rate):")
        print("Male Positive Rate:", male_rate)
        print("Female Positive Rate:", female_rate)

        if male_rate is not None and female_rate is not None:
            dp_gap = abs(male_rate - female_rate)
            print("Demographic Parity Gap:", dp_gap)

    # STEP 8: OUTPUT FOR SERVER
    client_output = {
        "weights": model.coef_,
        "bias": model.intercept_,
        "num_samples": len(X_train),
    }

    # STEP 8: SAVE CLIENT OUTPUT
    output_dir = Path(__file__).resolve().parent.parent / "client_output"
    output_dir.mkdir(exist_ok=True)

    np.savez(
        output_dir / "client_4_output.npz",
        weights=client_output["weights"],
        bias=client_output["bias"],
        num_samples=client_output["num_samples"],
        accuracy=accuracy,
        dp_gap=dp_gap if dp_gap is not None else -1,
    )

    print(f"Client 4 output saved to {output_dir / 'client_4_output.npz'}")

    print("\nModel Weights Shape:", model.coef_.shape)
    print("Number of Training Samples:", len(X_train))

    return {
        "weights": model.coef_.flatten(),
        "bias": model.intercept_,
        "num_samples": len(X_train),
        "local_accuracy": accuracy,
        "dp_gap": dp_gap,
    }