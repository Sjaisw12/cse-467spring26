"""
CLIENT NODE (CLIENT 8)

Same as previous clients, fairness is computed using RACE again.
"""

# ================================
# IMPORTS
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# ================================
# CONFIGURATION
# ================================

DATA_PATH = "client_8.xlsx"

TEST_SIZE = 0.2
RANDOM_STATE = 42

CLIP_NORM = 1.0
NOISE_SCALE = 0.1


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


# ================================
# STEP 3: TRAIN / TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# save original for fairness
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

norm = np.linalg.norm(weights)

if norm > CLIP_NORM:
    weights = weights * (CLIP_NORM / norm)

noise = np.random.normal(0, NOISE_SCALE, weights.shape)
weights_noisy = weights + noise

model.coef_ = weights_noisy


# ================================
# STEP 7: LOCAL TESTING
# ================================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Client 8 Overall Accuracy:", accuracy)


# ================================
# STEP 7.1: FAIRNESS (RACE)
# ================================

print("\n=== Fairness Metrics (Race) ===")

white_col = None
black_col = None

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
        gap = abs(white_rate - black_rate)
        print("Demographic Parity Gap:", gap)


# ================================
# STEP 8: OUTPUT
# ================================

client_output = {
    "weights": model.coef_,
    "bias": model.intercept_,
    "num_samples": len(X_train)
}
# ================================
# STEP 8: SAVE CLIENT OUTPUT FOR SERVER
# ================================

import os

os.makedirs("../client_output", exist_ok=True)

np.savez(
    "../client_output/client_8_output.npz",
    weights=client_output["weights"],
    bias=client_output["bias"],
    num_samples=client_output["num_samples"],
    accuracy=accuracy
)

print("Client 8 output saved to ../client_output/client_8_output.npz")

print("\nModel Weights Shape:", model.coef_.shape)
print("Number of Training Samples:", len(X_train))