"""
CLIENT NODE (CLIENT 6)

Same as previous clients, fairness is computed using AGE GROUP again.
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

DATA_PATH = "client_6.xlsx"

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

print("Client 6 Overall Accuracy:", accuracy)


# ================================
# STEP 7.1: FAIRNESS (AGE GROUP)
# ================================

print("\n=== Fairness Metrics (Age Group) ===")

age_cols = [col for col in X.columns if "age_group" in col.lower()]

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
        gap = max(rates) - min(rates)
        print("\nDemographic Parity Gap:", gap)


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
    "../client_output/client_6_output.npz",
    weights=client_output["weights"],
    bias=client_output["bias"],
    num_samples=client_output["num_samples"],
    accuracy=accuracy
)

print("Client 6 output saved to ../client_output/client_6_output.npz")

print("\nModel Weights Shape:", model.coef_.shape)
print("Number of Training Samples:", len(X_train))