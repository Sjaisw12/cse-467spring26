"""
CLIENT NODE (CLIENT 1)

This file simulates ONE federated learning client.

Responsibilities:
1. Load its own local dataset
2. Preprocess data (features + labels)
3. Split into train/test (local evaluation)
4. Train logistic regression model
5. Apply Local Differential Privacy (DP)
6. Evaluate locally
7. Prepare weights to send to server

IMPORTANT:
- Each client runs independently
- No raw data leaves this client (privacy goal)
"""

# ================================
# IMPORTS
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ================================
# CONFIGURATION (YOU CAN TUNE THESE)
# ================================

DATA_PATH = "client_1.xlsx"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Differential Privacy parameters
CLIP_NORM = 1.0          # controls gradient magnitude
NOISE_SCALE = 0.1        # controls privacy strength (higher = more privacy, less accuracy)


# ================================
# STEP 1: LOAD DATA
# ================================

df = pd.read_excel(DATA_PATH)

# NOTE:
# Data should already be cleaned from your previous script.
# If not, you would need to:
# - remove missing values
# - encode categorical features


# ================================
# STEP 2: PREPARE FEATURES + LABEL
# ================================

"""
IMPORTANT CONCEPT:

X = features (input)
y = label (target)

We MUST NOT include the target in features (data leakage).
"""

y = df["income_group"]  # target (0 = low, 1 = high)
X = df.drop("income_group", axis=1)

# NOTE:
# At this stage:
# X should already be numeric (after one-hot encoding earlier)
# If not, you MUST encode before training


# ================================
# STEP 3: TRAIN / TEST SPLIT
# ================================

"""
WHY THIS IS ESSENTIAL:

- We need unseen data to evaluate performance
- Otherwise model will just memorize (overfitting)

Each client evaluates locally on its own data.
"""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# ================================
# STEP 4: TRAIN LOGISTIC REGRESSION
# ================================

"""
WHY LOGISTIC REGRESSION?

- Simple baseline
- Interpretable (important for fairness analysis)
- Works well for binary classification

Model learns:
P(y=1 | X) = probability of high income
"""

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ================================
# STEP 5: APPLY LOCAL DIFFERENTIAL PRIVACY (CORE PART)
# ================================

"""
WHAT IS HAPPENING HERE?

In real DP-SGD:
- gradients are clipped per sample
- noise is added during training

Here we SIMULATE it by:
1. Clipping model weights
2. Adding Gaussian noise

WHY THIS IS IMPORTANT:
- Prevents leakage of individual user data
- Makes model privacy-preserving

TRADE-OFF:
- More noise = more privacy BUT less accuracy
"""

# Extract learned weights
weights = model.coef_
bias = model.intercept_

# ---- STEP 5A: CLIP WEIGHTS ----

norm = np.linalg.norm(weights)

if norm > CLIP_NORM:
    weights = weights * (CLIP_NORM / norm)

# ---- STEP 5B: ADD NOISE ----

noise = np.random.normal(0, NOISE_SCALE, weights.shape)
weights_noisy = weights + noise

# Update model with noisy weights
model.coef_ = weights_noisy


# ================================
# STEP 6: LOCAL TESTING
# ================================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Client 1 Overall Accuracy:", accuracy)


# ================================
# STEP 6.1: FAIRNESS METRICS (ADDED)
# ================================

"""
We measure fairness across gender.

Assumption:
- After encoding, gender is represented as:
    gender_Male (1 = male, 0 = female)

If your column name is different, print X.columns and adjust.
"""

# ---- FIND GENDER COLUMN ----
gender_col = None

for col in X.columns:
    if "gender" in col.lower():
        gender_col = col
        break

if gender_col is None:
    print("⚠️ Gender column not found — cannot compute fairness")
else:
    # ---- SPLIT TEST DATA BY GENDER ----
    male_idx = X_test[gender_col] == 1
    female_idx = X_test[gender_col] == 0

    # ---- MALE METRICS ----
    if male_idx.sum() > 0:
        male_acc = accuracy_score(y_test[male_idx], y_pred[male_idx])
        male_positive_rate = np.mean(y_pred[male_idx])
    else:
        male_acc = None
        male_positive_rate = None

    # ---- FEMALE METRICS ----
    if female_idx.sum() > 0:
        female_acc = accuracy_score(y_test[female_idx], y_pred[female_idx])
        female_positive_rate = np.mean(y_pred[female_idx])
    else:
        female_acc = None
        female_positive_rate = None

    # ---- PRINT RESULTS ----
    print("\n=== Fairness Metrics (Gender) ===")
    print("Male Accuracy:", male_acc)
    print("Female Accuracy:", female_acc)

    print("\nDemographic Parity (Positive Prediction Rate):")
    print("Male Positive Rate:", male_positive_rate)
    print("Female Positive Rate:", female_positive_rate)

    # ---- FAIRNESS GAP ----
    if male_positive_rate is not None and female_positive_rate is not None:
        dp_gap = abs(male_positive_rate - female_positive_rate)
        print("Demographic Parity Gap:", dp_gap)


# ================================
# STEP 7: PREPARE OUTPUT FOR SERVER
# ================================

"""
WHAT GETS SENT TO SERVER?

ONLY:
- model weights
- bias

NEVER:
- raw data

WHY:
- preserves privacy (core idea of FL)
"""

client_output = {
    "weights": model.coef_,
    "bias": model.intercept_,
    "num_samples": len(X_train)  # used for weighted averaging (FedAvg)
}


# ================================
# WHAT IS MISSING (IMPORTANT FOR PROJECT)
# ================================

"""
1. FAIRNESS METRICS (VERY IMPORTANT FOR YOUR PROJECT)

You need to compute:
- Demographic Parity
- Equal Opportunity

Example:
compare predictions across gender/race groups

-----------------------------------------

2. MULTIPLE CLIENTS

Right now:
- only client1 exists

Later:
- 10 clients
- server aggregates weights

-----------------------------------------

3. FEDERATED TRAINING LOOP

You still need:
- server.py
- aggregation logic (FedAvg)

-----------------------------------------

4. REAL DP IMPLEMENTATION

Current approach = SIMPLIFIED

Better approach (optional upgrade):
- use Opacus (PyTorch DP library)

-----------------------------------------

5. LOGGING + EXPERIMENT TRACKING

For your report:
- store accuracy
- store epsilon (privacy budget)
- store fairness metrics

-----------------------------------------
"""


# ================================
# DEBUG / SANITY CHECK
# ================================

print("\nModel Weights Shape:", model.coef_.shape)
print("Number of Training Samples:", len(X_train))