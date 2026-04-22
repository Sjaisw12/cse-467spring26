# Federated Learning with Differential Privacy: Privacy–Accuracy–Fairness Trade-offs

## 📌 Project Overview

This project explores how Differential Privacy (DP) impacts model accuracy and fairness in a Federated Learning (FL) setting.

In Federated Learning, multiple clients train models locally on their own data and only share model parameters (not raw data) with a central server. While this improves privacy, additional DP mechanisms introduce noise, which can affect both performance and fairness.

The goal of this project is to analyze the trade-offs between:

- Privacy
- Accuracy
- Fairness

---

## 🧠 Key Concepts

- Federated Learning (FL): Decentralized training across multiple clients
- Differential Privacy (DP): Adds noise to protect sensitive data
- Fairness Metrics: Evaluates bias across demographic groups

---

## ⚙️ Project Structure

cse-467spring26-main/ │ ├── client1/ ├── client2/ ├── client3/ ├── ... ├── client10/ │ └── clientX_node.py │ ├── client_output/ │ └── client_X.xlsx │ ├── server.py ├── README.md

---

## 📊 Dataset

- Synthetic dataset with attributes:
  - Gender
  - Race
  - Age Group
  - Income Group (target)

- Each client receives a subset of the dataset to simulate decentralized data (non-IID setting).

---

## 🏗️ Methodology

### 1. Local Training (Clients)

Each client:

- Loads its own dataset
- Encodes categorical features
- Splits data into train/test
- Trains a Logistic Regression model
- Applies Local Differential Privacy
  - Gradient clipping
  - Gaussian noise addition

---

### 2. Fairness Evaluation

Fairness is evaluated using Demographic Parity:

- Positive Rate = % of predictions classified as “High Income”
- Compared across groups:
  - Gender
  - Race
  - Age Group

---

### 3. Federated Aggregation (Server)

The server:

- Collects model weights from all clients
- Applies Federated Averaging (FedAvg)

Global Model = Weighted Average of Client Models

---

## 📈 Experiments

We evaluate three settings:

### 🔹 1. No Differential Privacy

- Baseline model
- Highest accuracy expected

### 🔹 2. Local Differential Privacy

- Noise added at client level
- Trade-off between privacy and performance

### 🔹 3. (Optional) Global Differential Privacy

- Noise added during server aggregation

---

## 📉 Metrics

- Accuracy
- Demographic Parity Gap
- Group-wise Accuracy

---

## 🔍 Observations

- Accuracy typically ranges between 0.70 – 0.75
- Adding DP noise reduces accuracy
- Fairness varies across clients and demographic groups
- In some cases:
  - Models show accuracy disparity
  - Or unequal prediction rates

---

## ⚠️ Key Insight

A model can appear fair (low disparity) but still be biased if it:

- Rarely predicts certain classes
- Performs poorly on specific groups

---

## ▶️ How to Run

### Run a client:

bash python clientX_node.py

### Run server (after collecting outputs):

bash python server.py

---

## 🧪 Requirements

Install dependencies:
bash pip install pandas numpy scikit-learn openpyxl

---

## 👨‍💻 Authors

- Kamaal Alag
- Devendra Janyani
- Manya Shukla
- Ronit Khanna
- Sanyam Jaiswal

---

## 📌 Conclusion

This project demonstrates that:

- Privacy, accuracy, and fairness are interdependent
- Increasing privacy (via DP) often reduces accuracy
- Fairness outcomes are complex and dataset-dependent

---

## 🚀 Future Work

- Use advanced DP frameworks (e.g., Opacus)
- Evaluate additional fairness metrics:
  - Equal Opportunity
  - Equalized Odds
- Experiment with deeper models (neural networks)
