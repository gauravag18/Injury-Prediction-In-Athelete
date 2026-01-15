import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "csv", "dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")

df = pd.read_csv(CSV_PATH)

X = df.drop(columns=["label", "risk_label", "clip"], errors="ignore")
y = df["risk_label"] if "risk_label" in df.columns else df["label"]

print("Dataset shape:", X.shape)
print("Class distribution:\n", y.value_counts())

# Logistic Regression pipeline (expects RAW X)
lr_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)

# Decision Tree (expects PREPROCESSED X)
tree = joblib.load(
    os.path.join(MODEL_DIR, "decision_tree.pkl")
)

# Hybrid Rule-Augmented Logistic Regression
hybrid_lr = joblib.load(
    os.path.join(MODEL_DIR, "hybrid_rule_aug_lr.pkl")
)

# Shared preprocessing
preprocess = joblib.load(
    os.path.join(MODEL_DIR, "preprocess.pkl")
)

# Rule encoder (MUST exist)
rule_encoder_path = os.path.join(MODEL_DIR, "rule_encoder.pkl")
if not os.path.exists(rule_encoder_path):
    raise RuntimeError(
        "rule_encoder.pkl not found. Re-run train.py and save the encoder."
    )

rule_encoder = joblib.load(rule_encoder_path)

# Logistic Regression (PIPELINE)
print(" Logistic Regression ")

y_pred_lr = lr_pipeline.predict(X)

print("Accuracy :", accuracy_score(y, y_pred_lr))
print("Macro F1 :", f1_score(y, y_pred_lr, average="macro"))
print("\nClassification Report:\n")
print(classification_report(y, y_pred_lr))
print("Confusion Matrix:\n")
print(confusion_matrix(y, y_pred_lr))

# Decision Tree
print(" Decision Tree ")

X_pre = preprocess.transform(X)

y_pred_dt = tree.predict(X_pre)

print("Accuracy :", accuracy_score(y, y_pred_dt))
print("Macro F1 :", f1_score(y, y_pred_dt, average="macro"))
print("\nClassification Report:\n")
print(classification_report(y, y_pred_dt))
print("Confusion Matrix:\n")
print(confusion_matrix(y, y_pred_dt))

# Hybrid Rule-Augmented Logistic 
print(" Hybrid Rule-Augmented LR ")

# Generate rule features using SAME tree & encoder
leaves = tree.apply(X_pre)
rule_features = rule_encoder.transform(leaves.reshape(-1, 1))

# Hybrid feature vector
X_hybrid = np.hstack([X_pre, rule_features])

y_pred_hybrid = hybrid_lr.predict(X_hybrid)

print("Accuracy :", accuracy_score(y, y_pred_hybrid))
print("Macro F1 :", f1_score(y, y_pred_hybrid, average="macro"))
print("\nClassification Report:\n")
print(classification_report(y, y_pred_hybrid))
print("Confusion Matrix:\n")
print(confusion_matrix(y, y_pred_hybrid))
