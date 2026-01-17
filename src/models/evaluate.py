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

# Load models & preprocessors
# Logistic Regression (pipeline, raw X)
lr_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)

# Decision Tree
tree = joblib.load(
    os.path.join(MODEL_DIR, "decision_tree.pkl")
)

# Hybrid Rule-Augmented LR
hybrid_lr = joblib.load(
    os.path.join(MODEL_DIR, "hybrid_rule_aug_lr.pkl")
)

# Shared preprocessing (tree + hybrid)
preprocess = joblib.load(
    os.path.join(MODEL_DIR, "preprocess.pkl")
)

rule_encoder = joblib.load(
    os.path.join(MODEL_DIR, "rule_encoder.pkl")
)

# XGBoost
xgb_bundle = joblib.load(
    os.path.join(MODEL_DIR, "xgboost.pkl")
)
xgb = xgb_bundle["model"]
xgb_imputer = xgb_bundle["imputer"]
xgb_scaler = xgb_bundle["scaler"]

# XGBoost → LR Stack
stack_bundle = joblib.load(
    os.path.join(MODEL_DIR, "xgb_lr_stack.pkl")
)
stack_xgb = stack_bundle["xgb"]
stack_meta = stack_bundle["meta_lr"]
stack_imputer = stack_bundle["imputer"]
stack_scaler = stack_bundle["scaler"]

# Helper: evaluation printer
def evaluate_model(name, y_true, y_pred):
    print("\n" + "=" * 30)
    print(f" {name}")
    print("=" * 30)
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Macro F1 :", f1_score(y_true, y_pred, average="macro"))
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

# Logistic Regression
y_pred_lr = lr_pipeline.predict(X)
evaluate_model("Logistic Regression", y, y_pred_lr)

# Decision Tree
X_pre = preprocess.transform(X)
y_pred_dt = tree.predict(X_pre)
evaluate_model("Decision Tree", y, y_pred_dt)

# Hybrid Rule-Augmented Logistic Regression
leaves = tree.apply(X_pre)
rule_features = rule_encoder.transform(leaves.reshape(-1, 1))
X_hybrid = np.hstack([X_pre, rule_features])

y_pred_hybrid = hybrid_lr.predict(X_hybrid)
evaluate_model("Hybrid Rule-Augmented LR", y, y_pred_hybrid)

# XGBoost (Best Performance)
X_xgb = xgb_scaler.transform(xgb_imputer.transform(X))
y_pred_xgb = xgb.predict(X_xgb)
evaluate_model("XGBoost (Best)", y, y_pred_xgb)

# XGBoost → Logistic Regression Stack
X_stack = stack_scaler.transform(stack_imputer.transform(X))
stack_probs = stack_xgb.predict_proba(X_stack)
y_pred_stack = stack_meta.predict(stack_probs)

evaluate_model("XGBoost + LR Stack", y, y_pred_stack)

print("\nEvaluation complete for all models.")
