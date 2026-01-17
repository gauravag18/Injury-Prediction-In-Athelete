import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    recall_score
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "csv", "dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")
FIG_DIR = os.path.join(PROJECT_ROOT, "docs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

X = df.drop(columns=["label", "risk_label", "clip"], errors="ignore")
y = df["risk_label"] if "risk_label" in df.columns else df["label"]

# Load models & preprocessors
lr_pipeline = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
tree = joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl"))
hybrid_lr = joblib.load(os.path.join(MODEL_DIR, "hybrid_rule_aug_lr.pkl"))
preprocess = joblib.load(os.path.join(MODEL_DIR, "preprocess.pkl"))
rule_encoder = joblib.load(os.path.join(MODEL_DIR, "rule_encoder.pkl"))

xgb_bundle = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
xgb = xgb_bundle["model"]
xgb_imputer = xgb_bundle["imputer"]
xgb_scaler = xgb_bundle["scaler"]

ebm_bundle = joblib.load(os.path.join(MODEL_DIR, "ebm.pkl"))
ebm = ebm_bundle["model"]
ebm_imputer = ebm_bundle["imputer"]
ebm_scaler = ebm_bundle["scaler"]

# Logistic Regression
y_pred_lr = lr_pipeline.predict(X)

# Decision Tree
X_pre = preprocess.transform(X)
y_pred_dt = tree.predict(X_pre)

# Hybrid Rule-Augmented LR
leaves = tree.apply(X_pre)
rule_features = rule_encoder.transform(leaves.reshape(-1, 1))
X_hybrid = np.hstack([X_pre, rule_features])
y_pred_hybrid = hybrid_lr.predict(X_hybrid)

# XGBoost
X_xgb = xgb_scaler.transform(xgb_imputer.transform(X))
y_pred_xgb = xgb.predict(X_xgb)

# EBM (BEST MODEL)
X_ebm = ebm_scaler.transform(ebm_imputer.transform(X))
y_pred_ebm = ebm.predict(X_ebm)

# Helper: Confusion matrix plot
def save_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename))
    plt.close()

# Confusion matrices
save_confusion(y, y_pred_lr, "Logistic Regression", "confusion_lr.png")
save_confusion(y, y_pred_dt, "Decision Tree", "confusion_dt.png")
save_confusion(y, y_pred_hybrid, "Hybrid Rule-Augmented LR", "confusion_hybrid.png")
save_confusion(y, y_pred_xgb, "XGBoost", "confusion_xgb.png")
save_confusion(y, y_pred_ebm, "EBM (Best Model)", "confusion_ebm.png")

# Accuracy & Macro-F1 comparison
models = [
    "Logistic Regression",
    "Decision Tree",
    "Hybrid Model",
    "XGBoost",
    "EBM (Best)"
]

accuracy = [
    accuracy_score(y, y_pred_lr),
    accuracy_score(y, y_pred_dt),
    accuracy_score(y, y_pred_hybrid),
    accuracy_score(y, y_pred_xgb),
    accuracy_score(y, y_pred_ebm)
]

macro_f1 = [
    f1_score(y, y_pred_lr, average="macro"),
    f1_score(y, y_pred_dt, average="macro"),
    f1_score(y, y_pred_hybrid, average="macro"),
    f1_score(y, y_pred_xgb, average="macro"),
    f1_score(y, y_pred_ebm, average="macro")
]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x - width/2, accuracy, width, label="Accuracy")
plt.bar(x + width/2, macro_f1, width, label="Macro F1")
plt.xticks(x, models, rotation=20)
plt.ylabel("Score")
plt.title("Model Performance Comparison (EBM Best)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_comparison.png"))
plt.close()

# Per-class recall=
labels = [0, 1, 2]

recall_data = [
    recall_score(y, y_pred_lr, average=None, labels=labels),
    recall_score(y, y_pred_dt, average=None, labels=labels),
    recall_score(y, y_pred_hybrid, average=None, labels=labels),
    recall_score(y, y_pred_xgb, average=None, labels=labels),
    recall_score(y, y_pred_ebm, average=None, labels=labels)
]

x = np.arange(len(labels))
width = 0.15

plt.figure(figsize=(10, 5))
for i, recalls in enumerate(recall_data):
    plt.bar(x + (i - 2) * width, recalls, width, label=models[i])

plt.xticks(x, ["Low Risk (0)", "Medium Risk (1)", "High Risk (2)"])
plt.ylabel("Recall")
plt.title("Per-Class Recall Comparison")
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "recall_per_class.png"))
plt.close()

# Risk score distribution (BEST MODEL = EBM)
ebm_probs = ebm.predict_proba(X_ebm)
risk_score = ebm_probs[:, 1] + 2 * ebm_probs[:, 2]

plt.figure(figsize=(8, 5))
plt.hist(risk_score[y == 0], bins=20, alpha=0.6, label="Low Risk (0)")
plt.hist(risk_score[y == 1], bins=20, alpha=0.6, label="Medium Risk (1)")
plt.hist(risk_score[y == 2], bins=20, alpha=0.6, label="High Risk (2)")
plt.xlabel("Risk Score")
plt.ylabel("Frequency")
plt.title("Risk Score Distribution (EBM – Best Model)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "risk_score_distribution.png"))
plt.close()

print("All evaluation plots saved to docs/figures/")
