import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "csv", "dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")
FIG_DIR = os.path.join(PROJECT_ROOT, "docs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
X = df.drop(columns=["label", "risk_label", "clip"], errors="ignore")
y = df["risk_label"] if "risk_label" in df.columns else df["label"]

lr_pipeline = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
tree = joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl"))
hybrid_lr = joblib.load(os.path.join(MODEL_DIR, "hybrid_rule_aug_lr.pkl"))
preprocess = joblib.load(os.path.join(MODEL_DIR, "preprocess.pkl"))
rule_encoder = joblib.load(os.path.join(MODEL_DIR, "rule_encoder.pkl"))

# Logistic Regression
y_pred_lr = lr_pipeline.predict(X)

# Decision Tree
X_pre = preprocess.transform(X)
y_pred_dt = tree.predict(X_pre)

# Hybrid model
leaves = tree.apply(X_pre)
rule_features = rule_encoder.transform(leaves.reshape(-1, 1))
X_hybrid = np.hstack([X_pre, rule_features])
y_pred_hybrid = hybrid_lr.predict(X_hybrid)

# Per-class Recall Bar Chart
labels = [0, 1, 2]
model_names = ["Logistic Regression", "Decision Tree", "Hybrid Model"]

recall_lr = recall_score(y, y_pred_lr, average=None, labels=labels)
recall_dt = recall_score(y, y_pred_dt, average=None, labels=labels)
recall_hybrid = recall_score(y, y_pred_hybrid, average=None, labels=labels)

x = np.arange(len(labels))
width = 0.25

plt.figure()
plt.bar(x - width, recall_lr, width, label="Logistic Regression")
plt.bar(x, recall_dt, width, label="Decision Tree")
plt.bar(x + width, recall_hybrid, width, label="Hybrid Model")

plt.xticks(x, ["Low Risk (0)", "Medium Risk (1)", "High Risk (2)"])
plt.ylabel("Recall")
plt.title("Per-Class Recall Comparison")
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "recall_per_class.png"))
plt.close()

# Risk score definition
y_prob = hybrid_lr.predict_proba(X_hybrid)
risk_score = y_prob[:, 1] + 2 * y_prob[:, 2]

plt.figure()
plt.hist(risk_score[y == 0], bins=20, alpha=0.6, label="Low Risk (0)")
plt.hist(risk_score[y == 1], bins=20, alpha=0.6, label="Medium Risk (1)")
plt.hist(risk_score[y == 2], bins=20, alpha=0.6, label="High Risk (2)")

plt.xlabel("Risk Score")
plt.ylabel("Frequency")
plt.title("Risk Score Distribution (Hybrid Model)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "risk_score_distribution.png"))
plt.close()

print("Extra plots saved to docs/figures/")
