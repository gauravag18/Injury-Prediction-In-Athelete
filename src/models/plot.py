import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score

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

# Confusion matrix plots
def save_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename))
    plt.close()

save_confusion(y, y_pred_lr, "Logistic Regression", "confusion_lr.png")
save_confusion(y, y_pred_dt, "Decision Tree", "confusion_dt.png")
save_confusion(y, y_pred_hybrid, "Hybrid Rule-Augmented LR", "confusion_hybrid.png")

# Model comparison bar chart
models = ["Logistic Regression", "Decision Tree", "Hybrid Model"]
accuracy = [
    accuracy_score(y, y_pred_lr),
    accuracy_score(y, y_pred_dt),
    accuracy_score(y, y_pred_hybrid)
]
macro_f1 = [
    f1_score(y, y_pred_lr, average="macro"),
    f1_score(y, y_pred_dt, average="macro"),
    f1_score(y, y_pred_hybrid, average="macro")
]

x = np.arange(len(models))
width = 0.35

plt.figure()
plt.bar(x - width/2, accuracy, width, label="Accuracy")
plt.bar(x + width/2, macro_f1, width, label="Macro F1")
plt.xticks(x, models, rotation=15)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_comparison.png"))
plt.close()

print("Plots saved to docs/figures/")
