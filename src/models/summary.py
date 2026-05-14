import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "csv",
    "dataset.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "trained"
)

FIG_DIR = os.path.join(
    PROJECT_ROOT,
    "docs",
    "figures"
)

os.makedirs(FIG_DIR, exist_ok=True)

# LOAD DATASET

df = pd.read_csv(CSV_PATH)

X = df.drop(
    columns=["label", "risk_label", "clip"],
    errors="ignore"
)

y = (
    df["risk_label"]
    if "risk_label" in df.columns
    else df["label"]
)

print("Dataset shape:", X.shape)

print("\nClass distribution:\n")
print(y.value_counts())

# LOAD MODELS

lr_pipeline = joblib.load(
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)

tree = joblib.load(
    os.path.join(MODEL_DIR, "decision_tree.pkl")
)

hybrid_lr = joblib.load(
    os.path.join(MODEL_DIR, "hybrid_rule_aug_lr.pkl")
)

preprocess = joblib.load(
    os.path.join(MODEL_DIR, "preprocess.pkl")
)

rule_encoder = joblib.load(
    os.path.join(MODEL_DIR, "rule_encoder.pkl")
)

xgb_bundle = joblib.load(
    os.path.join(MODEL_DIR, "xgboost.pkl")
)

xgb = xgb_bundle["model"]
xgb_imputer = xgb_bundle["imputer"]
xgb_scaler = xgb_bundle["scaler"]

ebm_bundle = joblib.load(
    os.path.join(MODEL_DIR, "ebm.pkl")
)

ebm = ebm_bundle["model"]
ebm_imputer = ebm_bundle["imputer"]
ebm_scaler = ebm_bundle["scaler"]

# OPTIONAL HYBRID XGB MODEL

hybrid_path = os.path.join(
    MODEL_DIR,
    "rule_guided_explainable_hybrid.pkl"
)

has_hybrid_xgb = os.path.exists(hybrid_path)

if has_hybrid_xgb:

    hybrid_bundle = joblib.load(hybrid_path)

    hybrid_tree = hybrid_bundle["tree"]
    hybrid_encoder = hybrid_bundle["encoder"]
    hybrid_imputer = hybrid_bundle["imputer"]
    hybrid_scaler = hybrid_bundle["scaler"]
    hybrid_xgb = hybrid_bundle["hybrid_ebm"]

    print("\nHybrid XGB model loaded.")

# PREDICTIONS

predictions = {}
probabilities = {}

# LOGISTIC REGRESSION

y_pred_lr = lr_pipeline.predict(X)

predictions["Logistic Regression"] = y_pred_lr

# DECISION TREE

X_pre = preprocess.transform(X)

y_pred_dt = tree.predict(X_pre)

predictions["Decision Tree"] = y_pred_dt

# HYBRID RULE AUGMENTED LR

leaves = tree.apply(X_pre)

rule_features = rule_encoder.transform(
    leaves.reshape(-1, 1)
)

X_hybrid = np.hstack([
    X_pre,
    rule_features
])

y_pred_hybrid = hybrid_lr.predict(X_hybrid)

predictions["Hybrid Rule LR"] = y_pred_hybrid

# XGBOOST

X_xgb = xgb_scaler.transform(
    xgb_imputer.transform(X)
)

y_pred_xgb = xgb.predict(X_xgb)

predictions["XGBoost"] = y_pred_xgb

probabilities["XGBoost"] = xgb.predict_proba(X_xgb)

# EBM

X_ebm = ebm_scaler.transform(
    ebm_imputer.transform(X)
)

y_pred_ebm = ebm.predict(X_ebm)

predictions["EBM"] = y_pred_ebm

probabilities["EBM"] = ebm.predict_proba(X_ebm)

# HYBRID XGB

if has_hybrid_xgb:

    X_hybrid_base = hybrid_scaler.transform(
        hybrid_imputer.transform(X)
    )

    hybrid_leaves = hybrid_tree.apply(
        X_hybrid_base
    )

    hybrid_rule_feat = hybrid_encoder.transform(
        hybrid_leaves.reshape(-1, 1)
    )

    X_hybrid_final = np.hstack([
        X_hybrid_base,
        hybrid_rule_feat
    ])

    y_pred_hybrid_xgb = hybrid_xgb.predict(
        X_hybrid_final
    )

    predictions["Hybrid XGB"] = y_pred_hybrid_xgb

    probabilities["Hybrid XGB"] = hybrid_xgb.predict_proba(
        X_hybrid_final
    )

# EVALUATION FUNCTION

def evaluate_model(name, y_true, y_pred):

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print(
        "Accuracy :",
        accuracy_score(y_true, y_pred)
    )

    print(
        "Macro F1 :",
        f1_score(
            y_true,
            y_pred,
            average="macro"
        )
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred
        )
    )

    print("\nConfusion Matrix:\n")

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )

# PRINT RESULTS

for model_name, preds in predictions.items():

    evaluate_model(
        model_name,
        y,
        preds
    )

# CONFUSION MATRICES

def save_confusion_matrix(
    y_true,
    y_pred,
    title,
    filename
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    disp.plot(
        cmap="Blues",
        values_format="d",
        ax=ax
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        os.path.join(FIG_DIR, filename),
        dpi=300
    )

    plt.close()

for model_name, preds in predictions.items():

    file_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    save_confusion_matrix(
        y,
        preds,
        model_name,
        f"confusion_{file_name}.png"
    )

# MODEL PERFORMANCE COMPARISON

model_names = list(predictions.keys())

accuracy_scores = [
    accuracy_score(y, predictions[m])
    for m in model_names
]

macro_f1_scores = [
    f1_score(
        y,
        predictions[m],
        average="macro"
    )
    for m in model_names
]

x = np.arange(len(model_names))

width = 0.35

plt.figure(figsize=(11, 5))

plt.bar(
    x - width / 2,
    accuracy_scores,
    width,
    label="Accuracy"
)

plt.bar(
    x + width / 2,
    macro_f1_scores,
    width,
    label="Macro F1"
)

plt.xticks(
    x,
    model_names,
    rotation=15
)

plt.ylabel("Score")

plt.ylim(0, 1.0)

plt.title("Model Performance Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "model_comparison.png"
    ),
    dpi=300
)

plt.close()

# PER CLASS RECALL

labels = [0, 1, 2]

recall_data = []

for model_name in model_names:

    recalls = recall_score(
        y,
        predictions[model_name],
        average=None,
        labels=labels
    )

    recall_data.append(recalls)

x = np.arange(len(labels))

width = 0.12

plt.figure(figsize=(12, 5))

for i, recalls in enumerate(recall_data):

    plt.bar(
        x + (i - len(model_names)/2) * width,
        recalls,
        width,
        label=model_names[i]
    )

plt.xticks(
    x,
    [
        "Low Risk (0)",
        "Medium Risk (1)",
        "High Risk (2)"
    ]
)

plt.ylabel("Recall")

plt.ylim(0, 1.0)

plt.title("Per-Class Recall Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "recall_per_class.png"
    ),
    dpi=300
)

plt.close()

# RISK SCORE DISTRIBUTION

best_model_name = "EBM"

if has_hybrid_xgb:
    best_model_name = "Hybrid XGB"

best_probs = probabilities[best_model_name]

risk_score = (
    best_probs[:, 1]
    + 2 * best_probs[:, 2]
)

plt.figure(figsize=(8, 5))

plt.hist(
    risk_score[y == 0],
    bins=20,
    alpha=0.6,
    label="Low Risk (0)"
)

plt.hist(
    risk_score[y == 1],
    bins=20,
    alpha=0.6,
    label="Medium Risk (1)"
)

plt.hist(
    risk_score[y == 2],
    bins=20,
    alpha=0.6,
    label="High Risk (2)"
)

plt.xlabel("Risk Score")

plt.ylabel("Frequency")

plt.title(
    f"Risk Score Distribution ({best_model_name})"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIG_DIR,
        "risk_score_distribution.png"
    ),
    dpi=300
)

plt.close()

# SUMMARY TABLE

summary_df = pd.DataFrame({
    "Model": model_names,
    "Accuracy": accuracy_scores,
    "Macro_F1": macro_f1_scores
})

summary_df = summary_df.sort_values(
    by="Macro_F1",
    ascending=False
)

print("\nMODEL SUMMARY : \n")

print(summary_df)

summary_df.to_csv(
    os.path.join(
        FIG_DIR,
        "model_summary.csv"
    ),
    index=False
)

print(
    "\nAll evaluation plots and reports saved to docs/figures/"
)