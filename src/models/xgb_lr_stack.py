import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "csv", "dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

X = df.drop(columns=["label", "risk_label", "clip"], errors="ignore")
y = df["risk_label"] if "risk_label" in df.columns else df["label"]

print("Dataset shape:", X.shape)
print("Class distribution:\n", y.value_counts())

# Train / test split (80–20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Preprocessing
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_scaled = scaler.transform(imputer.transform(X_test))

# Class Weights
class_counts = y_train.value_counts().to_dict()
total = len(y_train)
n_classes = len(class_counts)

class_weights = {
    c: total / (n_classes * count)
    for c, count in class_counts.items()
}

sample_weight = y_train.map(class_weights).values

# Base model: XGBoost
xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.5,
    reg_alpha=0.2,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

xgb.fit(
    X_train_scaled,
    y_train,
    sample_weight=sample_weight
)

print("\nXGBoost base model trained.")

# Meta features = XGBoost probabilities
X_train_meta = xgb.predict_proba(X_train_scaled)
X_test_meta = xgb.predict_proba(X_test_scaled)

# Meta-learner: Logistic Regression
meta_lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

meta_lr.fit(X_train_meta, y_train)

print("Logistic Regression meta-learner trained.")

# Final evaluation
y_pred = meta_lr.predict(X_test_meta)

print("\n--- STACKED MODEL PERFORMANCE ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Macro F1 :", f1_score(y_test, y_pred, average="macro"))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

joblib.dump(
    {
        "xgb": xgb,
        "meta_lr": meta_lr,
        "imputer": imputer,
        "scaler": scaler
    },
    os.path.join(MODEL_DIR, "xgb_lr_stack.pkl")
)

print("\nStacked model saved to models/trained/xgb_lr_stack.pkl")
