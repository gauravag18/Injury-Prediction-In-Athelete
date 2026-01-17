import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from interpret.glassbox import ExplainableBoostingClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "csv", "dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

X = df.drop(columns=["label", "risk_label", "clip"], errors="ignore")
y = df["risk_label"] if "risk_label" in df.columns else df["label"]

print("Dataset shape:", X.shape)
print("Class distribution:\n", y.value_counts())

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Preprocessing
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_proc = scaler.transform(imputer.transform(X_test))

# EBM Model
ebm = ExplainableBoostingClassifier(
    interactions=0,
    max_bins=256,
    learning_rate=0.05,
    random_state=42
)

# Training
ebm.fit(X_train_proc, y_train)

print("\nEBM training complete.")

# Evaluation
y_pred = ebm.predict(X_test_proc)

print("\n--- Test Performance ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Macro F1 :", f1_score(y_test, y_pred, average="macro"))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

joblib.dump(
    {
        "model": ebm,
        "imputer": imputer,
        "scaler": scaler
    },
    os.path.join(MODEL_DIR, "ebm.pkl")
)

print("\nEBM model saved to models/trained/ebm.pkl")
