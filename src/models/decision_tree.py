import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "csv", "dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Features and target
X = df.drop(columns=["label", "risk_label", "clip"], errors="ignore")
y = df["risk_label"] if "risk_label" in df.columns else df["label"]

print("Dataset shape:", X.shape)
print("Class distribution:\n", y.value_counts())

# Train / Test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Reset index for safe printing later
y_test = y_test.reset_index(drop=True)

print("\nTrain size:", X_train.shape)
print("Test size :", X_test.shape)

# Decision Tree Model
# Keep tree shallow for interpretability
tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

tree.fit(X_train, y_train)
print("\nDecision Tree training complete.")
print("Tree depth:", tree.get_depth())
print("Number of leaves:", tree.get_n_leaves())

# Evaluate on test set
y_pred = tree.predict(X_test)
y_prob = tree.predict_proba(X_test)

print("\n--- Test Performance ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Macro F1 :", f1_score(y_test, y_pred, average="macro"))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Risk score computation
if y_prob.shape[1] == 3:
    risk_score = y_prob[:, 1] + 2 * y_prob[:, 2]
else:
    risk_score = y_prob[:, 1]

print("\nSample Risk Scores:")
for i in range(min(5, len(risk_score))):
    print(f"True label: {y_test[i]} | Risk score: {risk_score[i]:.3f}")

# Save the trained Decision Tree model
joblib.dump(tree, os.path.join(MODEL_DIR, "decision_tree.pkl"))
print("\nDecision Tree model saved successfully.")
