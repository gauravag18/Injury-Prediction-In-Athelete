import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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

X = df.drop(columns=["label", "risk_label", "clip"], errors="ignore")
y = df["risk_label"] if "risk_label" in df.columns else df["label"]

print("Dataset shape:", X.shape)
print("Class distribution:\n", y.value_counts())

# Test + Train split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

y_test = y_test.reset_index(drop=True)

# Preprocessing: Imputation + Scaling
preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_pre = preprocess.fit_transform(X_train)
X_test_pre = preprocess.transform(X_test)

print("\nPreprocessing complete.")
print("Preprocessed train shape:", X_train_pre.shape)

# Decision Tree→ Rule Generator
tree = DecisionTreeClassifier(
    max_depth=7,           
    min_samples_leaf=5,    
    class_weight="balanced",
    random_state=42
)

tree.fit(X_train_pre, y_train)

print("\nDecision Tree trained.")
print("Tree depth:", tree.get_depth())
print("Number of leaves:", tree.get_n_leaves())

# Print interpretable rules
tree_rules = export_text(tree, feature_names=X.columns.tolist())
print("\nDecision Tree Rules (truncated):")
print(tree_rules[:1200])

# Convert tree leaves → rule features
train_leaves = tree.apply(X_train_pre)
test_leaves = tree.apply(X_test_pre)

encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

X_train_rules = encoder.fit_transform(train_leaves.reshape(-1, 1))
X_test_rules = encoder.transform(test_leaves.reshape(-1, 1))

print("Rule feature shape:", X_train_rules.shape)

# HYBRID FEATURES (RULES + ORIGINAL FEATURES)
X_train_hybrid = np.hstack([X_train_pre, X_train_rules])
X_test_hybrid = np.hstack([X_test_pre, X_test_rules])

print("Hybrid feature shape:", X_train_hybrid.shape)

# Logistic Regression on Hybrid Features
log_reg = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42
)

log_reg.fit(X_train_hybrid, y_train)

print("\nHybrid Rule-Augmented Logistic Regression trained.")

# Evaluation
y_pred = log_reg.predict(X_test_hybrid)
y_prob = log_reg.predict_proba(X_test_hybrid)

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

# Save preprocess + model
joblib.dump(preprocess, os.path.join(MODEL_DIR, "preprocess.pkl"))
joblib.dump(tree, os.path.join(MODEL_DIR, "decision_tree.pkl"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "rule_encoder.pkl"))
joblib.dump(log_reg, os.path.join(MODEL_DIR, "hybrid_rule_aug_lr.pkl"))

print("All models and encoders saved.")