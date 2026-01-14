import pandas as pd
import numpy as np
import os 

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# 1. Load Dataset
output_dir = "../../outputs/risk_scores"
os.makedirs(output_dir, exist_ok=True)
CSV_PATH = "../../outputs/csv/dataset.csv" 

df = pd.read_csv(CSV_PATH)

# Drop non-feature columns
X = df.drop(columns=["label", "clip"])
y = df["label"]

print("Dataset shape:", X.shape)
print("Class distribution:")
print(y.value_counts())

# 2. Train / Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size :", X_test.shape)

# 3. Train Decision Tree (Rule Generator)
tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

tree.fit(X_train, y_train)

print("\nDecision Tree trained.")
print("Number of leaves:", tree.get_n_leaves())

# 4. Convert Tree Leaves → Rule Features
# Each sample is assigned to a leaf (rule)
train_leaves = tree.apply(X_train)
test_leaves = tree.apply(X_test)

# One-hot encode leaf indices (rules)
encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

X_train_rules = encoder.fit_transform(train_leaves.reshape(-1, 1))
X_test_rules = encoder.transform(test_leaves.reshape(-1, 1))

print("\nRule feature shape (train):", X_train_rules.shape)
print("Rule feature shape (test) :", X_test_rules.shape)

# 5. Logistic Regression on Rules
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42
)

log_reg.fit(X_train_rules, y_train)

print("\nLogistic Regression trained on rule features.")

# 6. Evaluation
y_pred = log_reg.predict(X_test_rules)
y_prob = log_reg.predict_proba(X_test_rules)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Risk Score Computation
# Risk score emphasizes higher injury severity
# risk_score = P(medium risk) + 2 * P(high risk)
risk_score = y_prob[:, 1] + 2 * y_prob[:, 2]

print("\nSample Risk Scores:")
for i in range(5):
    print(f"Sample {i} | True label: {y_test.iloc[i]} | Risk score: {risk_score[i]:.3f}")

# Save risk scores
np.save("../outputs/risk_scores/rule_augmented_risk_scores.npy", risk_score)

print("\nRisk scores saved successfully.")
