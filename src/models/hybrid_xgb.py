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

from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier


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

os.makedirs(MODEL_DIR, exist_ok=True)


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


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# PREPROCESSING

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

print("\nPreprocessing complete.")
print("Train shape:", X_train_scaled.shape)


# STAGE 1:
# EXTRA TREES FEATURE LEARNER

extra = ExtraTreesClassifier(
    n_estimators=400,
    max_depth=8,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

extra.fit(X_train_scaled, y_train)

print("\nExtra Trees model trained.")


# LEAF EMBEDDINGS

train_leaves = extra.apply(X_train_scaled)
test_leaves = extra.apply(X_test_scaled)

train_leaf_features = train_leaves.reshape(
    train_leaves.shape[0],
    -1
)

test_leaf_features = test_leaves.reshape(
    test_leaves.shape[0],
    -1
)

print(
    "Leaf embedding shape:",
    train_leaf_features.shape
)


# HYBRID FEATURE CONCATENATION

X_train_hybrid = np.hstack([
    X_train_scaled,
    train_leaf_features
])

X_test_hybrid = np.hstack([
    X_test_scaled,
    test_leaf_features
])

print(
    "\nHybrid feature shape:",
    X_train_hybrid.shape
)


# CLASS WEIGHTS

class_counts = y_train.value_counts().to_dict()

total = len(y_train)

n_classes = len(class_counts)

class_weights = {
    c: total / (n_classes * count)
    for c, count in class_counts.items()
}

sample_weight = y_train.map(class_weights).values


# STAGE 2:
# XGBOOST META CLASSIFIER

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=450,
    max_depth=5,
    learning_rate=0.035,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=1,
    gamma=0.1,
    reg_alpha=0.05,
    reg_lambda=1.2,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42
)

xgb.fit(
    X_train_hybrid,
    y_train,
    sample_weight=sample_weight
)


# EVALUATION

y_pred = xgb.predict(X_test_hybrid)

print("\n================ TEST PERFORMANCE ================\n")

print(
    "Accuracy :",
    accuracy_score(y_test, y_pred)
)

print(
    "Macro F1 :",
    f1_score(y_test, y_pred, average="macro")
)

print("\nClassification Report:\n")

print(
    classification_report(y_test, y_pred)
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(y_test, y_pred)
)


# FEATURE IMPORTANCE

feature_names = (
    list(X.columns)
    +
    [
        f"leaf_{i}"
        for i in range(
            train_leaf_features.shape[1]
        )
    ]
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": xgb.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTop 15 Important Features:\n")

print(
    importance_df.head(15)
)


# SAVE MODEL

joblib.dump(
    {
        "imputer": imputer,
        "scaler": scaler,
        "extra_trees": extra,
        "xgb": xgb
    },
    os.path.join(
        MODEL_DIR,
        "extra_xgb_hybrid.pkl"
    )
)

print(
    "\nExtraTrees + XGBoost Hybrid saved successfully."
)