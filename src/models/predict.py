import os
import numpy as np
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")

# BEST MODEL: XGBoost
xgb_bundle = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
xgb_model = xgb_bundle["model"]
xgb_imputer = xgb_bundle["imputer"]
xgb_scaler = xgb_bundle["scaler"]

# INTERPRETABLE FALLBACK: Hybrid Rule-Augmented LR
hybrid_model = joblib.load(os.path.join(MODEL_DIR, "hybrid_rule_aug_lr.pkl"))
preprocess = joblib.load(os.path.join(MODEL_DIR, "preprocess.pkl"))
tree = joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl"))
rule_encoder = joblib.load(os.path.join(MODEL_DIR, "rule_encoder.pkl"))

# Prediction functions
def predict_risk_xgboost(feature_vector):
    """
    Predict injury risk using the best-performing XGBoost model.

    Parameters
    ----------
    feature_vector : np.ndarray of shape (n_features,)
        Engineered biomechanical features (same order as dataset.csv)

    Returns
    -------
    predicted_class : int
        0 = low, 1 = medium, 2 = high

    risk_score : float
        Continuous injury risk score
    """

    X = feature_vector.reshape(1, -1)

    # Preprocess
    X_proc = xgb_scaler.transform(xgb_imputer.transform(X))

    # Prediction
    probs = xgb_model.predict_proba(X_proc)[0]
    predicted_class = int(np.argmax(probs))

    # Risk score (monotonic)
    risk_score = probs[1] + 2.0 * probs[2]

    return predicted_class, float(risk_score)


def predict_risk_hybrid(feature_vector):
    """
    Predict injury risk using the interpretable hybrid rule-augmented LR model.

    Parameters
    ----------
    feature_vector : np.ndarray of shape (n_features,)

    Returns
    -------
    predicted_class : int
    risk_score : float
    """

    X = feature_vector.reshape(1, -1)

    # Preprocess original features
    X_pre = preprocess.transform(X)

    # Generate rule features
    leaf = tree.apply(X_pre)
    rule_feat = rule_encoder.transform(leaf.reshape(-1, 1))

    # Hybrid feature vector
    X_hybrid = np.hstack([X_pre, rule_feat])

    probs = hybrid_model.predict_proba(X_hybrid)[0]
    predicted_class = int(np.argmax(probs))

    risk_score = probs[1] + 2.0 * probs[2]

    return predicted_class, float(risk_score)


def predict_risk(feature_vector, model_type="xgboost"):
    """
    Unified prediction interface.

    Parameters
    ----------
    feature_vector : np.ndarray
    model_type : str
        "xgboost" (default, best accuracy)
        "hybrid"  (interpretable alternative)

    Returns
    -------
    predicted_class : int
    risk_score : float
    """

    if model_type == "xgboost":
        return predict_risk_xgboost(feature_vector)
    elif model_type == "hybrid":
        return predict_risk_hybrid(feature_vector)
    else:
        raise ValueError(
            "Invalid model_type. Choose 'xgboost' or 'hybrid'."
        )
