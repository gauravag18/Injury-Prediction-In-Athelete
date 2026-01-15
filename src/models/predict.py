import os
import numpy as np
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "trained")

model = joblib.load(os.path.join(MODEL_DIR, "hybrid_rule_aug_lr.pkl"))
preprocess = joblib.load(os.path.join(MODEL_DIR, "preprocess.pkl"))
tree = joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl"))
rule_encoder = joblib.load(os.path.join(MODEL_DIR, "rule_encoder.pkl"))

# Prediction function
def predict_risk(feature_vector):
    """
    Predict injury risk for a single sample.

    Parameters
    ----------
    feature_vector : np.ndarray, shape (n_features,)
        Engineered biomechanical features produced by the data pipeline
        (same order as columns in outputs/csv/dataset.csv).

    Returns
    -------
    predicted_class : int
        Injury risk class (0 = low, 1 = medium, 2 = high).

    risk_score : float
        Continuous injury risk score.
    """

    X = feature_vector.reshape(1, -1)

    # Preprocess original features
    X_pre = preprocess.transform(X)

    # Generate rule-based features
    leaf = tree.apply(X_pre)
    rule_feat = rule_encoder.transform(leaf.reshape(-1, 1))

    # Hybrid feature vector
    X_hybrid = np.hstack([X_pre, rule_feat])

    # Prediction
    probs = model.predict_proba(X_hybrid)[0]
    predicted_class = int(model.predict(X_hybrid)[0])

    # Risk score definition
    risk_score = probs[1] + 2 * probs[2]

    return predicted_class, float(risk_score)
