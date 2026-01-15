🏃‍♂️ Injury Prediction in Athletes Using Biomechanical Features

This project focuses on predicting injury risk in athletes using pose-based biomechanical features extracted from sports activity videos. The system processes raw pose annotations, engineers joint-level features, and applies interpretable machine learning models to classify injury risk.

The final model uses a Hybrid Rule-Augmented Logistic Regression approach, combining decision-tree-derived biomechanical rules with logistic regression for improved interpretability and calibrated risk estimation.

📌 Project Overview

Input

Pose annotations from the Penn Action dataset (.mat files)

Processing

Joint normalization

Joint angle computation

Statistical & biomechanical feature engineering

Models

Logistic Regression (baseline)

Decision Tree (non-linear baseline)

Hybrid Rule-Augmented Logistic Regression (final model)

Output

Injury Risk Class

0 → Low

1 → Medium

2 → High

Continuous injury risk score

📂 Repository Structure
Injury_Prediction_In_Athlete/
│
├── data/
│   └── penn_action/               # Raw dataset (not tracked)
│
├── outputs/
│   ├── poses_raw/
│   ├── poses_clean/
│   ├── joint_angles/
│   ├── features/
│   └── csv/
│       └── dataset.csv            # Final engineered dataset
│
├── src/
│   ├── data_pipeline/
│   │   ├── 01_find_actions.py
│   │   ├── 02_extract_pose.py
│   │   ├── 03_clean_normalize_pose.py
│   │   ├── 04_joint_angles.py
│   │   ├── 05_feature_creation.py
│   │   ├── 06_risk_label.py
│   │   ├── 07_create_csv.py
│   │   └── run_pipeline.py
│   │
│   └── models/
│       ├── logistic_regression.py     # Baseline LR
│       ├── decision_tree.py           # Decision Tree baseline
│       ├── train.py                   # Hybrid Rule-Augmented LR
│       ├── evaluate.py                # Model evaluation
│       ├── predict.py                 # Inference function
│       ├── plot.py                    # Confusion matrices & comparisons
│       └── chart.py                   # Recall & risk-score plots
│
├── models/
│   └── trained/
│       ├── preprocess.pkl
│       ├── decision_tree.pkl
│       ├── logistic_regression.pkl
│       ├── hybrid_rule_aug_lr.pkl
│       └── rule_encoder.pkl
│
├── docs/
│   └── figures/                   # Saved plots
│
├── requirements.txt
├── .gitignore
└── README.md

⚙️ Installation

Create a virtual environment (recommended) and install dependencies:

pip install -r requirements.txt

Dependencies

numpy

pandas

scikit-learn

scipy

joblib

matplotlib

🔄 Data Pipeline

The complete preprocessing pipeline is implemented in:

src/data_pipeline/run_pipeline.py

Pipeline Steps

Extract action labels

Extract raw joint coordinates

Clean & normalize poses

Compute joint angles

Engineer statistical features

Assign injury risk labels

Create final CSV dataset

Run the Pipeline
python src/data_pipeline/run_pipeline.py


Output:

outputs/csv/dataset.csv

🧠 Model Training
1️⃣ Logistic Regression (Baseline)
python src/models/logistic_regression.py

2️⃣ Decision Tree (Non-linear Baseline)
python src/models/decision_tree.py

3️⃣ Hybrid Rule-Augmented Logistic Regression (Final Model)
python src/models/train.py


This script:

Trains a relaxed decision tree to extract biomechanical rules

Encodes tree leaves as rule-based features

Combines original features + rule features

Trains logistic regression on the hybrid feature space

Saves all trained artifacts to:

models/trained/

📊 Evaluation

Evaluate all trained models:

python src/models/evaluate.py

Metrics Reported

Accuracy

Macro F1-score

Precision / Recall per class

Confusion matrices

Note:
Final performance is reported on a held-out test set (80–20 split). Full-dataset evaluation is used only for diagnostic comparison.

📈 Visualizations

Generate plots:

python src/models/plot.py
python src/models/chart.py

Generated Figures

Confusion matrices (all models)

Accuracy & Macro-F1 comparison

Per-class recall bar chart

Risk score distribution (hybrid model)

Saved to:

docs/figures/

🔮 Inference (Prediction)

predict.py exposes a clean inference function:

from src.models.predict import predict_risk

pred_class, risk_score = predict_risk(feature_vector)

Input

feature_vector: Engineered feature array (same order as dataset.csv)

Output

pred_class: Injury risk class (0 / 1 / 2)

risk_score: Continuous injury risk score

🏁 Final Model

Production Model: Hybrid Rule-Augmented Logistic Regression

Why This Model?

✅ Best test-set Macro F1
✅ Interpretable rule-based structure
✅ Calibrated risk scores
✅ Biomechanically meaningful splits (e.g., knee flexion, hip abduction, torso motion)

🔁 Reproducibility

All experiments use fixed random seeds:

random_state = 42

🚀 Future Work

Real-time inference directly from video streams

Integration with a lightweight web UI (FastAPI / Streamlit)

Extension to additional sports activities

Incorporation of anomaly-detection-based injury signals

👤 Author

Gaurav Agarwalla
