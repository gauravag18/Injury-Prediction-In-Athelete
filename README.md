# 🏃‍♂️ Injury Prediction in Athletes using Biomechanical Features

This project focuses on predicting injury risk in athletes using pose-based biomechanical features extracted from sports activity videos. The system processes raw pose annotations, engineers joint-level features, and applies interpretable machine learning models to classify injury risk.

## 📌 Project Overview

**Input**: Pose annotations (Penn Action dataset .mat files)  
**Processing**: Joint normalization → joint angles → engineered features  
**Models**:
- Logistic Regression (baseline)
- Decision Tree (non-linear baseline)  
- **Hybrid Rule-Augmented Logistic Regression (final model)**

**Output**:
- Injury risk class: `0 = Low`, `1 = Medium`, `2 = High`
- Continuous risk score

The final model combines decision-tree-derived biomechanical rules with logistic regression for improved interpretability and calibrated risk estimation.

## 📂 Repository Structure

Injury_Prediction_In_Athlete/
├── data/
│ └── penn_action/ # Raw dataset (not tracked)
├── outputs/
│ ├── poses_raw/
│ ├── poses_clean/
│ ├── joint_angles/
│ ├── features/
│ └── csv/
│ └── dataset.csv # Final engineered dataset
├── src/
│ ├── data_pipeline/
│ │ ├── 01_find_actions.py
│ │ ├── 02_extract_pose.py
│ │ ├── 03_clean_normalize_pose.py
│ │ ├── 04_joint_angles.py
│ │ ├── 05_feature_creation.py
│ │ ├── 06_risk_label.py
│ │ ├── 07_create_csv.py
│ │ └── run_pipeline.py
│ └── models/
│ ├── logistic_regression.py # Baseline LR
│ ├── decision_tree.py # Decision Tree baseline
│ ├── train.py # Hybrid Rule-Augmented LR
│ ├── evaluate.py # Model evaluation
│ ├── predict.py # Inference function
│ ├── plot.py # Confusion matrices & comparison plots
│ └── chart.py # Recall & risk-score plots
├── models/
│ └── trained/
│ ├── preprocess.pkl
│ ├── decision_tree.pkl
│ ├── logistic_regression.pkl
│ ├── hybrid_rule_aug_lr.pkl
│ └── rule_encoder.pkl
├── docs/
│ └── figures/ # Saved plots
├── requirements.txt
├── .gitignore
└── README.md


## ⚙️ Installation

Create a virtual environment (recommended) and install dependencies:

```bash
pip install -r requirements.txt
Dependencies

-numpy
-pandas
-scikit-learn
-scipy
-joblib
-matplotlib

🔄 Data Pipeline
The complete preprocessing pipeline is implemented in:
src/data_pipeline/run_pipeline.py

Pipeline steps:

-Extract action labels
-Extract raw joint coordinates
-Clean & normalize poses
-Compute joint angles
-Engineer statistical features
-Assign injury risk labels
-Create final CSV dataset

Run the pipeline:

bash
python src/data_pipeline/run_pipeline.py
This produces: outputs/csv/dataset.csv

🧠 Model Training
1️⃣ Logistic Regression (Baseline)
bash
python src/models/logistic_regression.py
2️⃣ Decision Tree
bash
python src/models/decision_tree.py
3️⃣ Hybrid Rule-Augmented Logistic Regression (Final Model)
bash
python src/models/train.py

This script:

-Trains a relaxed decision tree to extract biomechanical rules
-Encodes tree leaves as rule features
-Combines original features + rule features
-Trains logistic regression on the hybrid feature space
-Saves all trained artifacts to models/trained/

📊 Evaluation
Evaluate all saved models:

bash
python src/models/evaluate.py
Metrics reported:

-Accuracy
-Macro F1-score
-Precision/Recall per class
-Confusion matrices

Note: Final performance is reported on a held-out test set (80–20 split). Full dataset evaluation is used only for diagnostic comparison.

📈 Visualizations
Generate plots:

bash
python src/models/plot.py
python src/models/chart.py
Generated figures:

-Confusion matrices (all models)
-Accuracy & Macro-F1 comparison
-Per-class recall bar chart
-Risk score distribution (hybrid model)

Saved to: docs/figures/

🔮 Inference (Prediction)
predict.py exposes a clean inference function:

python
from src.models.predict import predict_risk

pred_class, risk_score = predict_risk(feature_vector)
feature_vector must be the engineered feature array (same order as dataset.csv)

Returns:

Injury risk class (0 / 1 / 2)

Continuous risk score

🏁 Final Model
Production Model: Hybrid Rule-Augmented Logistic Regression

Why this model?

✅ Best test-set Macro F1

✅ Interpretable rule-based structure

✅ Calibrated risk scores

✅ Biomechanically meaningful splits (knee flexion, hip abduction, torso motion)

🔁 Reproducibility
All experiments use fixed random seeds (random_state=42)

🚀 Future Work
-Real-time inference directly from video streams
-Integration with a lightweight web UI (FastAPI/Streamlit)
-Extension to additional sports activities
-Incorporation of anomaly detection based injury signals

👤 Author
Gaurav Agarwalla

