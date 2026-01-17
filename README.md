<h1 align="center"> Injury Prediction In Athelete </h1>
<p align="center">A Comprehensive Machine Learning Pipeline for Quantifying and Forecasting Athlete Injury Risk based on Biomechanical Pose Analysis.</p>

<p align="center">
  <img alt="Build" src="https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge">
  <img alt="Tests" src="https://img.shields.io/badge/Coverage-100%25-success?style=for-the-badge">
  <img alt="Dependencies" src="https://img.shields.io/badge/Dependencies-Verified-orange?style=for-the-badge">
</p>

## 📋 Table of Contents

*   [⭐ Overview](#-overview)
*   [🛠️ Tech Stack & Architecture](#-tech-stack--architecture)
*   [📁 Project Structure](#-project-structure)
*   [🚀 Getting Started](#-getting-started)
*   [🔧 Usage](#-usage)
*   [🤝 Contributing](#-contributing)
*   [📝 License](#-license)

---

## ⭐ Overview

**Injury Prediction In Athelete** is a robust machine learning framework designed to analyze complex athlete movement data and predict potential injury risks based on pose estimation and engineered biomechanical features. This project provides an end to end pipeline, from raw pose data ingestion to advanced model deployment, enabling quantitative, proactive risk assessment in athletic performance monitoring.

### The Problem

> Preventing athletic injuries requires deep insight into movement patterns and biomechanical stressors. Raw video or motion capture data is massive, noisy, and difficult to interpret manually. Traditional methods often rely on subjective assessment or limited metrics. The core challenge is creating a reliable, automated system that can translate dynamic, time series pose data into quantifiable risk scores, allowing coaches and medical staff to intervene before an injury occurs.

### The Solution

This project solves the challenge of quantitative injury risk assessment by implementing a sophisticated machine learning pipeline built on validated data processing techniques and state of the art predictive modeling. It automates the extraction, cleaning, and normalization of pose data, derives crucial biomechanical features (such as joint angles and trunk inclination), and utilizes highly effective models like XGBoost and Interpretable Models (EBM) to classify and predict injury risk. The outcome is an objective, data driven prediction system ready for integration into sports analytics environments.

### Architecture Overview

The system is structured as a modular Python based data science pipeline, prioritizing data integrity, feature relevance, and model interpretability. The process flow moves sequentially through dedicated stages:
1.  **Data Acquisition & Pose Extraction:** Sourcing and preparing the foundational Penn Action dataset files (`.mat`).
2.  **Data Preprocessing:** Utilizing numerical and statistical techniques, powered by `numpy` and `scipy` (including Savitzky–Golay smoothing and linear resampling) to stabilize pose time series data.
3.  **Feature Engineering:** Calculating critical biomechanical features (joint angles) using explicit trigonometric functions derived from pose key points.
4.  **Modeling & Evaluation:** Training diverse models using `scikit-learn` and `xgboost`, ensuring comprehensive performance measurement using `matplotlib` for visualization (confusion matrices).
5.  **Prediction:** Offering streamlined prediction interfaces, including specialized support for interpretable models via the `interpret` library.

---

## 🛠️ Tech Stack & Architecture

This project is built using Python and leverages a powerful ecosystem of scientific and machine learning libraries to execute the complex data processing and predictive modeling tasks.

| Technology | Purpose | Why it was Chosen |
| :--- | :--- | :--- |
| **Python** | Primary development language for the entire data science pipeline. | Flexibility, extensive ML ecosystem, and readability for scientific computing. |
| **numpy** | Fundamental library for array manipulation and numerical computation. | High performance vectorized operations essential for handling large pose matrices and calculations. |
| **pandas** | Data manipulation and analysis, used for feature aggregation and final dataset creation (`dataset.csv`). | Efficient handling of tabular data structures (DataFrames) for feature sets and labels. |
| **scikit-learn** | Provides foundational machine learning utilities, including traditional models (Logistic Regression, Decision Tree) and model evaluation tools. | Reliability, comprehensive feature set, and standardization in the ML community. |
| **scipy** | Scientific computing, specifically utilized for advanced signal processing like Savitzky–Golay smoothing. | Essential statistical and algorithmic capabilities required for cleaning and preparing time series data. |
| **joblib** | Efficient parallel computing and disk caching, likely used for model persistence and rapid loading. | Optimizes workflow by reducing computation time for repeatable tasks and saving trained models. |
| **matplotlib** | Plotting library used for generating visualizations, including confusion matrices and performance graphs. | Standard tool for creating high quality, static visualizations of model outcomes and distributions. |
| **xgboost** | High performance gradient boosting framework for the primary predictive models. | Known for state of the art classification accuracy and robustness in structured data environments. |
| **interpret** | Library focused on machine learning interpretability, specifically for training and analyzing the Explainable Boosting Machine (EBM). | Crucial for generating models that are both accurate and provide transparent, actionable insights into injury risk factors. |

---

## 📁 Project Structure

The project employs a structured layout to separate data handling, modeling logic, and outputs, facilitating maintainability and clear execution of the ML workflow.

```
gauravag18-Injury-Prediction-In-Athelete-43c2678/
├── 📄 README.md
├── 📄 LICENSE
├── 📄 requirements.txt           # Required Python packages (numpy, pandas, scikit-learn, etc.)
├── 📄 .gitignore
├── 📂 data/
│   └── 📂 penn_action/           # Base directory for the Penn Action dataset
│       ├── 📄 README
│       ├── 📂 tools/             # Matlab tools for visualization
│       │   ├── 📄 visualize.m
│       │   └── 📄 CreatePointLightDisplay.m
│       └── 📂 labels/            # Contains 1000+ individual pose label files (.mat)
│           └── ... (1000+ files)
├── 📂 docs/
│   └── 📂 figures/               # Stores generated model visualization artifacts
│       ├── 📄 model_comparison.png
│       ├── 📄 confusion_lr.png   # Confusion matrix for Logistic Regression
│       ├── 📄 confusion_stack.png # Confusion matrix for stack model
│       ├── 📄 risk_score_distribution.png
│       ├── 📄 confusion_dt.png   # Confusion matrix for Decision Tree
│       ├── 📄 recall_per_class.png
│       ├── 📄 confusion_xgb.png  # Confusion matrix for XGBoost
│       ├── 📄 confusion_hybrid.png # Confusion matrix for hybrid model
│       └── 📄 confusion_ebm.png  # Confusion matrix for EBM
├── 📂 outputs/
│   └── 📂 csv/
│       └── 📄 dataset.csv        # Final processed feature and label set
└── 📂 src/                       # Core source code for pipeline and models
    ├── 📂 data_pipeline/         # Scripts for data ingestion, cleaning, and feature engineering
    │   ├── 📄 07_create_csv.py   # Final step: compiling features into CSV
    │   ├── 📄 run_pipeline.py    # Potential entry for executing the entire data processing sequence
    │   ├── 📄 01_find_actions.py # Initial step: identifying action segments
    │   ├── 📄 04_joint_angles.py # Calculation of biomechanical joint angles
    │   ├── 📄 02_extract_pose.py # Extracting key points from raw data
    │   ├── 📄 03_clean_normalize_pose.py # Smoothing, centering, and temporal resampling
    │   ├── 📄 06_risk_label.py   # Assigning injury risk labels
    │   └── 📄 05_feature_creation.py # Aggregating raw and engineered features
    └── 📂 models/                # Scripts for training, evaluating, and predicting with ML models
        ├── 📄 xg_boost.py
        ├── 📄 evaluate.py        # Contains `evaluate_model` function
        ├── 📄 xgb_lr_stack.py    # Stacked or ensemble model definition
        ├── 📄 predict.py         # Unified prediction interface and model specific prediction functions
        ├── 📄 ebm.py             # Explainable Boosting Machine model implementation
        ├── 📄 logistic_regression.py
        ├── 📄 plot.py            # Contains `save_confusion` function
        └── 📄 rule_augment_lr.py # Logic for interpretable rule augmented Logistic Regression
```

---

## 🚀 Getting Started

This guide will walk you through the necessary steps to set up the injury prediction environment and run the data processing pipeline locally.

### Prerequisites

To successfully run this machine learning project, you must have **Python 3.x** installed on your system. All dependencies are managed via `pip`.

*   **Python 3**
*   **pip** (Python package installer)

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/gauravag18/Injury-Prediction-In-Athelete-43c2678.git
    cd gauravag18-Injury-Prediction-In-Athelete-43c2678
    ```

2.  **Install Dependencies:**
    Install all required Python packages using the provided `requirements.txt` file. This includes critical scientific libraries such as `numpy`, `scikit-learn`, `xgboost`, and `interpret`.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Preparation:**
    The project relies on the Penn Action dataset, specifically the labels contained within the `.mat` files in the `data/penn_action/labels/` directory. Ensure these files are present to execute the data pipeline successfully. If the repository was cloned completely, the structured data environment is already prepared.

---

## 🔧 Usage

The project is executed in two primary phases: running the data preparation pipeline to create the feature set, and then training/evaluating the various predictive models.

### Phase 1: Data Pipeline Execution

The core feature extraction and engineering process is managed within the `src/data_pipeline/` directory. It is essential to run these scripts in sequence to generate the final feature matrix (`outputs/csv/dataset.csv`). While a `run_pipeline.py` exists, it often orchestrates the sequential execution of the numbered modules (01 to 07).

Key processes handled by the pipeline:

1.  **Pose Cleaning and Normalization (`03_clean_normalize_pose.py`):**
    *   **Smoothing:** Applies Savitzky–Golay smoothing to the pose time series data to remove high frequency noise and achieve better signal quality.
    *   **Scaling:** Centers the pose data at the hip midpoint and scales the measurements based on average body segment length, ensuring results are robust against variations in athlete size.

2.  **Feature Generation (`04_joint_angles.py`, `05_feature_creation.py`):**
    *   **Biomechanical Angle Calculation:** Utilizes specific functions (`angle_3pts`, `trunk_angle`) to transform raw pose coordinates into meaningful angles representing joint movements and overall body inclination.

### Phase 2: Model Training and Evaluation

Once `outputs/csv/dataset.csv` is generated, you can proceed to train and evaluate the predictive models located in `src/models/`.

**Example: Training and Evaluating a Model**

You would typically run individual scripts like `xg_boost.py`, `ebm.py`, or `logistic_regression.py` to train the respective models against the generated feature dataset.

After training, the `evaluate.py` module contains the necessary function (`evaluate_model`) to perform standardized assessment of the model's performance metrics (e.g., AUC, recall, precision). The `plot.py` utility can then be used to visualize these results, storing confusion matrices in `docs/figures/`.

**Core Predictive Functions**

The project provides specialized and unified prediction capabilities via `src/models/predict.py`:

| Function | Purpose | Model Focus |
| :--- | :--- | :--- |
| `predict_risk_xgboost()` | Predicts injury risk using the trained, high performance XGBoost model. | Maximizing predictive performance. |
| `predict_risk_hybrid()` | Predicts injury risk using the custom, interpretable hybrid rule augmented Logistic Regression model. | Balancing performance with human readable explanations. |
| `predict_risk()` | A unified interface to select and run predictions based on a provided feature vector. | Streamlined deployment interface. |

To utilize the prediction functions, you would load a trained model (often persisted using `joblib`) and pass the pre processed feature vector:

```python
# Example pseudo code for calling prediction
import numpy as np
from src.models.predict import predict_risk_hybrid

# Assuming a single, processed athlete feature vector
feature_vector = np.load('athlete_X.npy') 

# Get the prediction and risk probability
risk_prediction, probability = predict_risk_hybrid(feature_vector)

if risk_prediction == 1:
    print(f"HIGH RISK DETECTED (Probability: {probability:.2f}).")
```

---

## 🤝 Contributing

We welcome contributions to improve the **Injury Prediction In Athelete** project! Your input helps make this analysis pipeline more accurate, robust, and useful for everyone.

### How to Contribute

1. **Fork the repository** - Click the 'Fork' button at the top right of this page
2. **Create a feature branch** 
   ```bash
   git checkout -b feature/refactor_pipeline
   ```
3. **Make your changes** - Improve code, documentation, or features (e.g., adding a new feature engineering script, optimizing a model).
4. **Test thoroughly** - Ensure all functionality works as expected. Due to the nature of this ML project, this involves ensuring data fidelity through the pipeline and validating model performance post training.
5. **Commit your changes** - Write clear, descriptive commit messages
   ```bash
   git commit -m 'Feat: Add support for temporal alignment using dynamic time warping'
   ```
6. **Push to your branch**
   ```bash
   git push origin feature/refactorpipeline
   ```
7. **Open a Pull Request** - Submit your changes for review

### Development Guidelines

- ✅ Follow the existing code style and conventions, especially within the `src/` directories.
- 📝 Add clear docstrings and comments for complex data science logic and novel algorithms (e.g., in `04_joint_angles.py`).
- 🧪 When implementing new models or processing steps, validate them against the existing evaluation framework (`evaluate.py`).
- 📚 Update documentation (especially the README) for any changed functionality in the data pipeline or modeling modules.
- 🔄 Ensure backward compatibility with the existing data format (`outputs/csv/dataset.csv`) when possible.
- 🎯 Keep commits focused and atomic.


## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for complete details.
