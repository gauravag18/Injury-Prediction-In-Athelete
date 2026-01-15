<h1 align="center"> Injury Prediction In Athelete </h1>
<p align="center"> Leveraging advanced pose analysis and machine learning to proactively quantify and predict biomechanical injury risk in athletic movements. </p>

<p align="center">
  <img alt="Build" src="https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge">
  <img alt="Issues" src="https://img.shields.io/badge/Issues-0%20Open-blue?style=for-the-badge">
  <img alt="Contributions" src="https://img.shields.io/badge/Contributions-Welcome-orange?style=for-for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge">
  <img alt="Python" src="https://img.shields.io/badge/Language-Python-blue?style=for-the-badge&logo=python">
  <img alt="Model" src="https://img.shields.io/badge/Model-Scikit--learn-informational?style=for-the-badge">
</p>

## 📑 Table of Contents

*   [Overview](#-overview)
*   [Key Features](#-key-features)
*   [Tech Stack & Architecture](#-tech-stack--architecture)
*   [Project Structure](#-project-structure)
*   [Getting Started](#-getting-started)
*   [Usage](#-usage)
*   [Contributing](#-contributing)
*   [License](#-license)

---

## ⭐ Overview

Injury Prediction In Athlete is a sophisticated machine learning framework designed to analyze athletic movements, extract critical biomechanical features, and predict the potential risk of injury. It provides actionable predictive insights for sports training, derived from detailed kinematic and pose analysis of complex actions.

### Architecture Overview

The core architecture follows a standardized Machine Learning pipeline approach, built entirely using Python and specialized scientific computing libraries. The system is divided into two major components: the **Data Pipeline** for robust feature engineering, and the **Modeling Suite** for training, evaluation, and risk prediction. The entire process leverages the power of `scikit-learn` for predictive modeling, `pandas` and `numpy` for high-performance data manipulation, and `scipy` for advanced signal processing like smoothing and interpolation.

---

## ✨ Key Features

This project offers a comprehensive suite of functionalities engineered to transform raw motion data into reliable risk predictions, focusing on actionable intelligence for the user:

### 🔬 Robust Data Preprocessing and Normalization

*   **Pose Standardization:** Ensures features are comparable across all athletes, regardless of body size or camera position, by **centering the pose at the hip midpoint** and **scaling features using the average body segment length**.
*   **Temporal Consistency:** Utilizes **linear temporal resampling** to standardize the frame rate and length of different action sequences, ensuring a uniform input representation for the feature engineering and modeling stages.
*   **Missing Data Integrity:** Includes functionality to **interpolate missing data points**, crucial for maintaining continuous motion trajectories and preventing gaps in the derived features.

### 📐 Advanced Biomechanical Feature Extraction

*   **Precise Joint Kinematics:** Automatically calculates detailed kinematic features, including **joint angles** using a precise three-point calculation method (`angle_3pts`), enabling the quantification of limb stress and range of motion.
*   **Trunk Stability Measurement:** Determines the **trunk inclination angle** by measuring the angle between the hip-to-head vector and the vertical axis. This feature is a critical proxy for core stability and postural risk during dynamic actions.
*   **Final Feature Aggregation:** Compiles all processed, normalized, and engineered data into a single, standardized CSV output (`dataset.csv`), ready for immediate use in training and validation.

---

## 🛠️ Tech Stack & Architecture

The project is built entirely on the Python scientific computing ecosystem, ensuring high performance, stability, and extensive compatibility with standard machine learning workflows.

| Technology | Purpose | Why it was Chosen |
| :--- | :--- | :--- |
| **Python** | Primary development language for scripting and ML logic. | Simplicity, vast library ecosystem, and dominance in data science. |
| **scikit-learn** | Machine learning framework for modeling (LR, DT) and evaluation. | Provides mature, high quality implementations of core algorithms like Logistic Regression and Decision Trees. |
| **NumPy** | Fundamental library for efficient numerical and array operations. | Essential for handling large, multi dimensional pose data matrices and kinematic calculations. |
| **Pandas** | Data structure and analysis tool for data preparation and handling. | Used extensively for managing the feature matrix, labeling, and pipeline orchestration. |
| **SciPy** | Scientific computing library for advanced mathematical functions. |
| **Joblib** | Serialization and pipelining utility. | Used for efficiently saving and loading trained models and complex feature processing objects. |
| **Matplotlib** | Data visualization tool. | Enables the generation of all evaluation plots, including confusion matrices and model comparison charts. |

---

## 📁 Project Structure

The project is organized into modular components for the data ingestion/feature engineering pipeline (`data_pipeline`) and the machine learning model suite (`models`), facilitating clear separation of concerns and maintainability.

```
📂 gauravag18-Injury-Prediction-In-Athelete-2297337/
├── 📄 .gitignore                 # Specifies files to ignore by Git (e.g., outputs)
├── 📄 requirements.txt           # Lists all required Python packages (dependencies)
├── 📄 README.md                  # This project documentation file
├── 📄 LICENSE                    # Project license (MIT)
├── 📂 src/                       # Source code for the processing and modeling logic
│   ├── 📄 run_pipeline.py        # Central script likely orchestrating the entire data pipeline
│   ├── 📂 data_pipeline/         # Scripts for data ingestion, cleaning, and feature engineering
│   │   ├── 📄 01_find_actions.py      # Identifies and segments actions within raw data
│   │   ├── 📄 02_extract_pose.py      # Extracts raw pose points and interpolates missing data
│   │   ├── 📄 03_clean_normalize_pose.py # Smoothing, centering, scaling, and temporal resampling logic
│   │   ├── 📄 04_joint_angles.py      # Computes core biomechanical features (joint and trunk angles)
│   │   ├── 📄 05_feature_creation.py  # Advanced feature creation module
│   │   ├── 📄 06_risk_label.py        # Module for calculating and assigning injury risk percentages (pct)
│   │   └── 📄 07_create_csv.py        # Final step: aggregates processed data into dataset.csv
│   └── 📂 models/                # Machine learning training, prediction, and evaluation modules
│       ├── 📄 decision_tree.py       # Implementation of the Decision Tree Classifier
│       ├── 📄 logistic_regression.py # Implementation of the Logistic Regression Classifier
│       ├── 📄 rule_augment_lr.py     # Hybrid modeling approach (Rule-Augmented Logistic Regression)
│       ├── 📄 evaluate.py            # Script for calculating performance metrics across models
│       ├── 📄 predict.py             # Function for single-sample injury risk prediction (predict_risk)
│       ├── 📄 plot.py                # Utilities for generating and saving visualization plots (save_confusion)
│       └── 📄 chart.py               # Chart generation helper utilities
├── 📂 outputs/                   # Directory for output artifacts
│   └── 📂 csv/                    # Location for processed datasets
│       └── 📄 dataset.csv          # The final, feature-engineered dataset
├── 📂 data/                      # Input data directory
│   └── 📂 penn_action/           # Contains raw Penn Action dataset labels
│       ├── 📄 README                 # Data source description
│       └── 📂 labels/              # Hundreds of raw .mat files containing kinematic labels
│           ├── 📄 1679.mat
│           └── ... (700+ other .mat files)
└── 📂 docs/                      # Documentation and visualization assets
    └── 📂 figures/               # Saved plots for model evaluation and analysis
        ├── 📄 confusion_dt.png       # Confusion matrix for Decision Tree model
        ├── 📄 confusion_hybrid.png   # Confusion matrix for Hybrid (Rule-Augment) model
        ├── 📄 confusion_lr.png       # Confusion matrix for Logistic Regression model
        ├── 📄 model_comparison.png   # Comparative performance of models
        ├── 📄 recall_per_class.png   # Recall distribution analysis
        └── 📄 risk_score_distribution.png # Distribution of assigned risk scores
```

---

## 🚀 Getting Started

To utilize the Injury Prediction In Athlete framework, you must set up the necessary Python environment and install the required scientific computing libraries.

### Prerequisites

This project relies exclusively on the Python ecosystem. Ensure you have:

*   **Python:** Version 3.8+ recommended.
*   **Package Manager:** `pip`

### Installation

Follow these steps to clone the repository and install all dependencies:

#### 1. Clone the Repository

```bash
# Clone the project repository
git clone https://github.com/gauravag18/Injury-Prediction-In-Athelete-2297337.git
cd gauravag18-Injury-Prediction-In-Athelete-2297337
```

#### 2. Install Dependencies

All required dependencies are specified in the `requirements.txt` file. Use `pip` to install them:

```bash
pip install -r requirements.txt
```

This command will install all necessary packages, including `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib`, and `matplotlib`.

#### 3. Data Preparation

The project expects the raw kinematic data in the standard `.mat` format within the structure defined in `data/penn_action/labels/`. Ensure that the raw data labels are correctly placed in the `data/penn_action/labels/` directory before proceeding to the Usage section.

---

## 🔧 Usage

The primary usage of this project involves executing the sequential steps of the data processing pipeline to generate a rich feature set, followed by training and evaluating the predictive models.

### 1. Data Processing and Feature Engineering

The `src/data_pipeline/` directory contains the modular scripts necessary to transform raw `.mat` files into the final predictive dataset (`outputs/csv/dataset.csv`). These modules must be run in sequence:

| Step | Script | Core Functionality & Output |
| :--- | :--- | :--- |
| 1 | `02_extract_pose.py` | Extracts pose data and performs initial missing data interpolation. |
| 2 | `03_clean_normalize_pose.py` | Applies `smooth_pose` (Savitzky–Golay smoothing), `center_and_scale` normalization, and `temporal_resample`. |
| 3 | `04_joint_angles.py` | Calculates key features: `angle_3pts` for joint kinematics and `trunk_angle` for stability assessment. |
| 4 | `05_feature_creation.py` | Creates the full feature set used for modeling. |
| 5 | `06_risk_label.py` | Computes the risk target variable using metrics like `pct`. |
| 6 | `07_create_csv.py` | Writes the final feature matrix and risk labels to `outputs/csv/dataset.csv`. |

While a specific overall entry point was not automatically detected, the structured file naming strongly implies a sequential execution is required. The `run_pipeline.py` script is architecturally positioned to manage this process.

### 2. Training and Evaluation

Once `dataset.csv` is generated, the models can be trained and evaluated using the scripts in `src/models/`.

1.  **Model Training:** Execute the training scripts for the individual models:
    *   Run `decision_tree.py`
    *   Run `logistic_regression.py`
    *   Run `rule_augment_lr.py` (Hybrid Model)

2.  **Model Evaluation and Visualization:** Utilize the evaluation modules to assess performance:
    *   Run `evaluate.py` to calculate key metrics and cross-validation scores.
    *   Run `plot.py` (which uses `save_confusion`) and `chart.py` to generate the necessary performance graphics, including:
        *   Confusion Matrices for all three models.
        *   Model performance comparison charts.
        *   Risk score distribution visualizations.
    
    All generated performance figures will be saved to the `docs/figures/` directory.

### 3. Predicting Injury Risk

The project provides a dedicated function for integration into real-time or production environments:

The function `predict_risk` within the `src/models/predict.py` module is used to obtain an immediate risk prediction for an unseen sample.

```python
from src.models.predict import predict_risk
import numpy as np

# Example: Assuming 'feature_vector' is a prepared array 
# containing the necessary biomechanical features for one athletic action.
feature_vector = np.array([...]) 

# Use the loaded model to predict the risk label
risk_prediction = predict_risk(feature_vector)

print(f"Predicted Injury Risk Label: {risk_prediction}")
```

The prediction function expects the feature vector to be pre-processed and normalized identically to the training data generated by the `data_pipeline`.

---

## 🤝 Contributing

We welcome contributions to improve the Injury Prediction In Athlete framework! Your input helps make this project better for athletes, trainers, and researchers worldwide.

### How to Contribute

1. **Fork the repository** - Click the 'Fork' button at the top right of this page
2. **Create a feature branch** 
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes** - Improve code, documentation, or features, particularly enhancing the feature engineering pipeline or adding new modeling techniques.
4. **Test thoroughly** - Ensure all functionality works as expected. Given the analytical nature of the project, verify that feature outputs and model evaluations remain stable.
5. **Commit your changes** - Write clear, descriptive commit messages
   ```bash
   git commit -m 'Feat: Added new elbow angle feature computation in 04_joint_angles'
   ```
6. **Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request** - Submit your changes for review

### Development Guidelines

- Follow the existing code style and conventions, especially for the Python data pipeline files
- 📝 Add comments for complex logic and biomechanical calculation algorithms.
- 🧪 If adding new models, ensure thorough evaluation using `evaluate.py` standards.
- 📚 Update documentation for any changed functionality or new feature descriptions.
- 🔄 Ensure backward compatibility when modifying core pipeline functions (e.g., `smooth_pose` or `center_and_scale`).
- 🎯 Keep commits focused and atomic.


### Questions?

Feel free to open an issue for any questions or concerns regarding the project's methodology, architecture, or contributions. We're here to help!

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for complete details.
 
---