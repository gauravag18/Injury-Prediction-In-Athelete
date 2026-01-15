import os

scripts = [
    "01_find_actions.py",
    "02_extract_pose.py",
    "03_clean_normalize_pose.py",
    "04_joint_angles.py",
    "05_feature_creation.py",
    "06_risk_label.py",
    "07_create_csv.py"
]

for script in scripts:
    print(f"Running {script}...")
    os.system(f"python src/data_pipeline/{script}")
