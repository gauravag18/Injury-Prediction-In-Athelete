import numpy as np
import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FEATURE_DIR = os.path.join(PROJECT_ROOT, "outputs", "features")
OUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "csv", "dataset.csv")

X = np.load(os.path.join(FEATURE_DIR, "X.npy"))
y = np.load(os.path.join(FEATURE_DIR, "y.npy"))
files = np.load(os.path.join(FEATURE_DIR, "files.npy"))

feature_cols = [f"f{i+1}" for i in range(X.shape[1])]

df = pd.DataFrame(X, columns=feature_cols)
df["label"] = y
df["clip"] = files

df.to_csv(OUT_PATH, index=False)

print("Final dataset CSV created")
print("Shape:", df.shape)
