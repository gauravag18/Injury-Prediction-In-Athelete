import numpy as np
import os
from scipy.stats import rankdata

FEATURE_DIR = "../../outputs/features"
OUT_DIR     = "../../outputs/features"

X     = np.load(os.path.join(FEATURE_DIR, "X.npy"))
files = np.load(os.path.join(FEATURE_DIR, "files.npy"))
N     = len(files)


LOW_CUTOFF  = 0.30   # bottom 30% -> label 0
HIGH_CUTOFF = 0.70   # 30–70% -> label 1, top 30% -> label 2

# Percentile normalization for each signal
def pct(x):
    return rankdata(x, method="average") / (len(x) + 1.0)  # in (0,1)

# Extract risk signals 
knee_max   = X[:, 3]      # max knee_L
trunk_max  = X[:, 23]     # max trunk
knee_asym  = X[:, 25]     # mean knee asym
knee_jerk  = X[:, -6]     # knee jerk energy
depth      = X[:, 31]     # squat depth / jump height

# Normalize each signal (0–1, higher = worse)
knee_risk    = pct(knee_max)
trunk_risk   = pct(trunk_max)
asym_risk    = pct(knee_asym)
control_risk = pct(knee_jerk)
depth_risk   = pct(depth)

# Composite risk score (0–1)
risk_score = (
    0.35 * knee_risk +
    0.20 * trunk_risk +
    0.20 * asym_risk +
    0.15 * control_risk +
    0.10 * depth_risk
)

# Convert composite score itself to percentile
risk_rank = rankdata(risk_score, method="average")
risk_pct  = risk_rank / (N + 1.0)   # in (0,1)

# Label assignment 
y = np.zeros(N, dtype=int)
y[(risk_pct > LOW_CUTOFF) & (risk_pct <= HIGH_CUTOFF)] = 1
y[risk_pct > HIGH_CUTOFF] = 2


np.save(os.path.join(OUT_DIR, "y.npy"), y)

print("Relative risk labeling complete")
u, c = np.unique(y, return_counts=True)
for ui, ci in zip(u, c):
    print(f"Risk {ui}: {ci}")
