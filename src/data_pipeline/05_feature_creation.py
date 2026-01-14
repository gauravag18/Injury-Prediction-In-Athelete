import os
import numpy as np

ANGLE_DIR = "../../outputs/angles"
POSE_DIR  = "../../outputs/poses_clean"
OUT_DIR   = "../../outputs/features"
os.makedirs(OUT_DIR, exist_ok=True)

# Pose joint indices
HEAD = 0
L_HIP = 1
R_HIP = 2
L_KNEE = 3
R_KNEE = 4
L_ANKLE = 5
R_ANKLE = 6

X = []
file_names = []

for file in os.listdir(ANGLE_DIR):
    angles = np.load(os.path.join(ANGLE_DIR, file))   # (100, 5)
    pose   = np.load(os.path.join(POSE_DIR, file))    # (100, 7, 2)

    features = []
    
    # 1. BASIC STATISTICS (5 angles × 5 stats = 25)
    for i in range(angles.shape[1]):
        a = angles[:, i]
        features.extend([
            a.mean(),
            a.std(),
            a.min(),
            a.max(),
            a.max() - a.min()
        ])

    # 2. ASYMMETRY FEATURES (KNEE + HIP)
    knee_diff = np.abs(angles[:, 0] - angles[:, 1])
    hip_diff  = np.abs(angles[:, 2] - angles[:, 3])

    features.extend([
        knee_diff.mean(),
        knee_diff.max(),
        knee_diff.std(),
        hip_diff.mean(),
        hip_diff.max(),
        hip_diff.std()
    ])

    # 3. MOVEMENT DEPTH / JUMP HEIGHT (POSE-BASED)
    hip_y = (pose[:, L_HIP, 1] + pose[:, R_HIP, 1]) / 2
    ankle_y = (pose[:, L_ANKLE, 1] + pose[:, R_ANKLE, 1]) / 2

    depth_or_jump = ankle_y.max() - hip_y.min()
    features.append(depth_or_jump)

    # 4. TEMPORAL RISK EXPOSURE
    risky_knee_ratio = np.mean(
        (angles[:, 0] > 140) | (angles[:, 1] > 140)
    )

    risky_trunk_ratio = np.mean(
        angles[:, 4] > 30
    )

    features.extend([
        risky_knee_ratio,
        risky_trunk_ratio
    ])

    # 5. MOVEMENT QUALITY (CONTROL)
    knee_vel = np.diff(angles[:, 0])
    knee_acc = np.diff(knee_vel)

    mean_knee_vel = np.mean(np.abs(knee_vel))
    knee_jerk_energy = np.mean(knee_acc ** 2)

    features.extend([
        mean_knee_vel,
        knee_jerk_energy
    ])

    # 6. COORDINATION (LEFT & RIGHT SEPARATE)
    hip_knee_corr_L = np.corrcoef(
        angles[:, 2], angles[:, 0]
    )[0, 1]

    hip_knee_corr_R = np.corrcoef(
        angles[:, 3], angles[:, 1]
    )[0, 1]

    coord_asym = np.abs(hip_knee_corr_L - hip_knee_corr_R)

    features.extend([
        hip_knee_corr_L,
        hip_knee_corr_R,
        coord_asym
    ])

    X.append(features)
    file_names.append(file)

X = np.array(X)

np.save(os.path.join(OUT_DIR, "X.npy"), X)
np.save(os.path.join(OUT_DIR, "files.npy"), np.array(file_names))

print("Feature extraction complete")
print("Feature matrix shape:", X.shape)
