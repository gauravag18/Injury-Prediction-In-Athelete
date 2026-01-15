import os
import numpy as np
from scipy.signal import savgol_filter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
IN_DIR = os.path.join(PROJECT_ROOT, "outputs", "poses_raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "poses_clean")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_FRAMES = 100

# joint indices after extraction
HEAD = 0
L_HIP = 1
R_HIP = 2
L_KNEE = 3
R_KNEE = 4
L_ANKLE = 5
R_ANKLE = 6


def smooth_pose(pose):
    """
    Savitzky–Golay smoothing over time
    pose: (frames, joints, 2)
    """
    smoothed = np.zeros_like(pose)
    for j in range(pose.shape[1]):
        for c in range(2):
            smoothed[:, j, c] = savgol_filter(
                pose[:, j, c],
                window_length=7,
                polyorder=2,
                mode="nearest"
            )
    return smoothed


def center_and_scale(pose):
    """
    Center at hip midpoint and scale using average body segment length
    pose: (frames, joints, 2)
    """
    # Hip center
    hip_center = (pose[:, L_HIP] + pose[:, R_HIP]) / 2
    pose = pose - hip_center[:, None, :]
    # Body segments
    leg = np.linalg.norm(pose[:, L_HIP] - pose[:, L_ANKLE], axis=1)
    thigh = np.linalg.norm(pose[:, L_HIP] - pose[:, L_KNEE], axis=1)
    trunk = np.linalg.norm(pose[:, L_HIP] - pose[:, HEAD], axis=1)

    # Robust scale (ignore zeros)
    scale = np.mean([leg.mean(), thigh.mean(), trunk.mean()])

    pose = pose / (scale + 1e-6)
    return pose

def temporal_resample(pose, target_len):
    """
    Linear temporal resampling
    """
    frames = pose.shape[0]
    old_idx = np.linspace(0, 1, frames)
    new_idx = np.linspace(0, 1, target_len)

    resampled = np.zeros((target_len, pose.shape[1], 2))
    for j in range(pose.shape[1]):
        for c in range(2):
            resampled[:, j, c] = np.interp(
                new_idx, old_idx, pose[:, j, c]
            )
    return resampled


for file in os.listdir(IN_DIR):
    pose = np.load(os.path.join(IN_DIR, file))

    pose = smooth_pose(pose)
    pose = center_and_scale(pose)
    pose = temporal_resample(pose, TARGET_FRAMES)

    np.save(os.path.join(OUT_DIR, file), pose)

print("Pose cleaning & normalization complete")
