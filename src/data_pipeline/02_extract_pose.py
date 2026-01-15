import os
import numpy as np
import scipy.io as sio

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "penn_action", "labels")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "poses_raw")
os.makedirs(OUT_DIR, exist_ok=True)

IMG_W, IMG_H = 640.0, 480.0

TARGET_ACTIONS = ["jumping_jacks", "squat"]

JOINT_IDS = {
    "head": 0,        # head / neck proxy
    "left_hip": 7,
    "right_hip": 8,
    "left_knee": 9,
    "right_knee": 10,
    "left_ankle": 11,
    "right_ankle": 12
}

def interpolate_missing(arr, vis):
    frames = np.arange(len(arr))
    valid = vis == 1

    if valid.sum() < 2:
        return np.zeros_like(arr)

    return np.interp(frames, frames[valid], arr[valid])

for file in os.listdir(LABEL_DIR):
    if not file.endswith(".mat"):
        continue

    mat = sio.loadmat(os.path.join(LABEL_DIR, file))

    # 🔹 FILTER BY ACTION
    action = mat["action"][0]
    if action not in TARGET_ACTIONS:
        continue

    if not all(k in mat for k in ["x", "y", "visibility"]):
        continue

    x = mat["x"]          # (13, frames)
    y = mat["y"]
    vis = mat["visibility"]

    pose_seq = []

    for j_idx in JOINT_IDS.values():
        xj = x[j_idx]
        yj = y[j_idx]
        vj = vis[j_idx]

        xj = interpolate_missing(xj, vj)
        yj = interpolate_missing(yj, vj)

        xj /= IMG_W
        yj /= IMG_H

        pose_seq.append(np.stack([xj, yj], axis=1))

    pose_seq = np.stack(pose_seq, axis=1)  # (frames, 7, 2)

    np.save(
        os.path.join(OUT_DIR, file.replace(".mat", ".npy")),
        pose_seq
    )

print("Pose extraction done for jumping_jacks & squat only")
