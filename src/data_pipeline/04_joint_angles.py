import os
import numpy as np

IN_DIR = "../../outputs/poses_clean"
OUT_DIR = "../../outputs/angles"
os.makedirs(OUT_DIR, exist_ok=True)

# Joint indices
HEAD = 0
L_HIP = 1
R_HIP = 2
L_KNEE = 3
R_KNEE = 4
L_ANKLE = 5
R_ANKLE = 6


def angle_3pts(a, b, c):
    """
    Compute angle at point b given points a-b-c
    """
    ba = a - b
    bc = c - b

    cosang = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


def trunk_angle(hip_center, head):
    """
    Trunk inclination angle:
    angle between (hip -> head) vector and vertical axis
    """
    trunk_vec = head - hip_center

    vertical = np.array([0, -1])  # upward direction

    cosang = np.dot(trunk_vec, vertical) / (
        np.linalg.norm(trunk_vec) * np.linalg.norm(vertical) + 1e-6
    )

    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))



for file in os.listdir(IN_DIR):
    pose = np.load(os.path.join(IN_DIR, file))
    frames = pose.shape[0]

    angles = []

    for f in range(frames):
        p = pose[f]

        # Knee angles
        knee_L = angle_3pts(p[L_HIP], p[L_KNEE], p[L_ANKLE])
        knee_R = angle_3pts(p[R_HIP], p[R_KNEE], p[R_ANKLE])

        # Hip angles
        hip_L = angle_3pts(p[HEAD], p[L_HIP], p[L_KNEE])
        hip_R = angle_3pts(p[HEAD], p[R_HIP], p[R_KNEE])

        # Trunk angle (average hip as base)
        hip_center = (p[L_HIP] + p[R_HIP]) / 2
        trunk = trunk_angle(hip_center, p[HEAD])

        angles.append([
            knee_L, knee_R,
            hip_L, hip_R,
            trunk
        ])

    angles = np.array(angles)  # (frames, 5)

    np.save(os.path.join(OUT_DIR, file), angles)

print("Joint angles computed per frame")
