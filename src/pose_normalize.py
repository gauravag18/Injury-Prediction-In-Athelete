import numpy as np
import matplotlib.pyplot as plt
raw_pose = np.array([
    [0.00, 0.80],   # head
    [-0.05, 0.50],  # left hip
    [0.05, 0.50],   # right hip
    [-0.08, 0.30],  # left knee
    [0.08, 0.30],   # right knee
    [-0.10, 0.05],  # left ankle
    [0.10, 0.05]    # right ankle
])

#HIP CENTERING
def hip_center(pose):
    hip_mid = (pose[1] + pose[2]) / 2.0
    return pose - hip_mid

# SCALE NORMALIZATION
def scale_pose(pose):
    leg = np.linalg.norm(pose[1] - pose[5])
    thigh = np.linalg.norm(pose[1] - pose[3])
    trunk = np.linalg.norm(pose[1] - pose[0])

    scale = np.mean([leg, thigh, trunk])
    return pose / (scale + 1e-6)


hip_pose = hip_center(raw_pose)
scaled_pose = scale_pose(hip_pose)


# PLOTTING FUNCTION
def plot_pose(ax, pose, title):
    connections = [
        (0, 1), (1, 3), (3, 5),  # left side
        (0, 2), (2, 4), (4, 6),  # right side
        (1, 2)                   # hip line
    ]

    for i, j in connections:
        ax.plot(
            [pose[i, 0], pose[j, 0]],
            [pose[i, 1], pose[j, 1]],
            color='black',
            linewidth=2
        )

    ax.scatter(pose[:, 0], pose[:, 1],
               color='blue', s=50, zorder=3)

    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal')

    # 🔥 Force identical centered limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    ax.axis('off')


# CREATE FIGURE
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

plot_pose(axes[0], raw_pose, "(a) Raw Pose")
plot_pose(axes[1], hip_pose, "(b) Hip-Centered")
plot_pose(axes[2], scaled_pose, "(c) Centered + Scaled")

plt.tight_layout()
plt.savefig(
    "pose_normalization_ieee.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("Pose normalization figure saved as pose_normalization_ieee.png")
