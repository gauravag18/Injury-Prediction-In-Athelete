import os
import scipy.io as sio
from collections import Counter

LABEL_DIR = "../../data/penn_action/labels"

actions = []

for file in os.listdir(LABEL_DIR):
    if file.endswith(".mat"):
        mat = sio.loadmat(os.path.join(LABEL_DIR, file))
        action = mat["action"][0]
        actions.append(action)

action_counts = Counter(actions)

print("Actions found:\n")
for act, cnt in action_counts.items():
    print(f"{act:15s} -> {cnt}")
