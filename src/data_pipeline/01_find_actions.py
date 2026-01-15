import os
import scipy.io as sio
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "penn_action", "labels")

actions = []

for file in os.listdir(LABEL_DIR):
    if file.endswith(".mat"):
        mat = sio.loadmat(os.path.join(LABEL_DIR, file))
        action = mat["action"][0]
        actions.append(action)

from collections import Counter
action_counts = Counter(actions)

print("Actions found:\n")
for act, cnt in action_counts.items():
    print(f"{act:15s} -> {cnt}")
