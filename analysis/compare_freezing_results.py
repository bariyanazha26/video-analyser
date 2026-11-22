import matplotlib.pyplot as plt
import numpy as np

experiments = [
    "A: Backbone Frozen",
    "B: No Freezing",
    "C: Head Frozen",
    "D: Both Frozen"
]

precision = [78.4, 45.7, 66.1, 70.6]
recall = [65.1, 38.9, 51.7, 78.5]
map50 = [74.8, 45.6, 55.6, 79.6]
map5095 = [43.0, 20.4, 28.5, 43.4]

x = np.arange(len(experiments))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(x - 1.5 * width, precision, width, label="Precision")
ax.bar(x - 0.5 * width, recall, width, label="Recall")
ax.bar(x + 0.5 * width, map50, width, label="mAP50")
ax.bar(x + 1.5 * width, map5095, width, label="mAP50-95")

ax.set_ylabel("Score (%)")
ax.set_title("YOLOv8L Freezing Experiments — Test Set Comparison")
ax.set_xticks(x)
ax.set_xticklabels(experiments)
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig("freezing_experiments_comparison.png", dpi=300)
plt.show()