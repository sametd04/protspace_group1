"""Bar chart of kNN classification accuracy results from knn_accuracy_3ftx.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

csv_path = Path("src/protspace/benchmark/results/knn_accuracy_3ftx.csv")
df = pd.read_csv(csv_path)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(df["method"], df["knn_accuracy"], color=["#4C72B0", "#55A868", "#C44E52"])
ax.set_ylabel("kNN Accuracy (LOO CV, k=15)")
ax.set_ylim(0, 1.05)
ax.set_title("kNN Classification Accuracy: pPCA vs UMAP vs PCA")

for bar, val in zip(bars, df["knn_accuracy"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{val:.3f}",
            ha="center", va="bottom", fontsize=11)

plt.tight_layout()
out = Path("src/protspace/benchmark/results/knn_accuracy_3ftx.png")
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
