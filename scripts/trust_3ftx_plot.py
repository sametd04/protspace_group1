"""Bar chart of trustworthiness results from trust_3ftx.py output."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

csv_path = Path("src/protspace/benchmark/results/trustworthiness_3ftx.csv")
df = pd.read_csv(csv_path)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(df["method"], df["trustworthiness"], color=["#4C72B0", "#55A868", "#C44E52"])
ax.set_ylabel("Trustworthiness (k=15)")
ax.set_ylim(0, 1.05)
ax.set_title("Trustworthiness: pPCA vs UMAP vs PCA")

for bar, val in zip(bars, df["trustworthiness"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{val:.3f}",
            ha="center", va="bottom", fontsize=11)

plt.tight_layout()
out = Path("src/protspace/benchmark/results/trustworthiness_3ftx.png")
plt.savefig(out, dpi=150)
print(f"Saved to {out}")
