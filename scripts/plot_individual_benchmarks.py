"""Generate 3 individual slide-ready benchmark plots from bootstrapped results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("src/protspace/benchmark/results")
CSV_PATH = RESULTS_DIR / "bootstrapped_metrics_3ftx.csv"

METHODS = ["PCA", "UMAP", "pPCA-ToxProt"]
COLORS = ["#4C72B0", "#55A868", "#C44E52"]


def compute_ci(values, ci=0.95):
    arr = np.array(values)
    mean = arr.mean()
    lower = np.percentile(arr, (1 - ci) / 2 * 100)
    upper = np.percentile(arr, (1 + ci) / 2 * 100)
    return mean, lower, upper


def plot_single_metric(df, metric_col, ylabel, title, filename, ylim=(0, 1.05)):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    means, ci_lows, ci_highs = [], [], []
    for method in METHODS:
        vals = df[df["method"] == method][metric_col].values
        mean, low, high = compute_ci(vals)
        means.append(mean)
        ci_lows.append(mean - low)
        ci_highs.append(high - mean)

    bars = ax.bar(METHODS, means, color=COLORS, edgecolor="none",
                  width=0.5)
    ax.errorbar(METHODS, means, yerr=[ci_lows, ci_highs],
                fmt="none", capsize=7, capthick=1.2, ecolor="#333333", linewidth=1.2)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=13, pad=10)
    ax.tick_params(axis="both", labelsize=11)

    for bar, val in zip(bars, means):
        y_offset = 0.02 if val >= 0 else -0.04
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                f"{val:.3f}", ha="center", va=va, fontsize=12, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = RESULTS_DIR / filename
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_bar_negative(df, metric_col, ylabel, title, filename, ylim=(-0.35, 0.15)):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    means, lows, highs = [], [], []
    for method in METHODS:
        vals = df[df["method"] == method][metric_col].values
        mean, low, high = compute_ci(vals)
        means.append(mean)
        lows.append(low)
        highs.append(high)

    ci_lows = [m - l for m, l in zip(means, lows)]
    ci_highs = [h - m for m, h in zip(means, highs)]

    bars = ax.bar(METHODS, means, color=COLORS, edgecolor="none", width=0.5)
    ax.errorbar(METHODS, means, yerr=[ci_lows, ci_highs],
                fmt="none", capsize=7, capthick=1.2, ecolor="#333333", linewidth=1.2)

    for bar, val, low in zip(bars, means, lows):
        ax.text(bar.get_x() + bar.get_width() / 2, low - 0.015,
                f"{val:.3f}", ha="center", va="top", fontsize=12, fontweight="bold")

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=13, pad=10)
    ax.tick_params(axis="both", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = RESULTS_DIR / filename
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    df = pd.read_csv(CSV_PATH)

    plot_single_metric(
        df, "trust_k15",
        ylabel="Trustworthiness",
        title="Trustworthiness (k=15)",
        filename="slide_trustworthiness.png",
    )

    plot_single_metric(
        df, "knn_acc",
        ylabel="kNN Accuracy",
        title="kNN Classification Accuracy (k=15)",
        filename="slide_knn_accuracy.png",
    )

    plot_bar_negative(
        df, "silhouette",
        ylabel="Silhouette Score",
        title="Silhouette Score",
        filename="slide_silhouette.png",
    )


if __name__ == "__main__":
    main()
