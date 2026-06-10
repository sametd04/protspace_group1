"""Visualizations for the CATH hierarchy preservation evaluation.

Produces two figures:
1. **Bar chart** — KHP at each CATH level per DR method (grouped bars) with
   horizontal dashed lines showing the random baseline for each level.
2. **Line chart** — Purity curve from Homology → Class (one line per method),
   showing how purity degrades as the hierarchy level becomes coarser, plus a
   dashed "random baseline" series for reference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Ordered finest → coarsest
_LEVEL_COLS = ["khp_homology", "khp_topology", "khp_architecture", "khp_class"]
_BASELINE_COLS = [
    "baseline_homology",
    "baseline_topology",
    "baseline_architecture",
    "baseline_class",
]
_LEVEL_NAMES = ["Homology (H)", "Topology (T)", "Architecture (A)", "Class (C)"]


def plot_khp_results(
    df: pd.DataFrame,
    output_dir: Path,
    k: int = 5,
) -> None:
    """Save bar chart and purity-curve figures.

    Parameters
    ----------
    df:
        Output of :func:`~protspace.benchmark.cath_hierarchy.run.run_cath_hierarchy_evaluation`.
        Must contain ``khp_*`` and ``baseline_*`` columns.
    output_dir:
        Directory where figures are written (``khp_bar.png``, ``khp_curve.png``).
    k:
        Value of k used — shown in figure titles.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed — skipping plots.")
        return

    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = df["method"].tolist()
    n_methods = len(methods)
    n_levels = len(_LEVEL_COLS)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Extract baselines from first row (same for all methods)
    baselines = [float(df[col].iloc[0]) for col in _BASELINE_COLS]

    # ------------------------------------------------------------------
    # Figure 1 — grouped bar chart with baseline reference lines
    # ------------------------------------------------------------------
    x = np.arange(n_levels)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (_, row) in enumerate(df.iterrows()):
        values = [float(row.get(col, float("nan"))) for col in _LEVEL_COLS]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=row["method"],
            color=colors[i % len(colors)],
            zorder=3,
        )
        for bar, val in zip(bars, values, strict=False):
            if val == val:  # not NaN
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )

    # Horizontal dashed lines for each level's random baseline
    for xi, baseline in enumerate(baselines):
        label = "random baseline" if xi == 0 else None
        ax.hlines(
            baseline,
            xi - 0.45,
            xi + 0.45,
            colors="black",
            linestyles="dashed",
            linewidths=1.2,
            zorder=4,
            label=label,
        )
        ax.text(
            xi + 0.47,
            baseline,
            f"{baseline:.3f}",
            va="center",
            fontsize=7,
            color="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(_LEVEL_NAMES)
    ax.set_ylabel(f"k-NN Purity (k={k})")
    ax.set_title(f"CATH Hierarchy Preservation (k={k})\nDashed lines = random baseline")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()

    bar_path = output_dir / "khp_bar.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    logger.info("Saved bar chart to %s", bar_path)

    # ------------------------------------------------------------------
    # Figure 2 — purity-curve with dashed random-baseline series
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (_, row) in enumerate(df.iterrows()):
        values = [float(row.get(col, float("nan"))) for col in _LEVEL_COLS]
        ax.plot(
            _LEVEL_NAMES,
            values,
            marker="o",
            label=row["method"],
            color=colors[i % len(colors)],
        )

    # Single dashed "random baseline" line
    ax.plot(
        _LEVEL_NAMES,
        baselines,
        marker="x",
        linestyle="dashed",
        color="black",
        linewidth=1.2,
        label="random baseline",
    )

    ax.set_ylabel(f"k-NN Purity (k={k})")
    ax.set_title(
        f"Purity Degradation Across CATH Hierarchy Levels (k={k})\n"
        "Dashed line = random baseline"
    )
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    curve_path = output_dir / "khp_curve.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    logger.info("Saved purity curve to %s", curve_path)
