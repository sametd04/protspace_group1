"""Visualizations for the CATH hierarchy preservation evaluation.

Produces two figures:

1. **Bar chart** — mean KHP ± std at each CATH level for every method
   (grouped bars with error bars), plus horizontal dashed lines for the
   random baseline and a distinct bar for the Original Embedding.

2. **Purity-curve** — mean KHP from Homology → Class for every method
   (shaded ±1 std band), plus random-baseline and Original Embedding lines.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Finest → coarsest (x-axis order)
_LEVEL_KEYS = ["homology", "topology", "architecture", "cath_class"]
_LEVEL_NAMES = ["Homology (H)", "Topology (T)", "Architecture (A)", "Class (C)"]

_REFERENCE_METHODS = {"Random Baseline", "Original Embedding"}
_REF_COLOURS = {
    "Original Embedding": "#444444",
    "Random Baseline": "#000000",
}
_REF_CURVE_STYLES = {
    "Original Embedding": {
        "linestyle": "--",
        "linewidth": 1.6,
        "marker": "s",
        "markersize": 5,
    },
    "Random Baseline": {
        "linestyle": ":",
        "linewidth": 1.4,
        "marker": "x",
        "markersize": 5,
    },
}


def plot_khp_results(
    df: pd.DataFrame,
    output_dir: Path,
    k_per_level: dict[str, int] | None = None,
) -> None:
    """Save bar chart and purity-curve figures.

    Parameters
    ----------
    df:
        Output of :func:`~protspace.benchmark.cath_hierarchy.run.run_cath_hierarchy_evaluation`.
        Must contain ``khp_{level}_mean``, ``khp_{level}_std`` and
        ``baseline_{level}`` columns.
    output_dir:
        Directory for output PNGs.
    k_per_level:
        Adaptive k values per level (used in figure titles).
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed — skipping plots.")
        return

    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dr_df = df[~df["method"].isin(_REFERENCE_METHODS)].reset_index(drop=True)
    ref_rows = {
        name: df[df["method"] == name].iloc[0]
        for name in _REFERENCE_METHODS
        if name in df["method"].values
    }

    if k_per_level:
        k_str = "  |  ".join(
            f"{ln}: k={k_per_level.get(lk, '?')}"
            for ln, lk in zip(_LEVEL_NAMES, _LEVEL_KEYS, strict=False)
        )
    else:
        k_str = "adaptive k"

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    dr_colors = [prop_cycle[i % len(prop_cycle)] for i in range(len(dr_df))]

    # ------------------------------------------------------------------
    # Figure 1 — Grouped bar chart with error bars
    # ------------------------------------------------------------------
    n_levels = len(_LEVEL_KEYS)
    x = np.arange(n_levels)

    has_orig = "Original Embedding" in ref_rows
    n_bars = len(dr_df) + (1 if has_orig else 0)
    width = min(0.8 / max(n_bars, 1), 0.13)

    # Collect (label, means, stds, color, extras) for each bar group
    bar_groups: list[tuple[str, list[float], list[float], str, dict]] = []
    for i, (_, row) in enumerate(dr_df.iterrows()):
        bar_groups.append(
            (
                str(row["method"]),
                [float(row[f"khp_{lk}_mean"]) for lk in _LEVEL_KEYS],
                [float(row[f"khp_{lk}_std"]) for lk in _LEVEL_KEYS],
                dr_colors[i],
                {},
            )
        )
    if has_orig:
        orig = ref_rows["Original Embedding"]
        bar_groups.append(
            (
                "Original Embedding",
                [float(orig[f"khp_{lk}_mean"]) for lk in _LEVEL_KEYS],
                [0.0] * n_levels,
                _REF_COLOURS["Original Embedding"],
                {"hatch": "//", "edgecolor": "white"},
            )
        )

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (label, means, stds, color, extras) in enumerate(bar_groups):
        offset = (i - n_bars / 2 + 0.5) * width
        has_err = any(s > 0 for s in stds)
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds if has_err else None,
            label=label,
            color=color,
            capsize=3,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            zorder=3,
            **extras,
        )
        for bar, val, std in zip(bars, means, stds, strict=False):
            if val == val and val > 0.02:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.006,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    rotation=90,
                )

    # Random baseline: dashed horizontal line per level
    if "Random Baseline" in ref_rows:
        brow = ref_rows["Random Baseline"]
        for xi, lk in enumerate(_LEVEL_KEYS):
            bval = float(brow[f"khp_{lk}_mean"])
            ax.hlines(
                bval,
                xi - 0.45,
                xi + 0.45,
                colors=_REF_COLOURS["Random Baseline"],
                linestyles="dashed",
                linewidths=1.2,
                zorder=4,
            )
            ax.text(
                xi + 0.47,
                bval,
                f"{bval:.3f}",
                va="center",
                fontsize=6.5,
                color=_REF_COLOURS["Random Baseline"],
            )

    baseline_patch = mpatches.Patch(
        facecolor=_REF_COLOURS["Random Baseline"],
        label="Random Baseline (dashed line)",
    )
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(
        handles + [baseline_patch],
        labs + ["Random Baseline"],
        loc="upper left",
        fontsize=7.5,
        framealpha=0.85,
        ncol=2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(_LEVEL_NAMES, fontsize=9)
    ax.set_ylabel("k-NN Purity (mean ± std)", fontsize=9)
    ax.set_title(
        f"CATH Hierarchy Preservation — k-NN Purity (Adaptive k)\n({k_str})",
        fontsize=9,
    )
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    bar_path = output_dir / "khp_bar.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    logger.info("Saved bar chart to %s", bar_path)

    # ------------------------------------------------------------------
    # Figure 2 — Purity-curve with shaded ±1 std bands
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    xp = np.arange(n_levels)

    for i, (_, row) in enumerate(dr_df.iterrows()):
        means = np.array([float(row[f"khp_{lk}_mean"]) for lk in _LEVEL_KEYS])
        stds = np.array([float(row[f"khp_{lk}_std"]) for lk in _LEVEL_KEYS])
        col = dr_colors[i]
        ax.plot(xp, means, marker="o", color=col, label=str(row["method"]), zorder=3)
        if stds.any():
            ax.fill_between(
                xp, means - stds, means + stds, alpha=0.15, color=col, zorder=2
            )

    for name, row_ref in ref_rows.items():
        means = np.array([float(row_ref[f"khp_{lk}_mean"]) for lk in _LEVEL_KEYS])
        ax.plot(
            xp,
            means,
            color=_REF_COLOURS[name],
            label=name,
            zorder=4,
            **_REF_CURVE_STYLES.get(name, {}),
        )

    ax.set_xticks(xp)
    ax.set_xticklabels(_LEVEL_NAMES, fontsize=9)
    ax.set_ylabel("k-NN Purity (mean ± 1 std)", fontsize=9)
    ax.set_title(
        "Purity Degradation Across CATH Hierarchy Levels (Adaptive k)\n"
        "Shaded regions = ±1 std across seeds",
        fontsize=9,
    )
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    curve_path = output_dir / "khp_curve.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    logger.info("Saved purity curve to %s", curve_path)
