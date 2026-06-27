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
    error_source: str = "seeds",
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
    error_source:
        What the ``khp_{level}_std`` columns represent — ``"seeds"`` (default,
        seed-robustness workflow) or ``"hyperparams"`` (hyperparameter-robustness
        workflow).  Only affects axis labels / titles, not the data drawn.
    """
    err_desc = (
        "hyperparameter configurations" if error_source == "hyperparams" else "seeds"
    )
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
    ax.set_ylabel(f"k-NN Purity (mean ± std across {err_desc})", fontsize=9)
    ax.set_title(
        f"CATH Hierarchy Preservation — k-NN Purity (Adaptive k)\n"
        f"({k_str})  —  error bars: ±1 std across {err_desc}",
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
        f"Shaded regions = ±1 std across {err_desc}",
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


# ---------------------------------------------------------------------------
# Best-vs-default improvement heatmap
# ---------------------------------------------------------------------------

# Display order + labels for the improvement heatmap (rows)
_HEATMAP_METHOD_ORDER = ["pca", "umap", "tsne", "pacmap", "localmap", "mds"]
_HEATMAP_METHOD_LABELS = {
    "pca": "PCA",
    "umap": "UMAP",
    "tsne": "t-SNE",
    "pacmap": "PaCMAP",
    "localmap": "LocalMAP",
    "mds": "MDS",
}
# Column order (coarse → fine)
_HEATMAP_LEVEL_ORDER = ["Class", "Architecture", "Topology", "Homology"]


def plot_improvement_heatmap(
    csv_path: Path,
    output_path: Path | None = None,
    colormap: str = "magma",
) -> Path:
    """Publication-quality heatmap of best-vs-default k-NN improvement.

    Reads ``best_vs_default_knn.csv`` (columns ``method``, ``hierarchy_level``,
    ``improvement``), pivots it into a method × hierarchy-level matrix and
    renders an annotated heatmap.  Also prints summary statistics to stdout.

    Parameters
    ----------
    csv_path:
        Path to ``best_vs_default_knn.csv``.
    output_path:
        PNG output path.  Defaults to ``knn_improvement_heatmap.png`` next to
        the input CSV.
    colormap:
        Perceptually uniform matplotlib colormap (e.g. ``"magma"``, ``"viridis"``).

    Returns
    -------
    Path to the saved PNG.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    csv_path = Path(csv_path)
    output_path = (
        Path(output_path)
        if output_path is not None
        else csv_path.with_name("knn_improvement_heatmap.png")
    )

    df = pd.read_csv(csv_path)

    # Pivot to method × hierarchy-level, then reorder rows/cols for display.
    matrix = df.pivot(index="method", columns="hierarchy_level", values="improvement")
    row_order = [m for m in _HEATMAP_METHOD_ORDER if m in matrix.index]
    matrix = matrix.reindex(index=row_order, columns=_HEATMAP_LEVEL_ORDER)
    matrix.index = [_HEATMAP_METHOD_LABELS.get(m, m) for m in matrix.index]

    # ------------------------------------------------------------------
    # Summary statistics → stdout
    # ------------------------------------------------------------------
    method_means = matrix.mean(axis=1)
    level_means = matrix.mean(axis=0)
    top_method = method_means.idxmax()
    top_level = level_means.idxmax()
    flat_idx = matrix.stack().idxmax()
    gmax_method, gmax_level = flat_idx
    gmax_value = matrix.stack().max()

    print("\n" + "=" * 60)
    print("  Best-vs-Default k-NN Improvement — Summary")
    print("=" * 60)
    print(
        f"  Largest avg improvement (method): {top_method} ({method_means.max():.3f})"
    )
    print(f"  Largest avg improvement (level) : {top_level} ({level_means.max():.3f})")
    print(
        f"  Global max improvement          : {gmax_value:.3f} "
        f"({gmax_method} @ {gmax_level})"
    )
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Heatmap
    # ------------------------------------------------------------------
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        matrix,
        ax=ax,
        cmap=colormap,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 14, "fontweight": "bold"},
        linewidths=1.0,
        linecolor="white",
        square=True,
        cbar_kws={
            "label": "Performance Improvement (Best kNN − Default kNN)",
            "shrink": 0.85,
        },
    )

    # High-contrast annotations: dark text on light cells, light on dark.
    import numpy as np

    vmin, vmax = float(np.nanmin(matrix.values)), float(np.nanmax(matrix.values))
    span = vmax - vmin if vmax > vmin else 1.0
    cmap = plt.get_cmap(colormap)
    flat = matrix.values.flatten()
    for text, val in zip(ax.texts, flat, strict=False):
        if np.isnan(val):
            text.set_text("")
            continue
        r, g, b, _ = cmap((val - vmin) / span)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text.set_color("white" if luminance < 0.5 else "black")

    ax.set_title(
        "Sensitivity of DR Methods to kNN Optimization",
        fontsize=18,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("CATH Hierarchy Level", fontsize=14, labelpad=10)
    ax.set_ylabel("DR Method", fontsize=14, labelpad=10)
    ax.tick_params(axis="x", labelsize=13, rotation=0)
    ax.tick_params(axis="y", labelsize=13, rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(12)
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved improvement heatmap to %s", output_path)
    return output_path
