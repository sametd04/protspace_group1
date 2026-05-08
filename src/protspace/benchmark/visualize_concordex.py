#!/usr/bin/env python3
"""Concordex visualisations for the benchmark presentation.

Reads the per-method projections written by ``run.py``, re-loads the same
labels, and produces four slide-ready figures:

1. ``concordex_by_method.png``    — bar chart of concordex across DR methods
                                     plus the random-permutation null line.
2. ``concordex_per_class.png``    — per-class concordex ratio for the
                                     best-performing method, ordered by
                                     class size.
3. ``concordex_label_matrix.png`` — heatmap of the aggregated label matrix
                                     ``D`` (LxL) for the best method, with
                                     the diagonal highlighted.
4. ``projections_concordex.png``  — 2x3 grid of all DR projections coloured
                                     by label, with concordex annotated.

Each plot is exported standalone so it can be dropped into the slide
placeholders without further editing.

Usage:
    python visualize_concordex.py
    DATA=globin python visualize_concordex.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from protspace.benchmark.concordex import (
    aggregated_label_matrix,
    calculate_concordex_per_class,
    neighborhood_consolidation_matrix,
)
from protspace.benchmark.labels import label_summary, load_labels_from_bundle
from protspace.data.loaders import load_h5

DATA = os.environ.get("DATA", "3ftx")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EMBEDDING_PATH = PROJECT_ROOT / f"output_{DATA}" / "tmp" / "prot_t5.h5"
BUNDLE_PATH = PROJECT_ROOT / f"output_{DATA}" / "data.parquetbundle"
RESULTS_DIR = Path(__file__).parent / "results" / DATA

METHOD_TITLES = {
    "pca": "PCA",
    "umap": "UMAP",
    "tsne": "t-SNE",
    "pacmap": "PaCMAP",
    "mds": "MDS",
    "localmap": "LocalMAP",
}

PRIMARY = "#1A4D5C"
SECONDARY = "#5B9AA8"
ACCENT = "#D4886B"
GREY = "#6B7280"
LIGHT_GREY = "#E5E7EB"

CONCORDEX_N_NEIGHBORS = 15
CONCORDEX_N_PERMUTATIONS = 15
CONCORDEX_RANDOM_STATE = 42


def _short(s: str, n: int = 24) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _load_labels() -> tuple[np.ndarray, list[str]]:
    if not EMBEDDING_PATH.exists():
        sys.exit(f"Embeddings not found at {EMBEDDING_PATH}")
    emb_set = load_h5([EMBEDDING_PATH])
    headers = list(emb_set.headers)
    if BUNDLE_PATH.exists():
        labels = load_labels_from_bundle(BUNDLE_PATH, headers)
        summary = label_summary(labels)
        print(
            f"Labels: {summary['n_labelled']}/{summary['n_total']} "
            f"proteins, {summary['n_classes']} classes"
        )
    else:
        sys.exit(
            f"No bundle at {BUNDLE_PATH}. Concordex visualisations need "
            f"labels."
        )
    return labels, headers


def _load_projections() -> dict[str, np.ndarray]:
    proj_files = sorted(RESULTS_DIR.glob("*_projection.npy"))
    if not proj_files:
        sys.exit(f"No projections in {RESULTS_DIR}. Run: python run.py")
    return {
        f.stem.replace("_projection", ""): np.load(f) for f in proj_files
    }


def _load_metrics_csv() -> pd.DataFrame | None:
    metrics_csv = RESULTS_DIR / "metrics.csv"
    if not metrics_csv.exists():
        return None
    return pd.read_csv(metrics_csv).set_index("method")


def _build_color_map(labels: np.ndarray) -> tuple[dict, dict]:
    classes = (
        pd.Series(labels)
        .dropna()
        .value_counts()
        .index.tolist()
    )
    sizes = pd.Series(labels).value_counts(dropna=True).to_dict()
    cmap = plt.colormaps["tab20"](np.linspace(0, 1, max(len(classes), 1)))
    color_map = {cls: cmap[i] for i, cls in enumerate(classes)}
    return color_map, sizes


# Plot 1
def plot_concordex_by_method(metrics_df: pd.DataFrame) -> Path:
    if "concordex" not in metrics_df.columns:
        sys.exit(
            "metrics.csv has no 'concordex' column. Re-run benchmark with "
            "concordex enabled."
        )
    df = (
        metrics_df["concordex"]
        .dropna()
        .sort_values(ascending=False)
        .rename_axis("method")
        .reset_index()
    )
    df["display"] = df["method"].map(lambda m: METHOD_TITLES.get(m, m.upper()))

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    bars = ax.bar(
        df["display"],
        df["concordex"],
        color=PRIMARY,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    # Random-permutation null reference at 1.0
    ax.axhline(
        1.0,
        color=ACCENT,
        linestyle="--",
        linewidth=1.2,
        zorder=2,
        label="random-permutation null (= 1)",
    )
    for bar, value in zip(bars, df["concordex"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=PRIMARY,
        )

    ax.set_ylabel("concordex   (observed / expected)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylim(0, max(df["concordex"].max() * 1.18, 1.5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.yaxis.grid(True, color=LIGHT_GREY, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title(
        f"Concordex by DR method  —  {DATA}",
        fontsize=12,
        loc="left",
        pad=10,
        color=PRIMARY,
    )

    fig.tight_layout()
    out = RESULTS_DIR / "concordex_by_method.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


# Plot 2
def plot_concordex_per_class(
    projections: dict[str, np.ndarray],
    labels: np.ndarray,
    metrics_df: pd.DataFrame,
) -> tuple[Path, str]:
    best_method = metrics_df["concordex"].dropna().idxmax()
    print(f"Best method by concordex: {best_method}")
    proj = projections[best_method]

    per_class = calculate_concordex_per_class(
        proj,
        labels,
        n_neighbors=CONCORDEX_N_NEIGHBORS,
        n_permutations=CONCORDEX_N_PERMUTATIONS,
        random_state=CONCORDEX_RANDOM_STATE,
    )
    df = pd.DataFrame(per_class).T
    df = df.sort_values("n_members", ascending=False)

    fig, ax = plt.subplots(figsize=(8.0, max(3.6, 0.35 * len(df) + 1.0)))
    bars = ax.barh(
        [_short(s) for s in df.index][::-1],
        df["ratio"][::-1],
        color=PRIMARY,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.axvline(
        1.0,
        color=ACCENT,
        linestyle="--",
        linewidth=1.2,
        zorder=2,
        label="null = 1",
    )
    for bar, n, ratio in zip(
        bars, df["n_members"][::-1], df["ratio"][::-1], strict=False
    ):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{ratio:.2f}  (n={int(n)})",
            ha="left",
            va="center",
            fontsize=8.5,
            color=GREY,
        )

    ax.set_xlabel(
        "per-class concordex   (observed within-class neighbour fraction "
        "/ expected)",
        fontsize=10,
    )
    ax.set_xlim(0, max(df["ratio"].max() * 1.25, 2.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color=LIGHT_GREY, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title(
        f"Per-class concordex on {METHOD_TITLES.get(best_method, best_method)} "
        f"({DATA})",
        fontsize=12,
        loc="left",
        pad=10,
        color=PRIMARY,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "concordex_per_class.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out, best_method


# Plot 3
def plot_label_matrix(
    projections: dict[str, np.ndarray],
    labels: np.ndarray,
    method: str,
) -> Path:
    proj = projections[method]
    valid_mask = np.array([lbl is not None for lbl in labels])
    coords = proj[valid_mask]
    valid_labels = labels[valid_mask]

    K, classes = neighborhood_consolidation_matrix(
        coords, valid_labels, n_neighbors=CONCORDEX_N_NEIGHBORS
    )
    D = aggregated_label_matrix(K, valid_labels, classes)

    # Order rows/cols by class size (descending) so the matrix reads top-down.
    order = np.argsort(
        [-(valid_labels == c).sum() for c in classes]
    )
    classes_o = classes[order]
    D_o = D[order][:, order]

    cmap = LinearSegmentedColormap.from_list(
        "concordex_teal", ["#FFFFFF", PRIMARY], N=256
    )

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    im = ax.imshow(D_o, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes_o)))
    ax.set_yticks(range(len(classes_o)))
    ax.set_xticklabels([_short(str(c), 18) for c in classes_o],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([_short(str(c), 18) for c in classes_o], fontsize=8)
    ax.set_xlabel("neighbour class  (column j)", fontsize=10)
    ax.set_ylabel("anchor class  (row a)", fontsize=10)

    # Highlight the diagonal: the within-class concordance entries.
    for i in range(len(classes_o)):
        ax.add_patch(
            plt.Rectangle(
                (i - 0.5, i - 0.5),
                1, 1,
                fill=False,
                edgecolor=ACCENT,
                linewidth=1.5,
            )
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "D[a, j]   (mean fraction of class-j neighbours\namong points of class a)",
        fontsize=9,
    )

    ax.set_title(
        f"Aggregated label matrix D   —   {METHOD_TITLES.get(method, method)}",
        fontsize=12,
        loc="left",
        pad=10,
        color=PRIMARY,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "concordex_label_matrix.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out



# Plot 4
def plot_projection_grid(
    projections: dict[str, np.ndarray],
    labels: np.ndarray,
    metrics_df: pd.DataFrame,
) -> Path:
    color_map, sizes = _build_color_map(labels)
    classes_sorted = sorted(color_map.keys(), key=lambda c: -sizes.get(c, 0))

    methods_in_order = [m for m in METHOD_TITLES if m in projections]
    n = len(methods_in_order)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(4.6 * cols, 4.4 * rows), squeeze=False
    )

    for ax, method in zip(axes.flat, methods_in_order, strict=False):
        coords = projections[method]
        for cls in classes_sorted:
            mask = labels == cls
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=14,
                alpha=0.85,
                color=color_map[cls],
                edgecolor="white",
                linewidth=0.25,
                label=f"{_short(cls)} (n={mask.sum()})",
            )
        unl = np.array([lbl is None for lbl in labels])
        if unl.any():
            ax.scatter(
                coords[unl, 0],
                coords[unl, 1],
                s=8,
                alpha=0.35,
                color="lightgrey",
                label=f"unlabelled (n={unl.sum()})",
            )

        title = METHOD_TITLES.get(method, method.upper())
        if method in metrics_df.index and "concordex" in metrics_df.columns:
            c = metrics_df.loc[method, "concordex"]
            if not pd.isna(c):
                title = f"{title}   concordex = {c:.2f}"
        ax.set_title(title, fontsize=10, color=PRIMARY)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.15)

    for ax in axes.flat[n:]:
        ax.axis("off")

    handles, lbls = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            lbls,
            loc="upper center",          
            ncol=4,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),  
        )

    n_total = len(labels)
    n_lab = sum(1 for lbl in labels if lbl is not None)
    fig.suptitle(
        f"Projections coloured by functional label — {DATA}  "
        f"(n={n_total}, labelled={n_lab})",
        fontsize=12,
        y=1.0,
        color=PRIMARY,
    )
    
    # Squeeze the plots into the middle 90% of the image, 
    # leaving 8% blank space at the bottom for the legend and 2% at the top for the title
    fig.tight_layout(rect=[0, 0.08, 1, 0.98]) 
    
    out = RESULTS_DIR / "projections_concordex.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out



def main():
    if not RESULTS_DIR.exists():
        sys.exit(f"No results at {RESULTS_DIR}. Run: python run.py")

    labels, _ = _load_labels()
    projections = _load_projections()
    metrics_df = _load_metrics_csv()
    if metrics_df is None or "concordex" not in metrics_df.columns:
        sys.exit(
            "metrics.csv missing or has no concordex column. "
            "Re-run run.py first."
        )

    out_paths = []
    out_paths.append(plot_concordex_by_method(metrics_df))
    per_class_path, best_method = plot_concordex_per_class(
        projections, labels, metrics_df
    )
    out_paths.append(per_class_path)
    out_paths.append(plot_label_matrix(projections, labels, best_method))
    out_paths.append(plot_projection_grid(projections, labels, metrics_df))

    print("\nFinished. Drop these into the slide placeholders:")
    for p in out_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()