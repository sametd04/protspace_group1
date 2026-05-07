#!/usr/bin/env python3
"""Visualize normalized vs raw projections side-by-side with colors.

Creates matplotlib plots showing both normalized and raw projections
with protein classes colored and coordinate ranges displayed.
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

# Suppress numerical precision warnings from sklearn
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from protspace.benchmark.io import BenchmarkPaths, benchmark_paths
from protspace.benchmark.labels import label_summary, load_silhouette_labels
from protspace.benchmark.run import METHOD_TITLES, METHODS
from protspace.data.loaders import load_h5


def visualize_comparison(paths: BenchmarkPaths) -> None:
    """Generate side-by-side comparison plots with colors."""
    # Load embeddings and run both normalized and raw
    emb_set = load_h5([paths.embedding_path])
    embeddings = emb_set.data
    headers = emb_set.headers

    # Load labels for coloring
    labels = load_silhouette_labels(paths, headers)
    if labels is not None:
        summary = label_summary(labels)
        print(
            f"Labels: {summary['n_labelled']}/{summary['n_total']} proteins, {summary['n_classes']} classes"
        )
    else:
        labels = np.array([None] * len(headers), dtype=object)

    # Get unique classes for color mapping
    valid_labels = [lbl for lbl in labels if lbl is not None]
    classes = sorted(set(valid_labels)) if valid_labels else []

    # Create color map
    if classes:
        cmap = plt.colormaps["tab20"](np.linspace(0, 1, len(classes)))
        color_map = {cls: cmap[i] for i, cls in enumerate(classes)}
    else:
        color_map = {}

    # Run benchmark methods
    from protspace.benchmark import benchmark_methods
    from protspace.utils.constants import DimensionReductionConfig

    config = DimensionReductionConfig(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        perplexity=min(30, embeddings.shape[0] // 4),
    )

    print("Running normalized projections...")
    results_norm = benchmark_methods(
        embeddings=embeddings,
        methods=METHODS,
        config=config,
        normalize=True,
    )

    print("Running raw projections...")
    results_raw = benchmark_methods(
        embeddings=embeddings,
        methods=METHODS,
        config=config,
        normalize=False,
    )

    # Create comparison plots
    n_methods = len(METHODS)
    fig, axes = plt.subplots(n_methods, 2, figsize=(12, 4 * n_methods))

    if n_methods == 1:
        axes = axes.reshape(1, -1)

    for i, method in enumerate(METHODS):
        # Plot normalized (left column)
        ax_norm = axes[i, 0]
        proj_norm = results_norm[method].projection

        # Plot each class with its color
        for cls in classes:
            mask = labels == cls
            if mask.any():
                ax_norm.scatter(
                    proj_norm[mask, 0],
                    proj_norm[mask, 1],
                    c=[color_map[cls]],
                    s=10,
                    alpha=0.7,
                    label=cls,
                )

        # Plot unlabeled points in gray
        unlabeled = np.array([lbl is None for lbl in labels])
        if unlabeled.any():
            ax_norm.scatter(
                proj_norm[unlabeled, 0],
                proj_norm[unlabeled, 1],
                c="lightgray",
                s=5,
                alpha=0.4,
                label="unlabeled",
            )

        ax_norm.set_title(f"{METHOD_TITLES.get(method, method.upper())} - Normalized")
        ax_norm.set_xlabel("X")
        ax_norm.set_ylabel("Y")
        ax_norm.grid(True, alpha=0.3)
        ax_norm.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
        ax_norm.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
        ax_norm.set_aspect("equal")

        # Add coordinate range
        norm_range = f"X: [{proj_norm[:, 0].min():.2f}, {proj_norm[:, 0].max():.2f}]\\nY: [{proj_norm[:, 1].min():.2f}, {proj_norm[:, 1].max():.2f}]"
        ax_norm.text(
            0.02,
            0.98,
            norm_range,
            transform=ax_norm.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Plot raw (right column)
        ax_raw = axes[i, 1]
        proj_raw = results_raw[method].projection

        # Plot each class with its color
        for cls in classes:
            mask = labels == cls
            if mask.any():
                ax_raw.scatter(
                    proj_raw[mask, 0],
                    proj_raw[mask, 1],
                    c=[color_map[cls]],
                    s=10,
                    alpha=0.7,
                    label=cls,
                )

        # Plot unlabeled points in gray
        if unlabeled.any():
            ax_raw.scatter(
                proj_raw[unlabeled, 0],
                proj_raw[unlabeled, 1],
                c="lightgray",
                s=5,
                alpha=0.4,
                label="unlabeled",
            )

        ax_raw.set_title(f"{METHOD_TITLES.get(method, method.upper())} - Raw")
        ax_raw.set_xlabel("X")
        ax_raw.set_ylabel("Y")
        ax_raw.grid(True, alpha=0.3)
        ax_raw.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
        ax_raw.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
        ax_raw.set_aspect("equal")

        # Add coordinate range
        raw_range = f"X: [{proj_raw[:, 0].min():.2f}, {proj_raw[:, 0].max():.2f}]\\nY: [{proj_raw[:, 1].min():.2f}, {proj_raw[:, 1].max():.2f}]"
        ax_raw.text(
            0.02,
            0.98,
            raw_range,
            transform=ax_raw.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Add legend (only once, at the bottom)
    if classes:
        handles, labels_legend = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles[: len(classes)],  # Exclude unlabeled
            labels_legend[: len(classes)],
            loc="lower center",
            ncol=4,
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
        )

    plt.tight_layout()

    # Save figure
    output_path = paths.output_dir / "normalization_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\\nSaved comparison plot to: {output_path}")

    # Also save as PDF
    output_pdf = paths.output_dir / "normalization_comparison.pdf"
    plt.savefig(output_pdf, bbox_inches="tight")
    print(f"Saved PDF version to: {output_pdf}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualize normalized vs raw projections with colors"
    )
    parser.add_argument(
        "--data",
        default=os.environ.get("DATA", "3ftx"),
        help="Dataset name (default: env DATA or 3ftx)",
    )
    args = parser.parse_args(argv)
    paths = benchmark_paths(args.data)
    visualize_comparison(paths)


if __name__ == "__main__":
    main()
