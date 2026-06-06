#!/usr/bin/env python3
"""Plot robustness comparison across all DR methods.

Visualizes seed robustness vs hyperparameter robustness for:
UMAP, t-SNE, PaCMAP, LocalMAP, and MDS.

Also provides group-level heatmap plotting for group-based analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from protspace.benchmark.robustness.config import METHOD_CONFIGS


def load_method_data(method_name: str, results_base: Path) -> dict:
    """Load robustness data for a single method.

    Args:
        method_name: Name of the DR method (lowercase)
        results_base: Base path to results directory

    Returns:
        Dictionary with seed_robustness and hyperparam_robustness scores
    """
    # Get hyperparam experiments from config
    hyperparam_experiments = list(
        METHOD_CONFIGS[method_name]["hyperparam_experiments"].keys()
    )

    csv_path = results_base / method_name / "summary.csv"

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Skipping {method_name.upper()}.")
        return None

    # Read summary CSV
    df = pd.read_csv(csv_path)

    # Extract seed robustness
    seed_row = df[df["experiment"] == "seed"]
    if seed_row.empty:
        print(f"Warning: No seed experiment found for {method_name.upper()}")
        seed_robustness = None
    else:
        seed_robustness = float(seed_row["mean_knn_overlap"].iloc[0])

    # Extract hyperparameter robustness (average over all hyperparam experiments)
    hyperparam_rows = df[df["experiment"].isin(hyperparam_experiments)]

    if hyperparam_rows.empty:
        print(f"Warning: No hyperparameter experiments found for {method_name.upper()}")
        hyperparam_robustness = None
    else:
        hyperparam_robustness = float(hyperparam_rows["mean_knn_overlap"].mean())

    return {
        "method": method_name.upper(),
        "seed_robustness": seed_robustness,
        "hyperparam_robustness": hyperparam_robustness,
    }


def create_comparison_plot(data: list[dict], output_path: Path, dataset: str) -> None:
    """Create grouped bar chart comparing robustness across methods.

    Args:
        data: List of dicts with method, seed_robustness, hyperparam_robustness
        output_path: Path to save the plot
        dataset: Dataset name for title
    """
    # Extract data for plotting
    methods = [d["method"] for d in data]
    seed_scores = [d["seed_robustness"] for d in data]
    hyperparam_scores = [d["hyperparam_robustness"] for d in data]

    # Set up the plot
    x = np.arange(len(methods))
    width = 0.25  # Width of bars

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create grouped bars
    bars1 = ax.bar(
        x - width / 2, seed_scores, width, label="Seed robustness", alpha=0.8
    )
    bars2 = ax.bar(
        x + width / 2,
        hyperparam_scores,
        width,
        label="Hyperparameter robustness",
        alpha=0.8,
    )

    # Customize plot
    ax.set_ylabel("Mean k-NN overlap stability", fontsize=16, fontweight="bold")
    ax.set_xlabel("DR method", fontsize=16, fontweight="bold")
    ax.set_title(
        f"Robustness of DR methods under perturbations (Dataset: {dataset})",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=16)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=13,
            )

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Add horizontal reference lines
    ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.3, linewidth=1)
    ax.text(
        len(methods) - 0.5,
        0.82,
        "High stability (>0.8)",
        fontsize=13,
        color="green",
        alpha=0.6,
    )
    ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.3, linewidth=1)
    ax.text(
        len(methods) - 0.5,
        0.52,
        "Moderate stability (>0.5)",
        fontsize=13,
        color="orange",
        alpha=0.6,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def print_summary_table(data: list[dict]) -> None:
    """Print summary table of robustness scores.

    Args:
        data: List of dicts with method, seed_robustness, hyperparam_robustness
    """
    print("\n" + "=" * 80)
    print("ROBUSTNESS COMPARISON SUMMARY")
    print("=" * 80)
    print(
        f"\n{'Method':<12} {'Seed Robustness':<20} {'Hyperparam Robustness':<20} {'Overall':<12}"
    )
    print("-" * 80)

    for d in data:
        seed = d["seed_robustness"]
        hyper = d["hyperparam_robustness"]
        overall = (seed + hyper) / 2  # Simple average

        print(f"{d['method']:<12} {seed:<20.6f} {hyper:<20.6f} {overall:<12.6f}")

    print("-" * 80)
    print("\nInterpretation:")
    print("  • >0.8 = High stability (recommended for production)")
    print("  • 0.5-0.8 = Moderate stability (use with caution)")
    print("  • <0.5 = Low stability (avoid for robust analyses)")
    print("=" * 80 + "\n")


def main():
    """Load data, create plot, and print summary."""
    parser = argparse.ArgumentParser(
        description="Plot robustness comparison across DR methods"
    )
    parser.add_argument(
        "--dataset",
        default="3ftx",
        help="Dataset name (default: 3ftx)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: results/{dataset})",
    )
    args = parser.parse_args()

    # Determine results directory
    if args.output_dir:
        results_base = args.output_dir
    else:
        # Default: results/{dataset} from project root
        project_root = Path(__file__).parent.parent.parent.parent.parent
        results_base = project_root / "results" / args.dataset

    print("\n" + "=" * 80)
    print("DR METHOD ROBUSTNESS COMPARISON")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Loading data from: {results_base}\n")

    # Load data for all methods
    all_data = []
    for method_name in METHOD_CONFIGS.keys():
        print(f"Loading {method_name.upper()}...")
        data = load_method_data(method_name, results_base)
        if data is not None:
            all_data.append(data)

    if not all_data:
        print("Error: No data loaded. Exiting.")
        return

    print(f"\nSuccessfully loaded {len(all_data)} methods")

    # Create comparison plot
    output_path = results_base / "robustness" / "comparison_plot.png"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    create_comparison_plot(all_data, output_path, args.dataset)

    # Print summary table
    print_summary_table(all_data)


def plot_group_heatmap(
    results_df: pd.DataFrame,
    experiment_type: str,
    output_path: Path,
    knn_k: int = 15,
    dataset_name: str = "Dataset",
) -> None:
    """Create heatmap of DR method robustness by protein group.
    
    Args:
        results_df: DataFrame with columns: method, experiment_type, group, robustness
        experiment_type: Which experiment type to plot
        output_path: Path to save the heatmap
        knn_k: Value of k for k-NN overlap
        dataset_name: Name of dataset for title
    """
    # Filter to specific experiment type
    plot_df = results_df[results_df["experiment_type"] == experiment_type].copy()

    if len(plot_df) == 0:
        print(f"No data for experiment type: {experiment_type}")
        return

    # Pivot for heatmap
    heatmap_data = plot_df.pivot(
        index="group", columns="method", values="robustness"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, len(heatmap_data) * 0.5)))

    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        center=0.8,
        cbar_kws={"label": "k-NN Overlap (Robustness)"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    ax.set_title(
        f"DR Method Robustness by Protein Group - {dataset_name}\n"
        f"Experiment: {experiment_type.upper()} | k={knn_k}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("DR Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Protein Group", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved heatmap: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
