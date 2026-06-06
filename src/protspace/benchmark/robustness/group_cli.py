#!/usr/bin/env python3
"""Group-based DR robustness analysis CLI wrapper.

Thin wrapper script that uses the refactored library modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from protspace.benchmark.robustness.data_utils import load_and_prepare_data
from protspace.benchmark.robustness.group_analysis import GroupDRRobustnessAnalyzer
from protspace.benchmark.robustness.plot import plot_group_heatmap


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Group-level DR robustness analysis"
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        required=True,
        help="Path to embeddings file (.h5)",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Path to metadata CSV",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="DR methods to test (default: all)",
    )
    parser.add_argument(
        "--experiment-types",
        nargs="+",
        default=["seed"],
        help="Experiment types (default: seed). Options: seed, n_neighbors, min_dist, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: results/{dataset}/robustness/group_analysis)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name for organizing results (e.g., '3ftx', 'swissprot_rr')",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="major_group",
        help="Column name for grouping (default: major_group)",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=15,
        help="Minimum group size (default: 15)",
    )
    parser.add_argument(
        "--identifier-column",
        type=str,
        default="identifier",
        help="Column name for identifiers in metadata (default: identifier)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Dataset",
        help="Dataset name for plot titles (default: Dataset)",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.dataset:
        # Use benchmark-style path: src/protspace/benchmark/results/{dataset}/robustness/group_analysis
        benchmark_root = Path(__file__).resolve().parent.parent / "src" / "protspace" / "benchmark"
        output_dir = benchmark_root / "results" / args.dataset / "robustness" / "group_analysis"
    else:
        # Fallback: use embedding filename in benchmark results
        dataset_name = args.embedding_path.stem
        benchmark_root = Path(__file__).resolve().parent.parent / "src" / "protspace" / "benchmark"
        output_dir = benchmark_root / "results" / dataset_name / "robustness" / "group_analysis"
    
    # Use dataset name from args, or derive from embedding path
    dataset_display_name = args.dataset_name or args.dataset or args.embedding_path.stem

    print("=" * 70)
    print("GROUP-LEVEL DR ROBUSTNESS ANALYSIS")
    print("=" * 70)
    print(f"Embeddings: {args.embedding_path}")
    print(f"Metadata: {args.metadata_path}")
    print(f"Dataset: {dataset_display_name}")
    print(f"Group column: {args.group_column}")
    print(f"Min group size: {args.min_group_size}")
    print(f"Methods: {args.methods or 'all'}")
    print(f"Experiment types: {args.experiment_types}")
    print(f"Output: {output_dir}")
    print()

    # Load and prepare data
    embeddings, metadata_df = load_and_prepare_data(
        args.embedding_path,
        args.metadata_path,
        identifier_column=args.identifier_column,
    )

    # Initialize analyzer
    analyzer = GroupDRRobustnessAnalyzer(
        embeddings=embeddings,
        metadata_df=metadata_df,
        group_column=args.group_column,
        min_group_size=args.min_group_size,
        output_dir=output_dir,
    )

    # Run analysis
    results_df = analyzer.run_all_methods(
        methods=args.methods, experiment_types=args.experiment_types
    )

    # Print summary
    analyzer.print_summary(results_df)

    # Save results
    analyzer.save_results(results_df)

    # Create heatmaps for each experiment type
    print("\nGenerating heatmaps...")
    for exp_type in results_df["experiment_type"].unique():
        output_path = output_dir / f"heatmap_{exp_type}.png"
        plot_group_heatmap(
            results_df,
            experiment_type=exp_type,
            output_path=output_path,
            knn_k=analyzer.knn_k,
            dataset_name=dataset_display_name,
        )

    # Create combined hyperparameter heatmap
    hyperparam_types = [t for t in results_df["experiment_type"].unique() if t != "seed"]
    if hyperparam_types:
        print("\nCreating combined hyperparameter heatmap...")
        hyperparam_df = results_df[results_df["experiment_type"].isin(hyperparam_types)].copy()
        
        # Filter out invalid 0.0 values (methods that don't support those hyperparameters)
        hyperparam_df = hyperparam_df[hyperparam_df["robustness"] > 0.0]
        
        # Average robustness across all hyperparameter experiments
        hyperparam_avg = (
            hyperparam_df.groupby(["method", "group"])["robustness"]
            .mean()
            .reset_index()
        )
        hyperparam_avg["experiment_type"] = "hyperparameters_combined"
        hyperparam_avg["group_size"] = hyperparam_avg["group"].map(
            lambda g: analyzer.groups[g]["size"]
        )

        # Plot combined hyperparameter heatmap
        output_path = output_dir / "heatmap_hyperparameters_combined.png"
        plot_group_heatmap(
            hyperparam_avg,
            experiment_type="hyperparameters_combined",
            output_path=output_path,
            knn_k=analyzer.knn_k,
            dataset_name=dataset_display_name,
        )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
