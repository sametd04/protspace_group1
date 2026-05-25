#!/usr/bin/env python3
"""Unified robustness benchmarking CLI for DR methods.

Tests stability of dimensionality reduction methods using k-NN overlap metrics.
Supports: UMAP, t-SNE, PaCMAP, LocalMAP, MDS

Usage:
    # Run all methods with automatic plotting (recommended)
    uv run python -m protspace.benchmark.robustness.cli --methods all --plot

    # Run specific methods with plotting
    uv run python -m protspace.benchmark.robustness.cli --methods umap tsne --plot

    # Run experiments only (no plot)
    uv run python -m protspace.benchmark.robustness.cli --methods all

    # Generate plot from existing results
    uv run python -m protspace.benchmark.robustness.cli --skip-experiments --plot
"""

from __future__ import annotations

import argparse

from protspace.benchmark.io import benchmark_paths
from protspace.benchmark.robustness.config import get_all_methods
from protspace.benchmark.robustness.run import RobustnessRunner
from protspace.data.loaders import load_h5


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run robustness experiments for DR methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods with automatic plotting (recommended)
  uv run python -m protspace.benchmark.robustness.cli --methods all --plot

  # Run specific methods with plotting
  uv run python -m protspace.benchmark.robustness.cli --methods umap tsne --plot

  # Run experiments only (no visualization)
  uv run python -m protspace.benchmark.robustness.cli --methods all

  # Generate plot from existing results (skip experiments)
  uv run python -m protspace.benchmark.robustness.cli --skip-experiments --plot

  # Use a different dataset
  uv run python -m protspace.benchmark.robustness.cli --methods all --dataset toxprot --plot
        """,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=["all", "umap", "tsne", "pacmap", "localmap", "mds"],
        help="DR method(s) to benchmark. Use 'all' for all methods.",
    )

    parser.add_argument(
        "--dataset",
        default="3ftx",
        choices=["3ftx", "toxprot", "pla2g2", "cath_s40", "swissprot_rr"],
        help="Dataset to use (default: 3ftx)",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plot after experiments complete",
    )

    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Skip experiments and only generate plot from existing results",
    )

    return parser.parse_args()


def main():
    """Run robustness experiments for specified methods."""
    args = parse_args()

    # Determine which methods to run
    if "all" in args.methods:
        methods_to_run = get_all_methods()
    else:
        methods_to_run = args.methods

    dataset = args.dataset
    paths = benchmark_paths(dataset)

    print("\n" + "=" * 80)
    print("UNIFIED DR METHOD ROBUSTNESS BENCHMARKING")
    print("=" * 80)
    print(f"Dataset: {dataset}")
    print(f"Methods: {', '.join(m.upper() for m in methods_to_run)}")
    if args.plot:
        print("Plot: Will generate comparison plot after experiments")
    if args.skip_experiments:
        print("Mode: Plot-only (skipping experiments)")
    print("=" * 80 + "\n")

    # Run experiments (unless skipped)
    successful = []
    failed = []

    if not args.skip_experiments:
        # Load embeddings once
        print(f"Loading embeddings from {paths.embedding_path}...")
        emb_set = load_h5([paths.embedding_path])
        embeddings = emb_set.data

        print(
            f"Loaded {embeddings.shape[0]} proteins with {embeddings.shape[1]} features\n"
        )

        # Run robustness for each method
        for method in methods_to_run:
            try:
                runner = RobustnessRunner(
                    method=method,
                    embeddings=embeddings,
                    dataset=dataset,
                    output_base=paths.output_dir,
                )
                runner.run()
                successful.append(method)
            except Exception as e:
                print(f"\n⚠️  Error running {method}: {e}\n")
                failed.append(method)
                continue

        # Print experiment summary
        print("\n" + "=" * 80)
        print("ALL ROBUSTNESS EXPERIMENTS COMPLETED")
        print("=" * 80)
        print(f"Successful: {len(successful)}/{len(methods_to_run)}")
        if successful:
            print(f"  - {', '.join(successful)}")
        if failed:
            print(f"Failed: {len(failed)}/{len(methods_to_run)}")
            print(f"  - {', '.join(failed)}")
        print(f"\nResults saved to: {paths.output_dir}")
        print("=" * 80 + "\n")
    else:
        print("Skipping experiments (using existing results)\n")

    # Generate comparison plot if requested
    if args.plot:
        try:
            from protspace.benchmark.robustness.config import METHOD_CONFIGS
            from protspace.benchmark.robustness.plot import (
                create_comparison_plot,
                load_method_data,
                print_summary_table,
            )

            print("\n" + "=" * 80)
            print("GENERATING COMPARISON PLOT")
            print("=" * 80 + "\n")

            results_base = paths.output_dir / "robustness"

            # Load data for all methods
            all_data = []
            for method_name in METHOD_CONFIGS.keys():
                if not args.skip_experiments and method_name not in successful:
                    # Skip methods that failed during experiments
                    continue
                print(f"Loading {method_name.upper()} results...")
                data = load_method_data(method_name, results_base)
                if data is not None:
                    all_data.append(data)

            if all_data:
                print(f"\nSuccessfully loaded {len(all_data)} methods")

                # Create comparison plot
                output_path = results_base / "comparison_plot.png"
                output_path.parent.mkdir(exist_ok=True, parents=True)
                create_comparison_plot(all_data, output_path, dataset)

                # Print summary table
                print_summary_table(all_data)

                print("\n" + "=" * 80)
                print("PLOTTING COMPLETED")
                print(f"Plot saved to: {output_path}")
                print("=" * 80 + "\n")
            else:
                print("\n⚠️  No data available for plotting\n")

        except Exception as e:
            print(f"\n⚠️  Error generating plot: {e}\n")

    # Final summary
    if not args.skip_experiments or args.plot:
        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETED")
        print("=" * 80)
        if not args.skip_experiments:
            print(f"Experiments: {len(successful)}/{len(methods_to_run)} successful")
        if args.plot:
            print("Plot: Generated successfully")
        print(f"\nAll outputs in: {paths.output_dir}/robustness/")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
