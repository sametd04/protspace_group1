"""Core robustness benchmarking logic for DR methods.

Runs experiments and computes k-NN overlap stability metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from protspace.benchmark.harness import benchmark_method
from protspace.benchmark.metrics import calculate_knn_overlap
from protspace.benchmark.robustness.config import KNN_K, get_method_config
from protspace.utils.constants import DimensionReductionConfig


class RobustnessRunner:
    """Runs robustness experiments for a DR method."""

    def __init__(
        self,
        method: str,
        embeddings: np.ndarray,
        dataset: str,
        output_base: Path,
    ):
        """Initialize robustness runner.

        Args:
            method: DR method name
            embeddings: High-dimensional embeddings
            dataset: Dataset name
            output_base: Base output directory
        """
        self.method = method
        self.embeddings = embeddings
        self.dataset = dataset
        self.output_base = output_base

        # Load method configuration
        self.config = get_method_config(method)
        self.baseline_config = self.config["baseline"]
        self.seed_values = self.config["seed_values"]
        self.hyperparam_experiments = self.config["hyperparam_experiments"]

        # Setup output directory - save in robustness subfolder
        self.output_dir = output_base / "robustness" / method
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Will be set during baseline run
        self.baseline_projection = None
        self.baseline_params = None

    def run_baseline(self) -> None:
        """Run baseline with default parameters."""
        print("=" * 80)
        print("GENERATING BASELINE EMBEDDING")
        print("=" * 80)
        print(f"Method: {self.method.upper()}")
        print(f"Configuration: {self.baseline_config}\n")

        config = DimensionReductionConfig(**self.baseline_config)
        result = benchmark_method(
            embeddings=self.embeddings,
            method=self.method,
            config=config,
            normalize=False,
            metric_functions=None,
        )

        print(f"Baseline runtime: {result.time_seconds:.3f}s")
        print(f"Baseline projection shape: {result.projection.shape}\n")

        self.baseline_projection = result.projection
        self.baseline_params = result.params

    def run_seed_robustness(self) -> list[dict]:
        """Test robustness to random seed variations."""
        print("=" * 80)
        print("EXPERIMENT: SEED ROBUSTNESS")
        print("=" * 80)
        print(f"Testing random_state values: {self.seed_values}")
        print("All other parameters fixed to baseline\n")

        results = []

        for seed in self.seed_values:
            print(f"Running random_state={seed}...")

            config_dict = self.baseline_config.copy()
            config_dict["random_state"] = seed
            config = DimensionReductionConfig(**config_dict)

            result = benchmark_method(
                embeddings=self.embeddings,
                method=self.method,
                config=config,
                normalize=False,
                metric_functions=None,
            )

            knn_overlap = calculate_knn_overlap(
                self.baseline_projection, result.projection, n_neighbors=KNN_K
            )

            print(f"  Runtime: {result.time_seconds:.3f}s")
            print(f"  k-NN overlap vs baseline: {knn_overlap:.6f}\n")

            results.append(
                {
                    "method": self.method,
                    "experiment": "seed",
                    "params": f"random_state={seed}",
                    "knn_overlap": knn_overlap,
                    "runtime_seconds": result.time_seconds,
                }
            )

        return results

    def run_hyperparam_robustness(
        self, param_name: str, param_values: list
    ) -> list[dict]:
        """Test robustness to hyperparameter variations."""
        print("=" * 80)
        print(f"EXPERIMENT: {param_name.upper()} ROBUSTNESS")
        print("=" * 80)
        print(f"Testing {param_name} values: {param_values}")
        print("All other parameters fixed to baseline\n")

        results = []

        for value in param_values:
            print(f"Running {param_name}={value}...")

            config_dict = self.baseline_config.copy()
            config_dict[param_name] = value
            config = DimensionReductionConfig(**config_dict)

            result = benchmark_method(
                embeddings=self.embeddings,
                method=self.method,
                config=config,
                normalize=False,
                metric_functions=None,
            )

            knn_overlap = calculate_knn_overlap(
                self.baseline_projection, result.projection, n_neighbors=KNN_K
            )

            print(f"  Runtime: {result.time_seconds:.3f}s")
            print(f"  k-NN overlap vs baseline: {knn_overlap:.6f}\n")

            results.append(
                {
                    "method": self.method,
                    "experiment": param_name,
                    "params": f"{param_name}={value}",
                    "knn_overlap": knn_overlap,
                    "runtime_seconds": result.time_seconds,
                }
            )

        return results

    def run_all_experiments(self) -> pd.DataFrame:
        """Run all robustness experiments for this method.

        Returns:
            DataFrame with all results
        """
        print("\n" + "=" * 80)
        print(f"{self.method.upper()} ROBUSTNESS EXPERIMENTS WITH k-NN OVERLAP")
        print("=" * 80)
        print(f"Dataset: {self.dataset}")
        print(f"k-NN parameter: k={KNN_K}")
        print("=" * 80 + "\n")
        print(f"Output directory: {self.output_dir}\n")

        # Run baseline
        self.run_baseline()

        # Collect all results
        all_results = []

        # Seed robustness
        seed_results = self.run_seed_robustness()
        all_results.extend(seed_results)

        # Hyperparameter robustness
        for param_name, param_values in self.hyperparam_experiments.items():
            hyperparam_results = self.run_hyperparam_robustness(
                param_name, param_values
            )
            all_results.extend(hyperparam_results)

        return pd.DataFrame(all_results)

    @staticmethod
    def compute_summary_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics per experiment type."""
        summary = (
            results_df.groupby("experiment")["knn_overlap"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )
        summary.columns = [
            "experiment",
            "mean_knn_overlap",
            "std_knn_overlap",
            "min_knn_overlap",
            "max_knn_overlap",
            "n_runs",
        ]
        return summary

    def save_results(self, results_df: pd.DataFrame) -> None:
        """Save results to CSV and JSON files."""
        # Detailed results
        csv_path = self.output_dir / "results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to: {csv_path}")

        # Summary statistics
        summary_df = self.compute_summary_statistics(results_df)
        summary_csv_path = self.output_dir / "summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Summary statistics saved to: {summary_csv_path}")

        # Metadata
        metadata = {
            "baseline_config": self.baseline_params,
            "knn_k": KNN_K,
            "experiments": {
                "seed": {
                    "values": self.seed_values,
                    "n_runs": len(self.seed_values),
                }
            },
        }

        for param_name, param_values in self.hyperparam_experiments.items():
            metadata["experiments"][param_name] = {
                "values": param_values,
                "n_runs": len(param_values),
            }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")

    def print_summary(self, results_df: pd.DataFrame) -> None:
        """Print summary of results to console."""
        print("\n" + "=" * 80)
        print(f"{self.method.upper()} ROBUSTNESS SUMMARY STATISTICS")
        print("=" * 80)

        summary = self.compute_summary_statistics(results_df)

        print("\nk-NN Overlap Statistics by Experiment:")
        print("-" * 80)
        for _, row in summary.iterrows():
            print(f"\n{row['experiment'].upper()}:")
            print(f"  Mean:   {row['mean_knn_overlap']:.6f}")
            print(f"  Std:    {row['std_knn_overlap']:.6f}")
            print(f"  Min:    {row['min_knn_overlap']:.6f}")
            print(f"  Max:    {row['max_knn_overlap']:.6f}")
            print(f"  N runs: {int(row['n_runs'])}")

        print("\n" + "=" * 80)
        print("INTERPRETATION GUIDE")
        print("=" * 80)
        print("k-NN overlap score measures neighborhood preservation:")
        print("  • 1.0 = Perfect stability (all neighbors preserved)")
        print("  • 0.8-1.0 = High stability (small perturbations)")
        print("  • 0.5-0.8 = Moderate stability (noticeable changes)")
        print("  • <0.5 = Low stability (significant structural changes)")
        print("\nLower std = more robust to parameter changes")
        print("=" * 80 + "\n")

    def run(self) -> None:
        """Run all experiments, save results, and print summary."""
        results_df = self.run_all_experiments()
        self.print_summary(results_df)
        self.save_results(results_df)

        print("\n" + "=" * 80)
        print(f"{self.method.upper()} EXPERIMENTS COMPLETED")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 80 + "\n")
