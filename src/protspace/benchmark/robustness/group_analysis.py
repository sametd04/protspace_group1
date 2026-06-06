"""Group-based DR robustness analysis.

Analyzes DR method robustness per protein group instead of globally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from protspace.benchmark.harness import benchmark_method
from protspace.benchmark.metrics import calculate_knn_overlap
from protspace.benchmark.robustness.config import KNN_K, METHOD_CONFIGS, get_all_methods
from protspace.utils.constants import DimensionReductionConfig


class GroupDRRobustnessAnalyzer:
    """Analyzes DR method robustness per protein group."""

    def __init__(
        self,
        embeddings: np.ndarray,
        metadata_df: pd.DataFrame,
        group_column: str = "major_group",
        min_group_size: int = 15,
        knn_k: int = KNN_K,
        output_dir: Path | None = None,
    ):
        """Initialize the analyzer.

        Args:
            embeddings: High-dimensional protein embeddings (N, D)
            metadata_df: DataFrame with group annotations
            group_column: Column name for grouping
            min_group_size: Minimum samples per group
            knn_k: k for k-NN overlap computation
            output_dir: Output directory (optional)
        """
        self.embeddings = embeddings
        self.metadata_df = metadata_df.copy()
        self.group_column = group_column
        self.min_group_size = min_group_size
        self.knn_k = knn_k
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare groups
        self._prepare_groups()

    def _prepare_groups(self):
        """Filter and prepare protein groups."""
        # Remove NaN
        self.metadata_df = self.metadata_df.dropna(subset=[self.group_column])

        # Filter small groups
        group_sizes = self.metadata_df[self.group_column].value_counts()
        valid_groups = group_sizes[group_sizes >= self.min_group_size].index
        self.metadata_df = self.metadata_df[
            self.metadata_df[self.group_column].isin(valid_groups)
        ]

        # Get indices to keep and filter embeddings
        kept_indices = self.metadata_df.index.tolist()
        self.embeddings = self.embeddings[kept_indices]
        
        # Reset index after filtering
        self.metadata_df = self.metadata_df.reset_index(drop=True)

        # Build group dictionary
        self.groups = {}
        for group_name, group_df in self.metadata_df.groupby(self.group_column):
            self.groups[group_name] = {
                "indices": group_df.index.tolist(),
                "size": len(group_df),
            }

        print(f"\nGroups retained: {len(self.groups)}")
        print(f"Samples after filtering: {len(self.embeddings)}")
        for group_name, info in self.groups.items():
            print(f"  {group_name}: n={info['size']}")

    def compute_group_overlap(
        self,
        baseline_proj: np.ndarray,
        perturbed_proj: np.ndarray,
        group_name: str,
    ) -> float:
        """Compute k-NN overlap for a specific group.

        Args:
            baseline_proj: Baseline projection (N, 2)
            perturbed_proj: Perturbed projection (N, 2)
            group_name: Name of the group

        Returns:
            k-NN overlap score
        """
        indices = self.groups[group_name]["indices"]
        group_size = self.groups[group_name]["size"]

        # Extract group projections
        baseline_group = baseline_proj[indices]
        perturbed_group = perturbed_proj[indices]

        # Adaptive k
        k_actual = min(self.knn_k, group_size - 1)

        # Compute overlap
        overlap = calculate_knn_overlap(
            baseline_group, perturbed_group, n_neighbors=k_actual
        )

        return overlap

    def run_method_for_groups(
        self, method: str, experiment_type: str = "seed"
    ) -> dict[str, float]:
        """Run a DR method and compute per-group robustness.

        Args:
            method: DR method name
            experiment_type: Type of robustness test ('seed' or hyperparameter name)

        Returns:
            Dictionary mapping group names to mean robustness scores
        """
        print(f"\n{'='*70}")
        print(f"Running {method.upper()} - {experiment_type} robustness")
        print(f"{'='*70}")

        config = METHOD_CONFIGS[method]
        baseline_config = config["baseline"]

        # Run baseline
        print("Running baseline...")
        baseline_dr_config = DimensionReductionConfig(**baseline_config)
        baseline_result = benchmark_method(
            embeddings=self.embeddings,
            method=method,
            config=baseline_dr_config,
            normalize=False,
            metric_functions=None,
        )
        baseline_proj = baseline_result.projection
        print(f"  Baseline runtime: {baseline_result.time_seconds:.3f}s")

        # Collect group overlaps across experiments
        group_overlaps = {group_name: [] for group_name in self.groups.keys()}

        if experiment_type == "seed":
            seed_values = config["seed_values"]
            print(f"Testing {len(seed_values)} seeds: {seed_values}")

            for seed in seed_values:
                config_dict = baseline_config.copy()
                config_dict["random_state"] = seed

                dr_config = DimensionReductionConfig(**config_dict)
                result = benchmark_method(
                    embeddings=self.embeddings,
                    method=method,
                    config=dr_config,
                    normalize=False,
                    metric_functions=None,
                )

                # Compute overlap for each group
                for group_name in self.groups.keys():
                    overlap = self.compute_group_overlap(
                        baseline_proj, result.projection, group_name
                    )
                    group_overlaps[group_name].append(overlap)

                print(f"  Seed {seed}: done ({result.time_seconds:.2f}s)")

        else:
            # Hyperparameter experiment
            param_values = config["hyperparam_experiments"].get(experiment_type, [])
            if not param_values:
                print(f"  No experiments defined for {experiment_type}")
                return {group: 0.0 for group in self.groups.keys()}

            print(f"Testing {len(param_values)} values for {experiment_type}: {param_values}")

            for value in param_values:
                config_dict = baseline_config.copy()
                config_dict[experiment_type] = value

                dr_config = DimensionReductionConfig(**config_dict)
                result = benchmark_method(
                    embeddings=self.embeddings,
                    method=method,
                    config=dr_config,
                    normalize=False,
                    metric_functions=None,
                )

                # Compute overlap for each group
                for group_name in self.groups.keys():
                    overlap = self.compute_group_overlap(
                        baseline_proj, result.projection, group_name
                    )
                    group_overlaps[group_name].append(overlap)

                print(f"  {experiment_type}={value}: done ({result.time_seconds:.2f}s)")

        # Average across experiments
        group_mean_overlaps = {
            group: np.mean(overlaps) for group, overlaps in group_overlaps.items()
        }

        print("\nGroup robustness scores:")
        for group, score in sorted(group_mean_overlaps.items(), key=lambda x: x[1]):
            print(f"  {group:30s}: {score:.4f}")

        return group_mean_overlaps

    def run_all_methods(
        self, methods: list[str] | None = None, experiment_types: list[str] | None = None
    ) -> pd.DataFrame:
        """Run all DR methods and collect group-level results.

        Args:
            methods: List of methods to run (default: all)
            experiment_types: Types of experiments to run (default: ['seed'])

        Returns:
            DataFrame with columns: method, experiment_type, group, robustness, group_size
        """
        if methods is None:
            methods = get_all_methods()

        if experiment_types is None:
            experiment_types = ["seed"]

        results = []

        for method in methods:
            for exp_type in experiment_types:
                try:
                    group_scores = self.run_method_for_groups(method, exp_type)

                    for group_name, score in group_scores.items():
                        results.append(
                            {
                                "method": method,
                                "experiment_type": exp_type,
                                "group": group_name,
                                "robustness": score,
                                "group_size": self.groups[group_name]["size"],
                            }
                        )
                except Exception as e:
                    print(f"\nError running {method} - {exp_type}: {e}")
                    continue

        return pd.DataFrame(results)

    def save_results(self, results_df: pd.DataFrame) -> None:
        """Save results to CSV files.

        Args:
            results_df: Results DataFrame
        """
        if not self.output_dir:
            print("No output directory specified, skipping save")
            return

        # Save detailed results
        results_path = self.output_dir / "group_dr_robustness_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Saved detailed results: {results_path}")

        # Save method summary
        method_summary = (
            results_df.groupby(["method", "experiment_type"])["robustness"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        method_path = self.output_dir / "method_summary.csv"
        method_summary.to_csv(method_path, index=False)
        print(f"✓ Saved method summary: {method_path}")

        # Save group summary
        group_summary = (
            results_df.groupby(["group", "experiment_type"])["robustness"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        group_path = self.output_dir / "group_summary.csv"
        group_summary.to_csv(group_path, index=False)
        print(f"✓ Saved group summary: {group_path}")

    def print_summary(self, results_df: pd.DataFrame) -> None:
        """Print summary statistics.

        Args:
            results_df: Results DataFrame
        """
        print("\n" + "=" * 70)
        print("SUMMARY: DR METHOD ROBUSTNESS BY GROUP")
        print("=" * 70)

        # By method
        print("\n### By Method ###")
        method_summary = (
            results_df.groupby("method")["robustness"]
            .agg(["mean", "std", "min", "max"])
            .sort_values("mean", ascending=False)
        )
        print(method_summary)

        # By group
        print("\n### By Group ###")
        group_summary = (
            results_df.groupby("group")["robustness"]
            .agg(["mean", "std", "min", "max"])
            .sort_values("mean", ascending=False)
        )
        print(group_summary)

        # Find best/worst combinations
        print("\n### Top 5 (Method, Group) Combinations ###")
        top_5 = results_df.nlargest(5, "robustness")[
            ["method", "group", "robustness", "group_size"]
        ]
        print(top_5.to_string(index=False))

        print("\n### Bottom 5 (Method, Group) Combinations ###")
        bottom_5 = results_df.nsmallest(5, "robustness")[
            ["method", "group", "robustness", "group_size"]
        ]
        print(bottom_5.to_string(index=False))

        print("\n" + "=" * 70)
