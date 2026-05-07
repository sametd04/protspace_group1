#!/usr/bin/env python3
"""Benchmark pipeline orchestration.

Orchestrates the complete benchmark workflow:
1. Load embeddings
2. Load labels (if available)
3. Run all DR methods with metrics
4. Save results (CSV metrics + parquet bundle)

For CLI usage, see cli.py.
For visualization, see visualize.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa

from protspace.benchmark import benchmark_methods
from protspace.benchmark.io import BenchmarkPaths, benchmark_paths
from protspace.benchmark.labels import label_summary, load_silhouette_labels
from protspace.benchmark.metrics import default_metric_functions
from protspace.data.io.bundle import write_bundle
from protspace.data.loaders import load_h5
from protspace.utils.constants import DimensionReductionConfig

# Suppress numerical precision warnings from sklearn
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

# Benchmark configuration
METHODS = ["pca", "umap", "tsne", "pacmap", "mds", "localmap"]

# For visualize.py
METHOD_TITLES = {
    "pca": "PCA",
    "umap": "UMAP",
    "tsne": "t-SNE",
    "pacmap": "PaCMAP",
    "mds": "MDS",
    "localmap": "LocalMAP",
}
DEFAULT_DATASET = "3ftx"
DATASET_CHOICES = ("3ftx", "toxprot", "pla2g2", "cath_s40", "swissprot_rr")


def run_benchmark(paths: BenchmarkPaths) -> None:
    print(f"=== Benchmark on '{paths.data}' ===\n")
    print(f"Loading embeddings from {paths.embedding_path}")

    if not paths.embedding_path.exists():
        sys.exit(
            f"Embeddings not found at {paths.embedding_path}.\n"
            f"Generate with: protspace prepare -q '<query>' "
            f"-e prot_t5 -m pca2,umap2 -o output_{paths.data}"
        )

    emb_set = load_h5([paths.embedding_path])
    embeddings = emb_set.data
    headers = emb_set.headers

    print(
        f"Loaded {embeddings.shape[0]} proteins with {embeddings.shape[1]} features\n"
    )

    labels = load_silhouette_labels(paths, headers)
    if labels is not None:
        summary = label_summary(labels)
        print(
            f"Labels: {summary['n_labelled']}/{summary['n_total']} proteins, "
            f"{summary['n_classes']} classes\n"
        )

    config = DimensionReductionConfig(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        perplexity=min(30, embeddings.shape[0] // 4),
    )

    metric_functions = default_metric_functions(labels)

    print(f"Benchmarking methods: {', '.join(METHODS)}")
    print(f"Metrics: {', '.join(metric_functions.keys())}\n")
    print("=" * 70)

    # Run with normalization
    results = benchmark_methods(
        embeddings=embeddings,
        methods=METHODS,
        config=config,
        normalize=True,
        metric_functions=metric_functions,
    )

    # Run without normalization for comparison
    print("\nRunning without normalization for comparison...")
    _results_raw = benchmark_methods(
        embeddings=embeddings,
        methods=METHODS,
        config=config,
        normalize=False,
        metric_functions=metric_functions,
    )

    print("\nRESULTS:")
    print("=" * 70)
    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Runtime: {result.time_seconds:.3f}s")
        print(f"  Shape: {result.projection.shape}")
        if result.metrics:
            print("  Metrics:")
            for name, value in result.metrics.items():
                if not np.isnan(value):
                    print(f"    {name}: {value:.6f}")
                else:
                    print(f"    {name}: NaN")

    paths.output_dir.mkdir(exist_ok=True, parents=True)

    # Save metrics as CSV
    metrics_data = []
    for method, result in results.items():
        row = {"method": method, "runtime_seconds": result.time_seconds}
        row.update(result.metrics)
        metrics_data.append(row)

    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv = paths.output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # # Save raw projections as .npy for visualization
    # for method, result in results.items():
    #     np.save(paths.output_dir / f"{method}_projection.npy", result.projection)

    # Create ProtSpace bundle for web visualization
    print("\nCreating ProtSpace bundle for visualization...")

    # Load or create annotations table
    annotations_table = None
    if paths.bundle_path.exists():
        print(f"Loading annotations from {paths.bundle_path}")
        try:
            import io

            import pyarrow.parquet as pq

            from protspace.data.io.bundle import read_bundle

            parts_bytes, _ = read_bundle(paths.bundle_path)
            if len(parts_bytes) > 0 and parts_bytes[0]:
                # Parse first part (annotations) from bytes
                annotations_table = pq.read_table(io.BytesIO(parts_bytes[0]))
                print(
                    f"Annotations table: {len(annotations_table)} rows, {len(annotations_table.column_names)} columns"
                )
        except Exception as e:
            print(f"[warn] Could not load existing bundle: {e}")

    # Create minimal annotations table if none exists
    if annotations_table is None:
        print("Creating minimal annotations table from headers")
        annotations_df = pd.DataFrame(
            {
                "protein_id": headers,
                "identifier": headers,
            }
        )
        annotations_table = pa.Table.from_pandas(annotations_df)
        print(
            f"Annotations table: {len(annotations_table)} rows, {len(annotations_table.column_names)} columns"
        )

    # Create projections metadata table
    metadata_rows = []
    for method, result in results.items():
        metadata_rows.append(
            {
                "projection_name": f"{method.upper()}_benchmark",
                "dimensions": 2,
                "info_json": json.dumps(result.params),
            }
        )
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_table = pa.Table.from_pandas(metadata_df)
    print(f"Metadata table: {len(metadata_table)} rows")

    # Create projections data table
    data_rows = []
    for method, result in results.items():
        proj_name = f"{method.upper()}_benchmark"
        for i, header in enumerate(headers):
            data_rows.append(
                {
                    "projection_name": proj_name,
                    "identifier": header,
                    "x": np.float32(result.projection[i][0]),
                    "y": np.float32(result.projection[i][1]),
                    "z": None,
                }
            )
    data_df = pd.DataFrame(data_rows)
    data_table = pa.Table.from_pandas(data_df)
    print(f"Data table: {len(data_table)} rows")

    # Write bundle
    bundle_path = paths.output_dir / "benchmark.parquetbundle"
    print(f"Writing bundle to {bundle_path}...")
    write_bundle([annotations_table, metadata_table, data_table], bundle_path)
    print("Bundle written successfully!")

    print(f"\nProjections saved to {paths.output_dir}/")
    print(f"Metrics saved to {metrics_csv}")
    print(f"\nVisualize with: protspace serve {bundle_path}")
    print("=" * 70)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DR benchmark for one or multiple datasets."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=DATASET_CHOICES,
        help=f"Dataset to benchmark (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run benchmark for all configured datasets sequentially.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    datasets = DATASET_CHOICES if args.all_datasets else (args.dataset,)
    for index, dataset in enumerate(datasets):
        if index > 0:
            print("\n" + "#" * 90 + "\n")
        run_benchmark(benchmark_paths(dataset))


if __name__ == "__main__":
    main()
