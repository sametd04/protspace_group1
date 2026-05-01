#!/usr/bin/env python3
"""Minimal benchmark script for protein datasets.

Runs PCA, UMAP, and t-SNE benchmarks on specified dataset.

Usage:
    python run_benchmark_real_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from protspace.benchmark import benchmark_methods
from protspace.benchmark.metrics import calculate_trustworthiness
from protspace.data.loaders import load_h5
from protspace.utils.constants import DimensionReductionConfig

# ===== Configuration =====
# Dataset name (used for organizing results)
DATA = "3ftx"

# Fixed configuration - paths relative to project root
# File is at: src/protspace/benchmark/run_benchmark_real_data.py
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EMBEDDING_PATH = PROJECT_ROOT / f"output_{DATA}" / "tmp" / "prot_t5.h5"
METHODS = ["pca", "umap", "tsne", "pacmap", "mds", "localmap"]
# Save results in benchmark/results/$DATA/ directory
OUTPUT_DIR = Path(__file__).parent / "results" / DATA


def main():
    print("Loading embeddings from", EMBEDDING_PATH)

    # Load using existing protspace loader
    emb_set = load_h5([EMBEDDING_PATH])
    embeddings = emb_set.data

    print(
        f"Loaded {embeddings.shape[0]} proteins with {embeddings.shape[1]} features\n"
    )

    # Configure DR
    config = DimensionReductionConfig(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        perplexity=min(30, embeddings.shape[0] // 4),
    )

    # Use only implemented metrics (avoid NotImplementedError)
    metric_functions = {
        "trustworthiness": calculate_trustworthiness,
    }

    print(f"Benchmarking methods: {', '.join(METHODS)}")
    print(f"Metrics: {', '.join(metric_functions.keys())}\n")
    print("=" * 70)

    # Run benchmark using existing harness
    results = benchmark_methods(
        embeddings=embeddings,
        methods=METHODS,
        config=config,
        normalize=True,
        metric_functions=metric_functions,
    )

    # Print results
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

    # Save projections and metrics
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Save projections as .npy files
    for method, result in results.items():
        proj_file = OUTPUT_DIR / f"{method}_projection.npy"
        np.save(proj_file, result.projection)

    # Save metrics as CSV
    metrics_data = []
    for method, result in results.items():
        row = {"method": method, "runtime_seconds": result.time_seconds}
        row.update(result.metrics)
        metrics_data.append(row)

    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv = OUTPUT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    print(f"\nProjections saved to {OUTPUT_DIR}/")
    print(f"Metrics saved to {metrics_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
