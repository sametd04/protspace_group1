#!/usr/bin/env python3
"""Minimal benchmark script for protein datasets.

Runs a configured set of DR methods, computes timing and quality metrics
(trustworthiness, silhouette, concordex), and writes results to
``results/<DATA>/``.

Usage:
    python run.py
    DATA=globin python run.py     # override dataset via env var
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from protspace.benchmark import benchmark_methods
from protspace.benchmark.labels import label_summary, load_labels_from_bundle
from protspace.benchmark.metrics import (
    calculate_trustworthiness,
    make_concordex_metric,
    make_silhouette_metric,
)
from protspace.data.loaders import load_h5
from protspace.utils.constants import DimensionReductionConfig

DATA = os.environ.get("DATA", "3ftx")

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EMBEDDING_PATH = PROJECT_ROOT / f"output_{DATA}" / "tmp" / "prot_t5.h5"
BUNDLE_PATH = PROJECT_ROOT / f"output_{DATA}" / "data.parquetbundle"

METHODS = ["pca", "umap", "tsne", "pacmap", "mds", "localmap"]
OUTPUT_DIR = Path(__file__).parent / "results" / DATA

# Concordex configuration. The number of permutations trades off cost
# against null-mean precision; 15 is the default in the concordexR
# package and is comfortable for n in the low thousands.
CONCORDEX_N_NEIGHBORS = 15
CONCORDEX_N_PERMUTATIONS = 15
CONCORDEX_RANDOM_STATE = 42


def main():
    print(f"=== Benchmark on '{DATA}' ===\n")
    print(f"Loading embeddings from {EMBEDDING_PATH}")

    if not EMBEDDING_PATH.exists():
        sys.exit(
            f"Embeddings not found at {EMBEDDING_PATH}.\n"
            f"Generate with: protspace prepare -q '<query>' "
            f"-e prot_t5 -m pca2,umap2 -o output_{DATA}"
        )

    emb_set = load_h5([EMBEDDING_PATH])
    embeddings = emb_set.data
    headers = emb_set.headers
    print(
        f"Loaded {embeddings.shape[0]} proteins with "
        f"{embeddings.shape[1]} features\n"
    )

    # Labels are required for silhouette and concordex; both are skipped
    # automatically if the bundle is missing.
    labels = None
    if BUNDLE_PATH.exists():
        labels = load_labels_from_bundle(BUNDLE_PATH, headers)
        summary = label_summary(labels)
        print(
            f"Loaded labels: {summary['n_labelled']}/{summary['n_total']} "
            f"proteins, {summary['n_classes']} classes"
        )
        print("Top classes:")
        for cls, n in list(summary["classes"].items())[:5]:
            print(f"  {n:4d}  {cls}")
        print()
    else:
        print(
            f"[warn] No bundle at {BUNDLE_PATH} — silhouette and concordex "
            f"will be NaN\n"
        )

    config = DimensionReductionConfig(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        perplexity=min(30, embeddings.shape[0] // 4),
    )

    # Build the metrics dict. Both label-based metrics use closures that
    # capture ``labels`` so they conform to the harness signature
    # ``(embeddings, projection) -> float``.
    metric_functions: dict[str, callable] = {
        "trustworthiness": calculate_trustworthiness,
    }
    if labels is not None:
        metric_functions["silhouette"] = make_silhouette_metric(labels)
        metric_functions["concordex"] = make_concordex_metric(
            labels,
            n_neighbors=CONCORDEX_N_NEIGHBORS,
            n_permutations=CONCORDEX_N_PERMUTATIONS,
            random_state=CONCORDEX_RANDOM_STATE,
        )

    print(f"Benchmarking methods: {', '.join(METHODS)}")
    print(f"Metrics: {', '.join(metric_functions.keys())}\n")
    print("=" * 70)

    results = benchmark_methods(
        embeddings=embeddings,
        methods=METHODS,
        config=config,
        normalize=True,
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

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    for method, result in results.items():
        proj_file = OUTPUT_DIR / f"{method}_projection.npy"
        np.save(proj_file, result.projection)

    metrics_data = []
    for method, result in results.items():
        row = {"method": method, "runtime_seconds": result.time_seconds}
        row.update(result.metrics)
        metrics_data.append(row)

    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv = OUTPUT_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    np.save(OUTPUT_DIR / "headers.npy", np.array(headers, dtype=object))

    print(f"\nProjections saved to {OUTPUT_DIR}/")
    print(f"Metrics saved to {metrics_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()