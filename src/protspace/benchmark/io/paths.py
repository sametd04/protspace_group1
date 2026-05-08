"""Filesystem layout configuration for benchmark datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkPaths:
    """Filesystem layout for one dataset.

    Defines all paths needed for benchmark execution and visualization.
    """

    data: str
    project_root: Path
    dataset_dir: Path
    embedding_path: Path
    bundle_path: Path
    results_dir: Path
    output_dir: Path
    output_png: Path
    benchmark_bundle_path: Path

    @property
    def headers_npy(self) -> Path:
        """Path to cached headers file (fallback if embeddings are deleted)."""
        return self.output_dir / "headers.npy"


def benchmark_paths(data: str | None = None) -> BenchmarkPaths:
    """Construct paths for a benchmark dataset.

    Parameters
    ----------
    data
        Dataset name. Falls back to environment variable DATA or "3ftx".

    Returns
    -------
    BenchmarkPaths with all necessary filesystem locations.
    """
    d = (data or os.environ.get("DATA", "3ftx")).strip()
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    data_dir = project_root / "data"
    results_dir = Path(__file__).resolve().parent.parent / "results" / d

    dataset_dir = data_dir / d
    if not dataset_dir.exists() and data_dir.exists():
        # Fallback for case variants (e.g., 3FTx)
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.lower() == d.lower():
                dataset_dir = item
                break

    benchmark_bundle_path = results_dir / "benchmark.parquetbundle"

    # Look for source labels in data/<dataset>, with a fallback to an existing
    # benchmark bundle in results/<dataset>.
    bundle_path = None
    if (dataset_dir / "parquetbundle").exists():
        # Find first .parquetbundle file in parquetbundle/ directory
        bundle_files = list((dataset_dir / "parquetbundle").glob("*.parquetbundle"))
        if bundle_files:
            bundle_path = bundle_files[0]
    elif (dataset_dir / "data.parquetbundle").exists():
        bundle_path = dataset_dir / "data.parquetbundle"

    if bundle_path is None and benchmark_bundle_path.exists():
        bundle_path = benchmark_bundle_path
    elif bundle_path is None:
        # Default fallback
        bundle_path = dataset_dir / "data.parquetbundle"

    return BenchmarkPaths(
        data=d,
        project_root=project_root,
        dataset_dir=dataset_dir,
        embedding_path=dataset_dir / "prot_t5.h5",
        bundle_path=bundle_path,
        results_dir=results_dir,
        output_dir=results_dir,
        output_png=results_dir / f"projections_{d}.png",
        benchmark_bundle_path=benchmark_bundle_path,
    )
