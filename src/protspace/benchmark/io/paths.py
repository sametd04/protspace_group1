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
    embedding_path: Path
    bundle_path: Path
    output_dir: Path
    output_png: Path

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
    out = Path(__file__).resolve().parent.parent / "results" / d

    # Look for data in data/{dataset}/ directory
    # Dataset name might be capitalized (e.g., 3FTx vs 3ftx)
    data_dir = project_root / "data"
    dataset_dir = None

    # Try exact match first, then case-insensitive
    if (data_dir / d).exists():
        dataset_dir = data_dir / d
    else:
        # Try to find case-insensitive match
        for item in data_dir.iterdir():
            if item.is_dir() and item.name.lower() == d.lower():
                dataset_dir = item
                break

    # Fallback to output_{d} if data/{d} doesn't exist
    if dataset_dir is None:
        dataset_dir = project_root / f"output_{d}"

    # Look for bundle in parquetbundle subdirectory or directly
    bundle_path = None
    if (dataset_dir / "parquetbundle").exists():
        # Find first .parquetbundle file in parquetbundle/ directory
        bundle_files = list((dataset_dir / "parquetbundle").glob("*.parquetbundle"))
        if bundle_files:
            bundle_path = bundle_files[0]
    elif (dataset_dir / "data.parquetbundle").exists():
        bundle_path = dataset_dir / "data.parquetbundle"

    if bundle_path is None:
        # Default fallback
        bundle_path = dataset_dir / "data.parquetbundle"

    return BenchmarkPaths(
        data=d,
        project_root=project_root,
        embedding_path=dataset_dir / "tmp" / "prot_t5.h5",
        bundle_path=bundle_path,
        output_dir=out,
        output_png=out / f"projections_{d}.png",
    )
