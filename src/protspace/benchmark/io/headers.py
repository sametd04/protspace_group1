"""Protein ID (header) loading with fallback logic."""

from __future__ import annotations

import numpy as np

from protspace.benchmark.io.paths import BenchmarkPaths
from protspace.data.loaders import load_h5


def resolve_headers(paths: BenchmarkPaths) -> list[str]:
    """Get protein IDs for projections.

    Tries multiple sources in order:
    1. Load from embedding H5 file (primary source)
    2. Load from cached headers.npy (fallback if embeddings deleted)
    3. Raise error if neither exists

    Parameters
    ----------
    paths
        Benchmark paths configuration.

    Returns
    -------
    List of protein IDs in embedding order.

    Raises
    ------
    FileNotFoundError
        If neither embeddings nor cached headers are available.
    """
    if paths.embedding_path.exists():
        return load_h5([paths.embedding_path]).headers

    if paths.headers_npy.exists():
        return list(np.load(paths.headers_npy, allow_pickle=True))

    raise FileNotFoundError(
        f"No embeddings at {paths.embedding_path} and no {paths.headers_npy}"
    )
