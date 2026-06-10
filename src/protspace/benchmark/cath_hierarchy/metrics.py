"""k-NN Hierarchy Purity (KHP) metric for CATH hierarchy evaluation.

For a given 2D projection, KHP measures — at each CATH level — how often a
protein's k nearest neighbours share the same label as itself.

Observed ordering for any projection (including random):
    KHP(class) > KHP(architecture) > KHP(topology) > KHP(homology)

This ordering is driven by group size: the 5 CATH classes each contain
hundreds of proteins, so even a random projection yields high KHP(class)
(~0.36 baseline), while the 1 163 homology superfamilies average only ~2.6
members in the S40 dataset, giving a near-zero random baseline (~0.004).
The ordering therefore reflects the dataset's label-size distribution, not
projection quality.  What is meaningful is how far each method's KHP sits
*above* its random baseline, and how consistently the projection clusters
proteins at every level — especially the fine ones (topology, homology).

This module reuses the :class:`sklearn.neighbors.NearestNeighbors` pattern
already established in ``src/protspace/benchmark/metrics.py``.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors

from protspace.benchmark.cath_hierarchy.labels import CATHLabels

logger = logging.getLogger(__name__)

# Human-readable names for the four CATH levels (coarse → fine)
CATH_LEVELS: list[tuple[str, str]] = [
    ("cath_class", "Class (C)"),
    ("architecture", "Architecture (A)"),
    ("topology", "Topology (T)"),
    ("homology", "Homology (H)"),
]


def compute_khp_baselines(labels: CATHLabels) -> dict[str, float]:
    """Expected KHP under a uniformly random projection.

    For a protein in group g of size s_g, the probability that a randomly
    chosen *other* protein also belongs to g is ``(s_g - 1) / (N - 1)``.
    Averaging over all valid proteins gives the random baseline for each level.

    The baseline is independent of k because each neighbour is drawn
    independently and uniformly from the remaining N-1 proteins.

    Returns
    -------
    dict with keys ``"baseline_cath_class"``, ``"baseline_architecture"``,
    ``"baseline_topology"``, ``"baseline_homology"``.
    """
    N = int(labels.valid_mask.sum())
    level_arrays = {
        "cath_class": labels.cath_class,
        "architecture": labels.architecture,
        "topology": labels.topology,
        "homology": labels.homology,
    }
    baselines: dict[str, float] = {}
    for level_key, arr in level_arrays.items():
        valid_labels = arr[labels.valid_mask]
        counts = Counter(valid_labels.tolist())
        per_protein = np.array([(counts[v] - 1) / (N - 1) for v in valid_labels])
        baselines[f"baseline_{level_key}"] = float(per_protein.mean())
    return baselines


def knn_hierarchy_purity(
    projection: np.ndarray,
    labels: CATHLabels,
    k: int = 15,
) -> dict[str, float]:
    """Compute k-NN Hierarchy Purity at all four CATH levels.

    For every protein that has a valid CATH label, find its ``k`` nearest
    neighbours in the 2D projection (by Euclidean distance, excluding self).
    The purity at a given level is the fraction of those neighbours that share
    the same label at that level, averaged over all valid proteins.

    Parameters
    ----------
    projection:
        2D coordinates, shape ``(n_proteins, 2)``.
    labels:
        :class:`CATHLabels` aligned with ``projection``.
    k:
        Number of nearest neighbours to consider.

    Returns
    -------
    dict with keys ``"khp_class"``, ``"khp_architecture"``,
    ``"khp_topology"``, ``"khp_homology"`` (float in [0, 1] or NaN).
    Also includes ``"k"`` and ``"n_valid"`` for provenance.
    """
    if projection.shape[0] != len(labels.identifiers):
        raise ValueError(
            f"projection has {projection.shape[0]} rows but labels has "
            f"{len(labels.identifiers)} entries"
        )

    valid_idx = np.where(labels.valid_mask)[0]
    n_valid = len(valid_idx)

    if n_valid < k + 1:
        logger.warning(
            "Only %d valid proteins, but k=%d. Reducing k to %d.",
            n_valid,
            k,
            max(1, n_valid - 1),
        )
        k = max(1, n_valid - 1)

    coords_valid = projection[valid_idx]

    # Build one k-NN index for all valid proteins
    actual_k = min(k + 1, n_valid)  # +1 to exclude self
    nbrs = NearestNeighbors(n_neighbors=actual_k, metric="euclidean", algorithm="auto")
    nbrs.fit(coords_valid)
    _, indices = nbrs.kneighbors(coords_valid)
    # indices shape: (n_valid, actual_k) — first column is self
    neighbor_indices = indices[:, 1:]  # drop self, shape (n_valid, k)

    results: dict[str, float] = {"k": float(k), "n_valid": float(n_valid)}

    level_map = {
        "cath_class": labels.cath_class[valid_idx],
        "architecture": labels.architecture[valid_idx],
        "topology": labels.topology[valid_idx],
        "homology": labels.homology[valid_idx],
    }

    for level_key, level_labels in level_map.items():
        purities: list[float] = []
        for i in range(n_valid):
            own_label = level_labels[i]
            if not own_label:
                continue
            neighbor_labels = level_labels[neighbor_indices[i]]
            purity = (
                float(np.sum(neighbor_labels == own_label)) / neighbor_indices.shape[1]
            )
            purities.append(purity)

        metric_key = f"khp_{level_key}"
        results[metric_key] = float(np.mean(purities)) if purities else float("nan")
        logger.debug(
            "KHP @ %-15s k=%2d  purity=%.4f  (n=%d)",
            level_key,
            k,
            results[metric_key],
            len(purities),
        )

    return results


# ---------------------------------------------------------------------------
# Harness-compatible factory functions
# ---------------------------------------------------------------------------
# The benchmark harness expects callables with signature:
#   (embeddings: np.ndarray, projection: np.ndarray) -> float
# We close over `labels` and `k` to create thin wrappers.


def make_khp_metrics(
    labels: CATHLabels, k: int = 15
) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """Return a dict of harness-compatible KHP metric functions.

    Keys: ``"khp_homology"``, ``"khp_topology"``, ``"khp_architecture"``,
    ``"khp_class"``.

    Each callable has the signature ``(embeddings, projection) -> float``
    so it slots into :func:`protspace.benchmark.harness.benchmark_methods`.
    """

    def _make_fn(level_key: str) -> Callable[[np.ndarray, np.ndarray], float]:
        metric_key = f"khp_{level_key}"

        def metric_fn(embeddings: np.ndarray, projection: np.ndarray) -> float:  # noqa: ARG001
            return knn_hierarchy_purity(projection, labels, k=k)[metric_key]

        metric_fn.__name__ = metric_key
        return metric_fn

    return {
        "khp_homology": _make_fn("homology"),
        "khp_topology": _make_fn("topology"),
        "khp_architecture": _make_fn("architecture"),
        "khp_class": _make_fn("cath_class"),
    }
