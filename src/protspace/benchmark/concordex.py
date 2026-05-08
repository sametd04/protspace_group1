"""Concordex metric for dimensionality-reduction evaluation.

Implementation of the neighborhood concordance score introduced by
Jackson et al., "Characterization of spatial homogeneous regions in tissues
with concordex" (bioRxiv 2023, doi:10.1101/2023.06.28.546949).

The paper develops concordex in the context of spatial transcriptomics, where
a kNN graph is built from spatial coordinates and a categorical label per
cell (cell type) is provided. Two objects are central:

* The **neighborhood consolidation matrix** ``K`` of shape ``(n, L)``.
  ``K[i, j]`` is the fraction of node ``i``'s ``k`` nearest neighbors that
  carry label ``j``. Each row therefore sums to ``1``.

* The **aggregated label matrix** ``D`` of shape ``(L, L)``. ``D[a, b]`` is
  the mean of ``K[i, b]`` over all ``i`` with label ``a``. Its diagonal is
  the average within-class neighborhood concordance per class.

For DR evaluation we re-purpose the same construction by building the kNN
graph on the **2D projection** instead of spatial coordinates. The question
becomes: in the projected space, do points of the same functional class
sit in label-coherent neighborhoods, beyond what would happen by chance?

The scalar score is the trace ratio against a permutation null:

.. math::
   \\text{concordex} = \\frac{\\operatorname{tr}(D_{\\text{obs}})}{
        \\mathbb{E}\\bigl[\\operatorname{tr}(D_{\\pi})\\bigr]}

where ``\\pi`` is a uniform random permutation of the labels. A value of 1
indicates a chance-level layout; values greater than 1 indicate that
same-label points sit in coherent neighborhoods more than chance predicts.
The ratio is invariant to class imbalance because the permutation null is
computed on the same label distribution.

The harness signature is ``(embeddings, projection) -> float``. Labels are
bound into a closure via :func:`make_concordex_metric`, mirroring the
silhouette wiring already in the codebase.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _valid_label_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of labels that are non-None, non-NaN, non-empty.

    Mirrors the convention used by :func:`calculate_silhouette_score` so
    that downstream filtering is consistent across label-based metrics.
    """
    return np.array(
        [
            lbl is not None
            and not (isinstance(lbl, float) and np.isnan(lbl))
            and str(lbl).strip() != ""
            for lbl in labels
        ]
    )


def _build_knn_indices(
    X: np.ndarray, n_neighbors: int, metric: str
) -> tuple[np.ndarray, int]:
    """Return (indices, k) where ``indices`` is the ``(n, k)`` neighbor
    matrix excluding self-loops, and ``k`` is the actually-used neighbor
    count (capped at ``n - 1``)."""
    n = X.shape[0]
    k = min(n_neighbors, n - 1)
    if k < 1:
        raise ValueError("Need at least 2 samples to build a kNN graph.")

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    indices = nn.kneighbors(return_distance=False)[:, 1:]  # drop self
    return indices, k


def _consolidation_from_indices(
    indices: np.ndarray, labels: np.ndarray, classes: np.ndarray, k: int
) -> np.ndarray:
    """Compute the consolidation matrix ``K`` (n, L) given a fixed kNN
    index matrix and a label vector. Vectorised over classes."""
    neighbor_labels = labels[indices]  # (n, k)
    K = np.zeros((indices.shape[0], len(classes)), dtype=float)
    for j, cls in enumerate(classes):
        K[:, j] = (neighbor_labels == cls).sum(axis=1) / k
    return K


def _aggregated_matrix(
    K: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Aggregate ``K`` (n, L) into ``D`` (L, L), where ``D[a, b]`` is the
    mean fraction of label-``b`` neighbors among points of label ``a``."""
    L = len(classes)
    D = np.zeros((L, L), dtype=float)
    for a, cls in enumerate(classes):
        mask = labels == cls
        if mask.any():
            D[a] = K[mask].mean(axis=0)
    return D

def neighborhood_consolidation_matrix(
    X: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the neighborhood consolidation matrix ``K``.

    Parameters
    ----------
    X
        Coordinates used to construct the kNN graph. For DR evaluation
        this is the 2D projection; in the original paper it is the
        spatial coordinate matrix.
    labels
        Categorical labels aligned with the rows of ``X``.
    n_neighbors
        Number of neighbors per node, excluding self-loops.
    metric
        Distance metric for kNN.

    Returns
    -------
    K
        ``(n, L)`` consolidation matrix. ``K[i, j]`` is the fraction of
        ``i``'s neighbors that carry the ``j``-th class label.
    classes
        ``(L,)`` array of unique class labels, in the order corresponding
        to the columns of ``K``.
    """
    indices, k = _build_knn_indices(X, n_neighbors, metric)
    classes = np.unique(labels)
    K = _consolidation_from_indices(indices, labels, classes, k)
    return K, classes


def aggregated_label_matrix(
    K: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Aggregate the consolidation matrix to the per-class summary ``D``.

    ``D[a, b]`` is the mean fraction of label-``b`` neighbors for points
    of label ``a``. The diagonal ``D[a, a]`` is the within-class
    neighborhood concordance for class ``a``.
    """
    return _aggregated_matrix(K, labels, classes)


def calculate_concordex_score(
    embeddings: np.ndarray,  # kept for harness compatibility
    projection: np.ndarray,
    labels: np.ndarray | None = None,
    n_neighbors: int = 15,
    n_permutations: int = 15,
    random_state: int = 42,
    metric: str = "euclidean",
) -> float:
    """Concordex score on the projection coordinates.

    Computes the ratio of the observed trace of the aggregated label
    matrix ``D`` to its expectation under random label permutation. The
    kNN graph is constructed once on the projection coordinates and
    re-used across permutations: only the labels that decorate the graph
    change, so the null is a label-permutation null rather than a
    coordinate-permutation null.

    Parameters
    ----------
    embeddings
        Unused. Present so the function matches the harness signature
        ``(embeddings, projection) -> float``.
    projection
        ``(n, d)`` low-dimensional coordinates. Typically ``d = 2``.
    labels
        Categorical labels aligned with the rows of ``projection``.
        ``None`` returns ``NaN``, mirroring the silhouette implementation.
    n_neighbors
        Number of neighbors per node in the kNN graph.
    n_permutations
        Number of label permutations used to estimate the null mean.
    random_state
        Seed for the permutation RNG; ensures reproducibility.
    metric
        Distance metric for kNN. Defaults to euclidean, matching the
        rest of the harness.

    Returns
    -------
    float
        ``concordex = trace(D_obs) / E[trace(D_perm)]``.
        ``= 1``: chance-level layout.
        ``> 1``: same-label points cluster (the projection respects the
        label structure).
        ``< 1``: anti-clustering. ``NaN`` if labels are unusable or
        fewer than two classes survive filtering.

    Notes
    -----
    The trace formulation handles class imbalance naturally because the
    null is computed on the same label distribution. We do not divide
    ``trace(D)`` by ``L`` (number of classes) since the ratio cancels
    that factor; the absolute trace is reported only as an intermediate.
    """
    if labels is None:
        return float("nan")

    if projection.shape[0] != labels.shape[0]:
        raise ValueError(
            f"projection and labels must have same length, "
            f"got {projection.shape[0]} vs {labels.shape[0]}"
        )

    valid_mask = _valid_label_mask(labels)
    coords = projection[valid_mask]
    valid_labels = labels[valid_mask]

    if len(coords) < 2 or len(np.unique(valid_labels)) < 2:
        return float("nan")

    try:
        indices, k = _build_knn_indices(coords, n_neighbors, metric)
    except ValueError as e:
        warnings.warn(f"Concordex calculation failed: {e}", stacklevel=2)
        return float("nan")

    classes = np.unique(valid_labels)

    # Observed trace
    K_obs = _consolidation_from_indices(indices, valid_labels, classes, k)
    D_obs = _aggregated_matrix(K_obs, valid_labels, classes)
    observed = float(np.trace(D_obs))

    # Permutation null: keep the kNN graph fixed, permute labels only.
    rng = np.random.default_rng(random_state)
    null_traces = np.empty(n_permutations, dtype=float)
    for it in range(n_permutations):
        perm_labels = rng.permutation(valid_labels)
        K_perm = _consolidation_from_indices(
            indices, perm_labels, classes, k
        )
        D_perm = _aggregated_matrix(K_perm, perm_labels, classes)
        null_traces[it] = np.trace(D_perm)
    expected = float(null_traces.mean())

    if expected <= 0.0:
        return float("nan")

    return observed / expected


def calculate_concordex_per_class(
    projection: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 15,
    n_permutations: int = 15,
    random_state: int = 42,
    metric: str = "euclidean",
) -> dict[str, dict[str, float]]:
    """Per-class concordex breakdown.

    Returns a mapping ``class -> {observed, expected, ratio, n_members}``.
    ``observed`` is ``D[a, a]``, the mean fraction of same-class neighbors
    for points of class ``a``. ``expected`` is the same quantity averaged
    over permutations. ``ratio`` is ``observed / expected`` and is the
    per-class analogue of the global concordex score.
    """
    valid_mask = _valid_label_mask(labels)
    coords = projection[valid_mask]
    valid_labels = labels[valid_mask]
    if len(coords) < 2 or len(np.unique(valid_labels)) < 2:
        return {}

    indices, k = _build_knn_indices(coords, n_neighbors, metric)
    classes = np.unique(valid_labels)

    K_obs = _consolidation_from_indices(indices, valid_labels, classes, k)
    D_obs = _aggregated_matrix(K_obs, valid_labels, classes)

    rng = np.random.default_rng(random_state)
    null_diag = np.zeros((n_permutations, len(classes)), dtype=float)
    for it in range(n_permutations):
        perm_labels = rng.permutation(valid_labels)
        K_perm = _consolidation_from_indices(
            indices, perm_labels, classes, k
        )
        D_perm = _aggregated_matrix(K_perm, perm_labels, classes)
        null_diag[it] = np.diag(D_perm)
    expected_diag = null_diag.mean(axis=0)

    out: dict[str, dict[str, float]] = {}
    for a, cls in enumerate(classes):
        n_a = int((valid_labels == cls).sum())
        obs = float(D_obs[a, a])
        exp = float(expected_diag[a])
        ratio = obs / exp if exp > 0 else float("nan")
        out[str(cls)] = {
            "observed": obs,
            "expected": exp,
            "ratio": ratio,
            "n_members": n_a,
        }
    return out


def make_concordex_metric(
    labels: np.ndarray,
    n_neighbors: int = 15,
    n_permutations: int = 15,
    random_state: int = 42,
    metric: str = "euclidean",
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Factory binding ``labels`` into a harness-compatible concordex.

    The returned callable has signature ``(embeddings, projection) ->
    float`` so it can be passed straight into
    ``benchmark_methods(metric_functions=...)``. Labels must already be
    aligned with the embedding row order.
    """

    def concordex_metric(
        embeddings: np.ndarray, projection: np.ndarray
    ) -> float:
        return calculate_concordex_score(
            embeddings,
            projection,
            labels=labels,
            n_neighbors=n_neighbors,
            n_permutations=n_permutations,
            random_state=random_state,
            metric=metric,
        )

    concordex_metric.__name__ = "concordex"
    return concordex_metric


__all__ = [
    "aggregated_label_matrix",
    "calculate_concordex_per_class",
    "calculate_concordex_score",
    "make_concordex_metric",
    "neighborhood_consolidation_matrix",
]