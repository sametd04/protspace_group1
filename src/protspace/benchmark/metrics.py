"""Quality metrics for dimensionality reduction evaluation.

Each metric follows the harness pattern ``(embeddings, projection) -> float``
so it can be plugged into ``benchmark_methods(metric_functions=...)``.

Two flavours of metric:
- **Geometric metrics** (``calculate_trustworthiness``): only need the
  high-D embeddings and the 2D projection.
- **Label-based metrics** (``calculate_silhouette_score``): also need a
  categorical label per protein. Use the factory ``make_silhouette_metric``
  to bind labels into a closure that matches the harness signature.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.metrics import silhouette_score as sklearn_silhouette
from sklearn.neighbors import NearestNeighbors


def calculate_trustworthiness(
    embeddings: np.ndarray, projection: np.ndarray, n_neighbors: int = 15
) -> float:
    """Trustworthiness: how well the local k-NN structure of the
    high-dimensional embeddings is preserved in the 2D projection.

    Range ``[0, 1]``, higher = better. ``1`` = neighbourhoods perfectly
    preserved. Reference: Venna & Kaski, 2001.
    """
    n_samples = embeddings.shape[0]
    if n_samples < 3:
        return float("nan")

    k = min(n_neighbors, n_samples // 2)

    try:
        score = sklearn_trustworthiness(
            X=embeddings, X_embedded=projection, n_neighbors=k, metric="euclidean"
        )
        return float(score)
    except ValueError as e:
        warnings.warn(f"Trustworthiness calculation failed: {e}", stacklevel=2)
        return float("nan")


def calculate_knn_overlap(
    projection_a: np.ndarray, projection_b: np.ndarray, n_neighbors: int = 15
) -> float:
    """Compute k-NN overlap between two projections.

    Measures how many of the k nearest neighbors are preserved between
    two different projections of the same data. This quantifies embedding
    stability under perturbations (e.g., different random seeds or
    hyperparameters).

    Range ``[0, 1]``, higher = better. ``1`` = all neighbors perfectly
    preserved across projections.

    Parameters
    ----------
    projection_a
        First projection (n_samples, n_dims)
    projection_b
        Second projection (n_samples, n_dims)
    n_neighbors
        Number of nearest neighbors to consider

    Returns
    -------
    Mean k-NN overlap score across all points
    """
    if projection_a.shape != projection_b.shape:
        raise ValueError(
            f"Projections must have same shape, "
            f"got {projection_a.shape} vs {projection_b.shape}"
        )

    n_samples = projection_a.shape[0]
    if n_samples < 3:
        return float("nan")

    k = min(n_neighbors, n_samples - 1)  # Exclude self

    # Find k nearest neighbors in both projections
    nbrs_a = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(projection_a)
    nbrs_b = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(projection_b)

    _, indices_a = nbrs_a.kneighbors(projection_a)
    _, indices_b = nbrs_b.kneighbors(projection_b)

    # Remove self (first neighbor) and keep only k neighbors
    indices_a = indices_a[:, 1:]
    indices_b = indices_b[:, 1:]

    # Compute overlap for each point
    overlaps = []
    for i in range(n_samples):
        neighbors_a = set(indices_a[i])
        neighbors_b = set(indices_b[i])
        overlap = len(neighbors_a & neighbors_b) / k
        overlaps.append(overlap)

    return float(np.mean(overlaps))


def calculate_silhouette_score(
    embeddings: np.ndarray,  # noqa: ARG001  # signature kept for harness compat
    projection: np.ndarray,
    labels: np.ndarray | None = None,
    metric: str = "euclidean",
) -> float:
    """Silhouette score on the 2D projection coordinates.

    Measures how well same-label points are clustered together AND
    separated from other-label points in the projection. Range
    ``[-1, 1]``, higher = better.

    Notes
    -----
    The harness signature is ``(embeddings, projection) -> float``, but
    silhouette needs labels. Two ways to use this:

    1. Pass labels directly (only works if you call this function manually):

       >>> calculate_silhouette_score(emb, proj, labels=labels)

    2. Use the factory ``make_silhouette_metric`` to bind labels into a
       closure with the harness-compatible signature:

       >>> metric_fn = make_silhouette_metric(labels)
       >>> benchmark_methods(..., metric_functions={"silhouette": metric_fn})

    Returns ``NaN`` if no labels are provided, fewer than two distinct
    classes survive filtering, or fewer than two points remain.
    """
    if labels is None:
        return float("nan")

    if projection.shape[0] != labels.shape[0]:
        raise ValueError(
            f"projection and labels must have same length, "
            f"got {projection.shape[0]} vs {labels.shape[0]}"
        )

    valid_mask = np.array(
        [
            lbl is not None
            and not (isinstance(lbl, float) and np.isnan(lbl))
            and str(lbl).strip() != ""
            for lbl in labels
        ]
    )
    coords = projection[valid_mask]
    valid_labels = labels[valid_mask]

    if len(coords) < 2 or len(np.unique(valid_labels)) < 2:
        return float("nan")

    return float(sklearn_silhouette(coords, valid_labels, metric=metric))


def make_silhouette_metric(
    labels: np.ndarray, metric: str = "euclidean"
) -> Callable[[np.ndarray, np.ndarray], float]:
    """Factory that binds labels into a harness-compatible silhouette metric.

    The returned callable has signature ``(embeddings, projection) -> float``
    so it slots into ``benchmark_methods(metric_functions=...)``. Labels must
    already be aligned with the embedding row order.
    """

    def silhouette_metric(embeddings: np.ndarray, projection: np.ndarray) -> float:
        return calculate_silhouette_score(
            embeddings, projection, labels=labels, metric=metric
        )

    silhouette_metric.__name__ = "silhouette"
    return silhouette_metric


AVAILABLE_METRICS: dict[str, Callable] = {
    "trustworthiness": calculate_trustworthiness,
    "silhouette": calculate_silhouette_score,
}


def default_metric_functions(
    labels: np.ndarray | None = None,
) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """Get default metric functions for benchmark pipeline.

    Returns trustworthiness as baseline metric. If labels are provided,
    also includes silhouette score.

    Parameters
    ----------
    labels
        Categorical labels aligned with embedding rows. If None,
        silhouette metric is omitted.

    Returns
    -------
    Dictionary mapping metric names to callables with signature
    ``(embeddings, projection) -> float``.
    """
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "trustworthiness": calculate_trustworthiness
    }
    if labels is not None:
        metrics["silhouette"] = make_silhouette_metric(labels)
    return metrics
