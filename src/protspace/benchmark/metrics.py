from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.metrics import silhouette_score as sklearn_silhouette

from protspace.benchmark.concordex import (
    calculate_concordex_score,
    make_concordex_metric,
)

def calculate_trustworthiness(
    embeddings: np.ndarray, projection: np.ndarray, n_neighbors: int = 15
) -> float:
    """Trustworthiness of the projection w.r.t. the embedding kNN structure.

    Range ``[0, 1]``, higher = better. ``1`` means k-nearest-neighbour
    sets are perfectly preserved. Reference: Venna & Kaski, 2001.
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


def calculate_silhouette_score(
    embeddings: np.ndarray,  # signature kept for harness compatibility
    projection: np.ndarray,
    labels: np.ndarray | None = None,
    metric: str = "euclidean",
) -> float:
    """Silhouette score on the 2D projection.

    Range ``[-1, 1]``, higher = better. Use :func:`make_silhouette_metric`
    to bind labels into a harness-compatible closure.
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
    """Factory binding labels into a harness-compatible silhouette metric."""

    def silhouette_metric(embeddings: np.ndarray, projection: np.ndarray) -> float:
        return calculate_silhouette_score(
            embeddings, projection, labels=labels, metric=metric
        )

    silhouette_metric.__name__ = "silhouette"
    return silhouette_metric

AVAILABLE_METRICS: dict[str, Callable] = {
    "trustworthiness": calculate_trustworthiness,
    "silhouette": calculate_silhouette_score,
    "concordex": calculate_concordex_score,
}