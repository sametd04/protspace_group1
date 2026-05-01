"""Quality metrics for dimensionality reduction evaluation.

This module provides functions to calculate various quality metrics
for evaluating dimensionality reduction projections.
"""

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trustworthiness


def calculate_trustworthiness(
    embeddings: np.ndarray, projection: np.ndarray, n_neighbors: int = 15
) -> float:
    """Calculate trustworthiness metric using sklearn.

    Trustworthiness measures how well the local neighborhood structure
    is preserved in the low-dimensional projection. A value of 1 indicates
    perfect preservation, while lower values indicate distortion.

    Args:
        embeddings: Original high-dimensional embeddings (n_samples, n_features)
        projection: Low-dimensional projection (n_samples, n_components)
        n_neighbors: Number of neighbors to consider (default 15)

    Returns:
        Trustworthiness score (0 to 1, higher is better)
    """
    n_samples = embeddings.shape[0]

    # Need at least 3 samples for meaningful calculation
    if n_samples < 3:
        return float("nan")

    # Use safer cap (n_samples // 2) to avoid edge cases
    k = min(n_neighbors, n_samples // 2)

    try:
        score = sklearn_trustworthiness(
            X=embeddings, X_embedded=projection, n_neighbors=k, metric="euclidean"
        )
        return float(score)
    except ValueError as e:
        import warnings

        warnings.warn(f"Trustworthiness calculation failed: {e}"), stacklevel = 2
        return float("nan")


# Registry of available metrics
AVAILABLE_METRICS = {
    "trustworthiness": calculate_trustworthiness,
}
