"""Quality metrics for dimensionality reduction evaluation.

This module provides functions to calculate various quality metrics
for evaluating dimensionality reduction projections.
"""

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.neighbors import NearestNeighbors


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

        warnings.warn(f"Trustworthiness calculation failed: {e}", stacklevel=2)
        return float("nan")


def get_knn_indices(data: np.ndarray, k: int = 15) -> np.ndarray:
    """Get k nearest neighbor indices for each sample.
    
    Args:
        data: embeddings (n_samples, n_features)
        k: number of neighbors
    
    Returns:
        Array of shape (n_samples, k) with neighbor indices
    """
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    nn.fit(data)
    indices = nn.kneighbors(return_distance=False)[:, 1:]  # exclude self
    return indices


def compare_knn(high_dim_indices: np.ndarray, low_dim_indices: np.ndarray) -> list:
    """Compare KNN preservation between spaces.
    
    Args:
        high_dim_indices: KNN indices in original space (n_samples, k)
        low_dim_indices: KNN indices in projected space (n_samples, k)
    
    Returns:
        List of dicts with preservation stats per sample
    """
    results = []
    for i in range(high_dim_indices.shape[0]):
        high_set = set(high_dim_indices[i])
        low_set = set(low_dim_indices[i])
        overlap = len(high_set.intersection(low_set))
        k = high_dim_indices.shape[1]
        preservation = overlap / k
        results.append({
            "idx": i,
            "overlap": overlap,
            "preservation": preservation
        })
    return results


def calculate_knn_preservation(embeddings: np.ndarray, projection: np.ndarray, n_neighbors: int = 15) -> float:
    """Calculate mean k-NN preservation between high-dim and low-dim spaces.
    
    Args:
        embeddings: Original high-dimensional embeddings (n_samples, n_features)
        projection: Low-dimensional projection (n_samples, n_components)
        n_neighbors: Number of neighbors to compare (default 15)
    
    Returns:
        Mean preservation score in [0, 1], where 1 means perfect preservation
    """
    k = min(n_neighbors, embeddings.shape[0] - 1)
    if k < 1:
        return float("nan")
    
    high_knn = get_knn_indices(embeddings, k=k)
    low_knn = get_knn_indices(projection, k=k)
    comparison = compare_knn(high_knn, low_knn)
    
    return float(np.mean([c["preservation"] for c in comparison]))


def calculate_continuity(embeddings: np.ndarray, projection: np.ndarray, n_neighbors: int = 15) -> float:
    """Calculate mean continuity: how many non-neighbors in high-dim stay non-neighbors in low-dim.
    
    Args:
        embeddings: Original high-dimensional embeddings (n_samples, n_features)
        projection: Low-dimensional projection (n_samples, n_components)
        n_neighbors: Number of neighbors to check (default 15)
    
    Returns:
        Mean continuity score in [0, 1], where 1 means perfect continuity
    """
    k = min(n_neighbors, embeddings.shape[0] - 1)
    if k < 1:
        return float("nan")
    
    n_samples = embeddings.shape[0]
    high_knn = get_knn_indices(embeddings, k=k)
    low_knn = get_knn_indices(projection, k=k)
    
    continuities = []
    for i in range(n_samples):
        # Non-neighbors in high-dim space
        high_neighbors = set(high_knn[i])
        high_non_neighbors = set(range(n_samples)) - high_neighbors - {i}
        
        # Non-neighbors in low-dim space
        low_neighbors = set(low_knn[i])
        low_non_neighbors = set(range(n_samples)) - low_neighbors - {i}
        
        # How many high-dim non-neighbors are also low-dim non-neighbors
        # Sample k random non-neighbors from high-dim for fair comparison
        if high_non_neighbors and low_non_neighbors:
            k_non = min(k, len(high_non_neighbors), len(low_non_neighbors))
            high_non_sample = set(list(high_non_neighbors)[:k_non])
            overlap = len(high_non_sample.intersection(low_non_neighbors))
            continuity = overlap / k_non
        else:
            continuity = 1.0 if not high_non_neighbors else 0.0
        
        continuities.append(continuity)
    
    return float(np.mean(continuities))


# Registry of available metrics
AVAILABLE_METRICS = {
    "trustworthiness": calculate_trustworthiness,
    "knn_preservation": calculate_knn_preservation,
    "continuity": calculate_continuity,
}

