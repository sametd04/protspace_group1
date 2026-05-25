"""Configuration for DR method robustness experiments.

Defines baseline configurations and parameter ranges for all supported methods.
"""

from __future__ import annotations

# k-NN overlap parameter (consistent across all methods)
KNN_K = 15

# Common seed values for all methods
COMMON_SEED_VALUES = [0, 7, 13, 21, 99]

# Method-specific configurations
METHOD_CONFIGS = {
    "umap": {
        "baseline": {
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "random_state": 42,
            "metric": "euclidean",
        },
        "seed_values": COMMON_SEED_VALUES,
        "hyperparam_experiments": {
            "n_neighbors": [5, 15, 50],
            "min_dist": [0.0, 0.5, 0.8],
        },
    },
    "tsne": {
        "baseline": {
            "n_components": 2,
            "perplexity": 30,
            "learning_rate": 200,
            "metric": "euclidean",
            "random_state": 42,
        },
        "seed_values": COMMON_SEED_VALUES,
        "hyperparam_experiments": {
            "perplexity": [5, 30, 50],
            "learning_rate": [50, 200, 500],
        },
    },
    "pacmap": {
        "baseline": {
            "n_components": 2,
            "n_neighbors": 15,
            "mn_ratio": 0.5,
            "fp_ratio": 2.0,
            "random_state": 42,
        },
        "seed_values": COMMON_SEED_VALUES,
        "hyperparam_experiments": {
            "n_neighbors": [5, 10, 20, 50],
            "mn_ratio": [0.1, 0.3, 0.6, 0.9],
            "fp_ratio": [1.0, 3.0, 4.0, 5.0],
        },
    },
    "localmap": {
        "baseline": {
            "n_components": 2,
            "n_neighbors": 15,
            "mn_ratio": 0.5,
            "fp_ratio": 2.0,
            "random_state": 42,
        },
        "seed_values": COMMON_SEED_VALUES,
        "hyperparam_experiments": {
            "n_neighbors": [5, 10, 20, 50],
            "mn_ratio": [0.1, 0.3, 0.6, 0.9],
            "fp_ratio": [1.0, 3.0, 4.0, 5.0],
        },
    },
    "mds": {
        "baseline": {
            "n_components": 2,
            "n_init": 4,
            "max_iter": 300,
            "eps": 1e-3,
            "random_state": 42,
        },
        "seed_values": COMMON_SEED_VALUES,
        "hyperparam_experiments": {
            "n_init": [1, 4, 8],
            "max_iter": [100, 300, 500],
            "eps": [1e-2, 1e-3, 1e-4],
        },
    },
}


def get_method_config(method: str) -> dict:
    """Get configuration for a specific method.

    Args:
        method: DR method name

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If method is not supported
    """
    if method not in METHOD_CONFIGS:
        raise ValueError(
            f"Unknown method '{method}'. Must be one of: {list(METHOD_CONFIGS.keys())}"
        )
    return METHOD_CONFIGS[method]


def get_all_methods() -> list[str]:
    """Get list of all supported methods.

    Returns:
        List of method names
    """
    return list(METHOD_CONFIGS.keys())
