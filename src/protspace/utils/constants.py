"""Lightweight constants and config — no heavy dependencies (sklearn, umap, pacmap).

Import this module freely without triggering numba/pynndescent compilation.
"""

from dataclasses import dataclass, field, fields
from typing import Literal, get_args

# Method name constants
PCA_NAME = "pca"
TSNE_NAME = "tsne"
UMAP_NAME = "umap"
PACMAP_NAME = "pacmap"
MDS_NAME = "mds"
LOCALMAP_NAME = "localmap"
PPCA_NAME = "ppca"
KPPCA_NAME = "kppca"

REDUCER_METHODS = [
    PCA_NAME,
    TSNE_NAME,
    UMAP_NAME,
    PACMAP_NAME,
    MDS_NAME,
    LOCALMAP_NAME,
    PPCA_NAME,
    KPPCA_NAME
]

# Metric types
METRIC_TYPES = Literal["euclidean", "cosine"]

# ρPCA background selection strategies. "external" is selected automatically
# when a background dataset is provided via --ppca-background; the other three
# are auto-split policies on the target data.
BACKGROUND_STRATEGY_TYPES = Literal["external", "random", "uniform", "outlier"]

# k-ρPCA kernel sources and kernel functions
KERNEL_SOURCE_TYPES = Literal["embedding", "similarity", "precomputed"]
KERNEL_TYPES = Literal["gaussian", "inverse_distance", "linear"]


@dataclass(frozen=True)
class DimensionReductionConfig:
    """Configuration for dimension reduction methods.

    Parameters:
        n_components: Number of dimensions in reduced space (2 or 3)
        n_neighbors: Number of neighbors for manifold learning (>0)
        metric: Distance metric to use
        precomputed: Whether distances are precomputed
        min_dist: Minimum distance for UMAP (0-1)
        perplexity: Perplexity for t-SNE (5-50)
        learning_rate: Learning rate for t-SNE optimization (>0)
        mn_ratio: Ratio for PaCMAP (0-1)
        fp_ratio: Ratio for PaCMAP (>0)
        n_init: Number of initializations for MDS (>0)
        max_iter: Maximum iterations (>0)
        eps: Convergence tolerance (>0)
        random_state: Random seed for reproducibility (>= 0)
        regularization_mu: Tikhonov μ added to Σ_B before solving the
            ρPCA generalized eigenproblem (>= 0). Required (> 0) whenever
            n_background <= n_features.
        background_ratio: Fraction of target samples drawn as background
            in auto-split modes. Ignored when background_strategy =
            "external". Must lie strictly in (0, 1).
        background_strategy: Background construction policy.
            "external" uses the dataset passed via background_data (set by
            the pipeline from --ppca-background); the auto-split policies
            "random", "uniform", "outlier" partition the target data.
        standard_scale: If True, per-set standard-scale (column mean 0,
            unit variance) target and background matrices before computing
            covariances. Matches the Carilli/Jackson/Pachter convention.
            Strongly recommended for PLM embeddings whose dimensions have
            heterogeneous scale.
        kernel, kernel_source, kernel_bandwidth and background_kernel: consumed by k-ρPCA
    """

    n_components: int = field(default=2, metadata={"allowed": [2, 3]})
    n_neighbors: int = field(default=15, metadata={"gt": 0})
    metric: METRIC_TYPES = field(
        default="euclidean", metadata={"allowed": list(get_args(METRIC_TYPES))}
    )
    precomputed: bool = field(default=False)
    min_dist: float = field(default=0.1, metadata={"gte": 0, "lte": 1})
    perplexity: int = field(default=30, metadata={"gte": 5, "lte": 50})
    learning_rate: int = field(default=200, metadata={"gt": 0})
    mn_ratio: float = field(default=0.5, metadata={"gte": 0, "lte": 1})
    fp_ratio: float = field(default=2.0, metadata={"gt": 0})
    n_init: int = field(default=4, metadata={"gt": 0})
    max_iter: int = field(default=300, metadata={"gt": 0})
    eps: float = field(default=1e-3, metadata={"gt": 0})
    random_state: int = field(default=42, metadata={"gte": 0})

    # ρPCA parameters
    regularization_mu: float = field(default=1e-3, metadata={"gte": 0})
    background_ratio: float = field(default=0.3, metadata={"gt": 0, "lt": 1})
    background_strategy: BACKGROUND_STRATEGY_TYPES = field(
        default="outlier",
        metadata={"allowed": list(get_args(BACKGROUND_STRATEGY_TYPES))},
    )
    standard_scale: bool = field(default=True)

    # k-ρPCA parameters
    kernel: KERNEL_TYPES = field(
        default="gaussian",
        metadata={"allowed": list(get_args(KERNEL_TYPES))},
    )
    kernel_source: KERNEL_SOURCE_TYPES = field(
        default="embedding",
        metadata={"allowed": list(get_args(KERNEL_SOURCE_TYPES))},
    )
    # kernel_bandwidth=0 means "auto" (sqrt(median pairwise distance), the
    # rhopca reference heuristic). Any positive value is used literally.
    kernel_bandwidth: float = field(default=0.0, metadata={"gte": 0})
    background_kernel: bool = field(default=False)
    

    def __post_init__(self):
        """Validate configuration parameters."""
        for data_field in fields(self):
            value = getattr(self, data_field.name)
            metadata = data_field.metadata

            if "allowed" in metadata:
                if value not in metadata["allowed"]:
                    raise ValueError(
                        f"{data_field.name} must be one of {metadata['allowed']}"
                    )
            if "gt" in metadata:
                if value <= metadata["gt"]:
                    raise ValueError(
                        f"{data_field.name} must be greater than {metadata['gt']}"
                    )
            if "lt" in metadata:
                if value >= metadata["lt"]:
                    raise ValueError(
                        f"{data_field.name} must be less than {metadata['lt']}"
                    )
            if "gte" in metadata:
                if value < metadata["gte"]:
                    raise ValueError(
                        f"{data_field.name} must be greater than or equal to {metadata['gte']}"
                    )
            if "lte" in metadata:
                if value > metadata["lte"]:
                    raise ValueError(
                        f"{data_field.name} must be less than or equal to {metadata['lte']}"
                    )