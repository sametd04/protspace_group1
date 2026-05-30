"""Lightweight constants and config — no heavy dependencies.

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
    PCA_NAME, TSNE_NAME, UMAP_NAME, PACMAP_NAME,
    MDS_NAME, LOCALMAP_NAME, PPCA_NAME, KPPCA_NAME,
]

# Distance metric types
METRIC_TYPES = Literal["euclidean", "cosine"]

# ρPCA background construction strategies
BACKGROUND_STRATEGY_TYPES = Literal[
    "pool", "complement", "length_matched", "stratified", "mixed", "isolate"
]

# k-ρPCA kernel sources and kernel functions (paper 2)
KERNEL_SOURCE_TYPES = Literal["embedding", "similarity", "precomputed"]
KERNEL_TYPES = Literal["gaussian", "inverse_distance", "linear"]


@dataclass(frozen=True)
class DimensionReductionConfig:
    """Configuration for all dimension-reduction methods."""

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
    background_strategy: BACKGROUND_STRATEGY_TYPES = field(
        default="pool",
        metadata={"allowed": list(get_args(BACKGROUND_STRATEGY_TYPES))},
    )
    standard_scale: bool = field(default=True)
    samples_per_target: int = field(default=3, metadata={"gt": 0})
    n_length_bins: int = field(default=10, metadata={"gt": 0})

    # k-ρPCA parameters
    kernel: KERNEL_TYPES = field(
        default="gaussian",
        metadata={"allowed": list(get_args(KERNEL_TYPES))},
    )
    kernel_source: KERNEL_SOURCE_TYPES = field(
        default="embedding",
        metadata={"allowed": list(get_args(KERNEL_SOURCE_TYPES))},
    )
    kernel_bandwidth: float = field(default=0.0, metadata={"gte": 0})
    background_kernel: bool = field(default=False)

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            m = f.metadata
            if "allowed" in m and value not in m["allowed"]:
                raise ValueError(f"{f.name} must be one of {m['allowed']}")
            if "gt" in m and value <= m["gt"]:
                raise ValueError(f"{f.name} must be > {m['gt']}")
            if "lt" in m and value >= m["lt"]:
                raise ValueError(f"{f.name} must be < {m['lt']}")
            if "gte" in m and value < m["gte"]:
                raise ValueError(f"{f.name} must be ≥ {m['gte']}")
            if "lte" in m and value > m["lte"]:
                raise ValueError(f"{f.name} must be ≤ {m['lte']}")