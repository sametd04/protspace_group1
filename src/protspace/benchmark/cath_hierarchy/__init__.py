"""CATH hierarchy preservation evaluation for 2D DR projections."""

from protspace.benchmark.cath_hierarchy.labels import CATHLabels, load_cath_labels
from protspace.benchmark.cath_hierarchy.metrics import (
    compute_adaptive_k,
    compute_khp_baselines,
    knn_hierarchy_purity,
    knn_hierarchy_purity_adaptive,
)
from protspace.benchmark.cath_hierarchy.run import run_cath_hierarchy_evaluation

__all__ = [
    "CATHLabels",
    "load_cath_labels",
    "compute_adaptive_k",
    "compute_khp_baselines",
    "knn_hierarchy_purity",
    "knn_hierarchy_purity_adaptive",
    "run_cath_hierarchy_evaluation",
]
