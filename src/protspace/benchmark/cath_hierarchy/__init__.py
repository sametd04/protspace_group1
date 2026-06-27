"""CATH hierarchy preservation evaluation for 2D DR projections."""

from protspace.benchmark.cath_hierarchy.best_vs_default import (
    compute_best_vs_default,
    run_best_vs_default_summary,
)
from protspace.benchmark.cath_hierarchy.hierarchy_preservation import (
    compute_hierarchy_preservation,
    run_hierarchy_preservation_summary,
)
from protspace.benchmark.cath_hierarchy.hyperparam import (
    run_cath_hyperparam_evaluation,
)
from protspace.benchmark.cath_hierarchy.labels import CATHLabels, load_cath_labels
from protspace.benchmark.cath_hierarchy.metrics import (
    compute_adaptive_k,
    compute_khp_baselines,
    knn_hierarchy_purity,
    knn_hierarchy_purity_adaptive,
)
from protspace.benchmark.cath_hierarchy.run import run_cath_hierarchy_evaluation
from protspace.benchmark.cath_hierarchy.visualize import plot_improvement_heatmap

__all__ = [
    "CATHLabels",
    "load_cath_labels",
    "compute_adaptive_k",
    "compute_khp_baselines",
    "knn_hierarchy_purity",
    "knn_hierarchy_purity_adaptive",
    "run_cath_hierarchy_evaluation",
    "run_cath_hyperparam_evaluation",
    "compute_hierarchy_preservation",
    "run_hierarchy_preservation_summary",
    "compute_best_vs_default",
    "run_best_vs_default_summary",
    "plot_improvement_heatmap",
]
