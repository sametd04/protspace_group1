"""Robustness benchmarking for dimensionality reduction methods.

This module provides tools to quantify the stability of DR methods using k-NN
overlap metrics across different random seeds and hyperparameter configurations.

Supports both global robustness analysis (RobustnessRunner) and group-based
robustness analysis (GroupDRRobustnessAnalyzer).
"""

from protspace.benchmark.robustness.config import (
    KNN_K,
    get_all_methods,
    get_method_config,
)
from protspace.benchmark.robustness.run import RobustnessRunner
from protspace.benchmark.robustness.group_analysis import GroupDRRobustnessAnalyzer
from protspace.benchmark.robustness.data_utils import (
    extract_uniprot_id,
    align_embeddings_metadata,
    load_and_prepare_data,
)
from protspace.benchmark.robustness.plot import plot_group_heatmap

__all__ = [
    # Config
    "KNN_K",
    "get_all_methods",
    "get_method_config",
    # Global analysis
    "RobustnessRunner",
    # Group analysis
    "GroupDRRobustnessAnalyzer",
    # Data utilities
    "extract_uniprot_id",
    "align_embeddings_metadata",
    "load_and_prepare_data",
    # Plotting
    "plot_group_heatmap",
]
