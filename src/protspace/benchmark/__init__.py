"""Benchmarking tools for dimensionality reduction methods.

This package provides tools for benchmarking and evaluating dimensionality
reduction methods, including timing, quality metrics, and comparative analysis.
"""

from protspace.benchmark.harness import (
    BenchmarkResult,
    benchmark_method,
    benchmark_methods,
    normalize_projection,
)
from protspace.benchmark.metrics import AVAILABLE_METRICS

__all__ = [
    "BenchmarkResult",
    "benchmark_method",
    "benchmark_methods",
    "normalize_projection",
    "AVAILABLE_METRICS",
]
