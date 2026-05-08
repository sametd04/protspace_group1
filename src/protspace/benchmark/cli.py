#!/usr/bin/env python3
"""CLI for benchmarking DR methods on protein embeddings.

Usage:
    uv run python -m protspace.benchmark.cli --data 3ftx
    uv run python -m protspace.benchmark.cli --data 3ftx --plot
    uv run python -m protspace.benchmark.cli --data 3ftx --plot-only
    DATA=globin uv run python -m protspace.benchmark.cli --plot
"""

from __future__ import annotations

import argparse
import os

from protspace.benchmark.io import benchmark_paths
from protspace.benchmark.run import run_benchmark
from protspace.benchmark.visualize import visualize_comparison


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark DR methods (optionally save projection figure)."
    )
    parser.add_argument(
        "--data",
        default=os.environ.get("DATA", "3ftx"),
        help="Dataset name (default: env DATA or 3ftx)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "After benchmarking, write normalization_comparison.png in "
            "src/protspace/benchmark/results/<DATA>/"
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only render figure from existing results (no embedding required)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    paths = benchmark_paths(args.data)

    if args.plot_only:
        visualize_comparison(paths)
        return

    run_benchmark(paths)
    if args.plot:
        visualize_comparison(paths)


if __name__ == "__main__":
    main()
