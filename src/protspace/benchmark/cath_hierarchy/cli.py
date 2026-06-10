#!/usr/bin/env python3
"""CLI entry point for the CATH hierarchy preservation evaluation.

Runs all (or selected) DR methods on the CATH S40 ProtT5 embeddings, computes
k-NN Hierarchy Purity at each CATH level (Class, Architecture, Topology,
Homology), prints a comparison table, and saves results + plots.

Usage examples
--------------
# Full run — all 6 DR methods on all 3 000 proteins:
    uv run python src/protspace/benchmark/cath_hierarchy/cli.py

# Fast smoke-test on 500 proteins with only PCA and UMAP:
    uv run python src/protspace/benchmark/cath_hierarchy/cli.py \\
        --methods pca2,umap2 \\
        --max-proteins 500 \\
        --output src/protspace/benchmark/cath_hierarchy/results/test/

# Custom k and output dir:
    uv run python src/protspace/benchmark/cath_hierarchy/cli.py \\
        --k 10 \\
        --output src/protspace/benchmark/cath_hierarchy/results/k10/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# cli.py lives at src/protspace/benchmark/cath_hierarchy/cli.py
# → parent.parent.parent.parent is the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from protspace.benchmark.cath_hierarchy.run import run_cath_hierarchy_evaluation  # noqa: E402, I001
from protspace.benchmark.cath_hierarchy.visualize import plot_khp_results  # noqa: E402
from protspace.utils.constants import REDUCER_METHODS  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CATH hierarchy preservation evaluation via k-NN purity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--h5",
        default=str(_REPO_ROOT / "data" / "cath_s40" / "prot_t5.h5"),
        help="Path to the CATH S40 HDF5 embedding file (default: data/cath_s40/prot_t5.h5)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "results"),
        help="Output directory for results CSV and plots (default: src/protspace/benchmark/cath_hierarchy/results/)",
    )
    parser.add_argument(
        "--methods",
        default=",".join(REDUCER_METHODS),
        help=(
            "Comma-separated DR methods to benchmark "
            f"(default: all — {','.join(REDUCER_METHODS)})"
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbours for k-NN purity (default: 5)",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        metavar="N",
        help="Randomly subsample to N proteins before running DR (useful for testing)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v = INFO, -vv = DEBUG)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Configure logging
    log_level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    h5_path = Path(args.h5)
    if not h5_path.exists():
        print(f"ERROR: HDF5 file not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    output_dir = Path(args.output)

    print("\nCATH Hierarchy Preservation Evaluation")
    print(f"  H5 file    : {h5_path}")
    print(f"  Output dir : {output_dir}")
    print(f"  Methods    : {methods}")
    print(f"  k          : {args.k}")
    if args.max_proteins:
        print(f"  Max proteins: {args.max_proteins} (subsampled)")
    print()

    df = run_cath_hierarchy_evaluation(
        h5_path=h5_path,
        output_dir=output_dir,
        methods=methods,
        k=args.k,
        max_proteins=args.max_proteins,
    )

    if not args.no_plots:
        plot_khp_results(df, output_dir=output_dir, k=args.k)
        print(f"Plots saved to: {output_dir}/")

    print(f"\nResults CSV: {output_dir / 'cath_hierarchy_results.csv'}")


if __name__ == "__main__":
    main()
