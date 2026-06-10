"""Orchestrator for the CATH hierarchy preservation evaluation.

Loads embeddings, runs all (or selected) DR methods via the existing
:func:`~protspace.benchmark.harness.benchmark_methods` harness, computes
k-NN Hierarchy Purity (KHP) at all four CATH levels, and writes results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from protspace.benchmark.cath_hierarchy.labels import CATHLabels, load_cath_labels
from protspace.benchmark.cath_hierarchy.metrics import (
    compute_khp_baselines,
    knn_hierarchy_purity,
    make_khp_metrics,
)
from protspace.benchmark.harness import benchmark_methods
from protspace.utils.constants import REDUCER_METHODS, DimensionReductionConfig

logger = logging.getLogger(__name__)

# Ordered from finest to coarsest — used for display
_LEVEL_ORDER = ["khp_homology", "khp_topology", "khp_architecture", "khp_class"]
_LEVEL_LABELS = {
    "khp_homology": "Homology (H)",
    "khp_topology": "Topology (T)",
    "khp_architecture": "Architecture (A)",
    "khp_class": "Class (C)",
}


def run_cath_hierarchy_evaluation(
    h5_path: Path,
    output_dir: Path,
    methods: list[str] | None = None,
    k: int = 5,
    config: DimensionReductionConfig | None = None,
    max_proteins: int | None = None,
) -> pd.DataFrame:
    """Run CATH hierarchy preservation evaluation across DR methods.

    Parameters
    ----------
    h5_path:
        Path to ``data/cath_s40/prot_t5.h5``.
    output_dir:
        Directory where results CSV and plots are saved.
    methods:
        DR methods to benchmark (default: all six).
    k:
        Number of nearest neighbours for KHP.
    config:
        :class:`~protspace.utils.constants.DimensionReductionConfig` to pass
        to every DR method.  Uses defaults if *None*.
    max_proteins:
        If set, randomly subsample to this many proteins before running DR.
        Useful for fast testing (e.g. ``max_proteins=500``).

    Returns
    -------
    DataFrame with columns:
        ``method``, ``time_s``,
        ``khp_homology``, ``khp_topology``, ``khp_architecture``, ``khp_class``,
        ``baseline_homology``, ``baseline_topology``, ``baseline_architecture``,
        ``baseline_class``, ``k``, ``n_valid``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = list(REDUCER_METHODS)

    if config is None:
        config = DimensionReductionConfig()

    # ------------------------------------------------------------------
    # 1. Load embeddings + CATH labels
    # ------------------------------------------------------------------
    logger.info("Loading CATH embeddings and hierarchy labels …")
    cath_labels: CATHLabels = load_cath_labels(h5_path)

    embeddings = cath_labels.embeddings
    logger.info(
        "Dataset: %d proteins, %d with CATH labels, embedding dim=%d",
        len(cath_labels.identifiers),
        cath_labels.n_valid,
        embeddings.shape[1],
    )

    if max_proteins is not None and len(cath_labels.identifiers) > max_proteins:
        logger.info("Subsampling to %d proteins for speed …", max_proteins)
        rng = np.random.default_rng(42)
        idx = rng.choice(len(cath_labels.identifiers), size=max_proteins, replace=False)
        idx.sort()
        embeddings = embeddings[idx]
        cath_labels = CATHLabels(
            identifiers=[cath_labels.identifiers[i] for i in idx],
            embeddings=embeddings,
            homology=cath_labels.homology[idx],
            topology=cath_labels.topology[idx],
            architecture=cath_labels.architecture[idx],
            cath_class=cath_labels.cath_class[idx],
            valid_mask=cath_labels.valid_mask[idx],
        )
        logger.info(
            "After subsample: %d proteins, %d with CATH labels",
            len(cath_labels.identifiers),
            cath_labels.n_valid,
        )

    # ------------------------------------------------------------------
    # 2. Build harness-compatible metric closures
    # ------------------------------------------------------------------
    metric_functions = make_khp_metrics(cath_labels, k=k)

    # ------------------------------------------------------------------
    # 3. Run all DR methods via the existing harness
    # ------------------------------------------------------------------
    logger.info("Benchmarking %d DR methods: %s", len(methods), methods)
    benchmark_results = benchmark_methods(
        embeddings=embeddings,
        methods=methods,
        config=config,
        normalize=False,  # raw projection for accurate distance metrics
        metric_functions=metric_functions,
    )

    # ------------------------------------------------------------------
    # 4. Compute random baselines (independent of projection)
    # ------------------------------------------------------------------
    baselines = compute_khp_baselines(cath_labels)
    logger.info(
        "Random baselines (k-independent):  class=%.4f  arch=%.4f  topo=%.4f  homo=%.4f",
        baselines["baseline_cath_class"],
        baselines["baseline_architecture"],
        baselines["baseline_topology"],
        baselines["baseline_homology"],
    )

    # ------------------------------------------------------------------
    # 5. Collect results into a DataFrame
    # ------------------------------------------------------------------
    rows = []
    for method, result in benchmark_results.items():
        # The harness already computed KHP via the metric closures.
        # Also call knn_hierarchy_purity directly for n_valid / k provenance.
        khp_full = knn_hierarchy_purity(result.projection, cath_labels, k=k)

        row: dict[str, object] = {
            "method": method,
            "time_s": round(result.time_seconds, 2),
            # Observed KHP
            "khp_homology": result.metrics.get("khp_homology", float("nan")),
            "khp_topology": result.metrics.get("khp_topology", float("nan")),
            "khp_architecture": result.metrics.get("khp_architecture", float("nan")),
            "khp_class": result.metrics.get("khp_class", float("nan")),
            # Random baselines (same value for every method row — convenient for CSV)
            "baseline_homology": baselines["baseline_homology"],
            "baseline_topology": baselines["baseline_topology"],
            "baseline_architecture": baselines["baseline_architecture"],
            "baseline_class": baselines["baseline_cath_class"],
            "k": int(khp_full["k"]),
            "n_valid": int(khp_full["n_valid"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    csv_path = output_dir / "cath_hierarchy_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    _print_table(df, k, baselines)

    return df


def _print_table(df: pd.DataFrame, k: int, baselines: dict[str, float]) -> None:
    """Print a formatted comparison table to stdout."""

    def _fmt(v: object) -> str:
        try:
            return f"{float(v):.4f}"  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "   N/A"

    w = 76
    print()
    print("=" * w)
    print(f"  CATH Hierarchy Preservation — k-NN Purity (k={k})")
    print("=" * w)
    header = (
        f"  {'Method':<12}  {'Homology(H)':>12}  {'Topology(T)':>12}  "
        f"{'Arch.(A)':>10}  {'Class(C)':>10}  {'time(s)':>8}"
    )
    print(header)
    print("-" * w)

    for _, row in df.iterrows():
        line = (
            f"  {row['method']:<12}  "
            f"{_fmt(row['khp_homology']):>12}  "
            f"{_fmt(row['khp_topology']):>12}  "
            f"{_fmt(row['khp_architecture']):>10}  "
            f"{_fmt(row['khp_class']):>10}  "
            f"{row['time_s']:>8.1f}"
        )
        print(line)

    # Baseline row
    print("-" * w)
    baseline_line = (
        f"  {'random baseline':<12}  "
        f"{_fmt(baselines['baseline_homology']):>12}  "
        f"{_fmt(baselines['baseline_topology']):>12}  "
        f"{_fmt(baselines['baseline_architecture']):>10}  "
        f"{_fmt(baselines['baseline_cath_class']):>10}  "
        f"{'':>8}"
    )
    print(baseline_line)
    print("=" * w)
    print(
        "  Observed ordering is always Class > Architecture > Topology > Homology\n"
        "  (driven by group size, not projection quality — see metrics.py docstring)\n"
        "  Compare values across methods at the same level to assess DR quality.\n"
    )
