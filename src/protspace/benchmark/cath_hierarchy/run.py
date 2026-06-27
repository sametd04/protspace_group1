"""Orchestrator for the CATH hierarchy preservation evaluation.

Loads embeddings, runs DR methods across multiple random seeds, computes
k-NN Hierarchy Purity (KHP) with an adaptive per-level k, and compares
against two references:

* **Random baseline** — analytical expected KHP under a uniform random
  projection (k-independent).
* **Original Embedding** — KHP computed directly in the 1024-dimensional
  ProtT5 space; an upper bound on what a 2D projection can achieve.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from protspace.benchmark.cath_hierarchy.labels import CATHLabels, load_cath_labels
from protspace.benchmark.cath_hierarchy.metrics import (
    LEVEL_KEYS,
    compute_adaptive_k,
    compute_group_stats,
    compute_khp_baselines,
    knn_hierarchy_purity_adaptive,
)
from protspace.benchmark.harness import benchmark_method
from protspace.utils.constants import REDUCER_METHODS, DimensionReductionConfig

logger = logging.getLogger(__name__)

N_SEEDS_DEFAULT = 10
DEFAULT_SEEDS = [0, 7, 13, 21, 42, 99, 123, 256, 512, 1024]

# Display order: finest → coarsest
_LEVEL_DISPLAY = ["homology", "topology", "architecture", "cath_class"]
_LEVEL_LABELS = {
    "homology": "Homology (H)",
    "topology": "Topology (T)",
    "architecture": "Architecture (A)",
    "cath_class": "Class (C)",
}


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------


def load_labels(h5_path: Path, max_proteins: int | None = None) -> CATHLabels:
    """Load CATH embeddings + hierarchy labels, optionally subsampling.

    Shared by the seed-robustness and hyperparameter-robustness workflows so
    both operate on an identically prepared (and identically subsampled) dataset.

    Parameters
    ----------
    h5_path:
        Path to ``data/cath_s40/prot_t5.h5``.
    max_proteins:
        If given and smaller than the dataset, subsample to this many proteins
        using a fixed RNG seed (42) for reproducibility.
    """
    logger.info("Loading CATH embeddings and hierarchy labels …")
    cath_labels: CATHLabels = load_cath_labels(h5_path)

    logger.info(
        "Dataset: %d proteins (%d with CATH labels), embedding dim=%d",
        len(cath_labels.identifiers),
        cath_labels.n_valid,
        cath_labels.embeddings.shape[1],
    )

    if max_proteins is not None and len(cath_labels.identifiers) > max_proteins:
        logger.info("Subsampling to %d proteins …", max_proteins)
        rng = np.random.default_rng(42)
        idx = np.sort(
            rng.choice(len(cath_labels.identifiers), size=max_proteins, replace=False)
        )
        cath_labels = CATHLabels(
            identifiers=[cath_labels.identifiers[i] for i in idx],
            embeddings=cath_labels.embeddings[idx],
            homology=cath_labels.homology[idx],
            topology=cath_labels.topology[idx],
            architecture=cath_labels.architecture[idx],
            cath_class=cath_labels.cath_class[idx],
            valid_mask=cath_labels.valid_mask[idx],
        )

    return cath_labels


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_cath_hierarchy_evaluation(
    h5_path: Path,
    output_dir: Path,
    methods: list[str] | None = None,
    seeds: list[int] | None = None,
    config: DimensionReductionConfig | None = None,
    max_proteins: int | None = None,
) -> pd.DataFrame:
    """Run the full CATH hierarchy preservation evaluation.

    Parameters
    ----------
    h5_path:
        Path to ``data/cath_s40/prot_t5.h5``.
    output_dir:
        Directory for results CSV.
    methods:
        DR methods to benchmark (default: all six).
    seeds:
        Explicit list of seed values to use.  Overrides *n_seeds*.
        Defaults to :data:`DEFAULT_SEEDS`.
    config:
        Base :class:`~protspace.utils.constants.DimensionReductionConfig`
        (``random_state`` is overridden per seed).  Uses defaults if *None*.
    max_proteins:
        Subsample to this many proteins for faster testing.

    Returns
    -------
    DataFrame with rows for each DR method plus "Original Embedding" and
    "Random Baseline".  Columns include ``khp_{level}_mean``,
    ``khp_{level}_std``, ``baseline_{level}``, ``k_{level}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = list(REDUCER_METHODS)

    base_config = config or DimensionReductionConfig()
    seed_list = seeds if seeds is not None else DEFAULT_SEEDS

    # ------------------------------------------------------------------
    # 1. Load embeddings + labels
    # ------------------------------------------------------------------
    cath_labels = load_labels(h5_path, max_proteins=max_proteins)
    embeddings = cath_labels.embeddings

    # ------------------------------------------------------------------
    # 2. Group-size analysis → adaptive k
    # ------------------------------------------------------------------
    group_stats = compute_group_stats(cath_labels)
    k_per_level = compute_adaptive_k(cath_labels)
    _print_group_stats(group_stats, k_per_level)

    # ------------------------------------------------------------------
    # 3. Random baseline (analytical, k-independent)
    # ------------------------------------------------------------------
    baselines = compute_khp_baselines(cath_labels)

    # ------------------------------------------------------------------
    # 4. Original Embedding reference (high-D k-NN, run once)
    # ------------------------------------------------------------------
    logger.info(
        "Computing KHP in original %d-dim embedding space …", embeddings.shape[1]
    )
    orig_khp = knn_hierarchy_purity_adaptive(embeddings, cath_labels, k_per_level)

    # ------------------------------------------------------------------
    # 5. Multi-seed DR evaluation
    # ------------------------------------------------------------------
    rows: list[dict] = []

    for method in methods:
        logger.info(
            "Method %s — running %d seeds %s …", method, len(seed_list), seed_list
        )
        seed_khp: list[dict[str, float]] = []

        for seed in seed_list:
            # Override random_state; keep all other params from base_config
            cfg_dict = {
                k: getattr(base_config, k) for k in base_config.__dataclass_fields__
            }
            cfg_dict["random_state"] = seed
            seed_config = DimensionReductionConfig(**cfg_dict)

            result = benchmark_method(
                embeddings=embeddings,
                method=method,
                config=seed_config,
                normalize=False,
                metric_functions=None,
            )
            khp = knn_hierarchy_purity_adaptive(
                result.projection, cath_labels, k_per_level
            )
            seed_khp.append(khp)
            logger.debug(
                "  seed=%d  homo=%.4f  topo=%.4f  arch=%.4f  class=%.4f",
                seed,
                khp["khp_homology"],
                khp["khp_topology"],
                khp["khp_architecture"],
                khp["khp_cath_class"],
            )

        rows.append(
            _aggregate_seeds(method, seed_khp, baselines, k_per_level, len(seed_list))
        )

    # ------------------------------------------------------------------
    # 6. Assemble DataFrame — DR methods + references
    # ------------------------------------------------------------------
    orig_row = _make_reference_row(
        "Original Embedding", orig_khp, baselines, k_per_level, n_seeds=1
    )
    baseline_row = _make_baseline_row(baselines, k_per_level)

    df = pd.DataFrame([orig_row] + rows + [baseline_row])

    # ------------------------------------------------------------------
    # 7. Save + report
    # ------------------------------------------------------------------
    csv_path = output_dir / "cath_hierarchy_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    _print_table(df, k_per_level)

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_seeds(
    method: str,
    seed_khp: list[dict[str, float]],
    baselines: dict[str, float],
    k_per_level: dict[str, int],
    n_seeds: int,
) -> dict:
    row: dict = {"method": method, "n_seeds": n_seeds}
    for level in LEVEL_KEYS:
        vals = np.array([r[f"khp_{level}"] for r in seed_khp])
        row[f"khp_{level}_mean"] = float(np.mean(vals))
        row[f"khp_{level}_std"] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    _add_shared_cols(row, baselines, k_per_level)
    return row


def _make_reference_row(
    name: str,
    khp: dict[str, float],
    baselines: dict[str, float],
    k_per_level: dict[str, int],
    n_seeds: int,
) -> dict:
    row: dict = {"method": name, "n_seeds": n_seeds}
    for level in LEVEL_KEYS:
        row[f"khp_{level}_mean"] = khp[f"khp_{level}"]
        row[f"khp_{level}_std"] = 0.0
    _add_shared_cols(row, baselines, k_per_level)
    return row


def _make_baseline_row(
    baselines: dict[str, float], k_per_level: dict[str, int]
) -> dict:
    row: dict = {"method": "Random Baseline", "n_seeds": 0}
    for level in LEVEL_KEYS:
        row[f"khp_{level}_mean"] = baselines[f"baseline_{level}"]
        row[f"khp_{level}_std"] = 0.0
    _add_shared_cols(row, baselines, k_per_level)
    return row


def _add_shared_cols(
    row: dict, baselines: dict[str, float], k_per_level: dict[str, int]
) -> None:
    for level in LEVEL_KEYS:
        row[f"baseline_{level}"] = baselines[f"baseline_{level}"]
        row[f"k_{level}"] = k_per_level[level]
    row["n_valid"] = None  # filled externally if needed


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def _print_group_stats(stats: dict[str, dict], k_per_level: dict[str, int]) -> None:
    print()
    print("  CATH Group-Size Distribution (S40 dataset)")
    print(
        f"  {'Level':<14}  {'Groups':>7}  {'Median':>7}  {'Mean':>7}  "
        f"{'p75':>5}  {'p90':>5}  {'Singles':>8}  {'k':>4}"
    )
    print("  " + "-" * 68)
    for level in _LEVEL_DISPLAY:
        s = stats[level]
        k = k_per_level[level]
        print(
            f"  {level:<14}  {s['n_groups']:>7}  {s['median']:>7.1f}  "
            f"{s['mean']:>7.1f}  {s['p75']:>5.0f}  {s['p90']:>5.0f}  "
            f"{s['n_singletons']:>8}  {k:>4}"
        )
    print()


def _print_table(df: pd.DataFrame, k_per_level: dict[str, int]) -> None:
    k_h = k_per_level["homology"]
    k_t = k_per_level["topology"]
    k_a = k_per_level["architecture"]
    k_c = k_per_level["cath_class"]

    def _fmt(v: object) -> str:
        try:
            f = float(v)  # type: ignore[arg-type]
            return "  N/A  " if f != f else f"{f:.4f}"
        except (TypeError, ValueError):
            return "  N/A  "

    def _fmts(v: object) -> str:
        try:
            f = float(v)  # type: ignore[arg-type]
            return "       " if f != f or f == 0 else f"±{f:.4f}"
        except (TypeError, ValueError):
            return "       "

    w = 98
    print()
    print("=" * w)
    print("  CATH Hierarchy Preservation — k-NN Purity (adaptive k, mean ± std)")
    print(f"  k: Homology={k_h}  Topology={k_t}  Architecture={k_a}  Class={k_c}")
    print("=" * w)
    hdr = (
        f"  {'Method':<22}  {'Homology(H)':>18}  {'Topology(T)':>18}  "
        f"{'Arch.(A)':>18}  {'Class(C)':>18}"
    )
    print(hdr)
    print("-" * w)

    for _, row in df.iterrows():
        mean_h = _fmt(row["khp_homology_mean"])
        std_h = _fmts(row["khp_homology_std"])
        mean_t = _fmt(row["khp_topology_mean"])
        std_t = _fmts(row["khp_topology_std"])
        mean_a = _fmt(row["khp_architecture_mean"])
        std_a = _fmts(row["khp_architecture_std"])
        mean_c = _fmt(row["khp_cath_class_mean"])
        std_c = _fmts(row["khp_cath_class_std"])

        method = str(row["method"])
        separator = "─" * w if method == "Random Baseline" else None
        if separator:
            print(separator)
        print(
            f"  {method:<22}  {mean_h} {std_h}  {mean_t} {std_t}  "
            f"{mean_a} {std_a}  {mean_c} {std_c}"
        )

    print("=" * w)
    print(
        "  Ordering within each column: Original Embedding ≥ DR method > Random Baseline\n"
        "  Observed level ordering (class > arch > topo > homo) is driven by group size.\n"
    )
