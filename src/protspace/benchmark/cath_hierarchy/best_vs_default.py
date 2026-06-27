"""Best-vs-default tuning-gain analysis for the CATH evaluation.

A lightweight post-processing step that quantifies how much hyperparameter
tuning improves each CATH hierarchy level relative to the default/baseline
configuration, per DR method.

For each method and hierarchy level:

* **default_knn** — seed-averaged k-NN purity of the *baseline* configuration
  defined in :mod:`protspace.benchmark.robustness.config`.  These baselines are
  identical to the :class:`~protspace.utils.constants.DimensionReductionConfig`
  defaults, so the seed-mode results table
  (``cath_hierarchy_results.csv``) is exactly the baseline config, seed-averaged.
* **best_knn** — maximum seed-averaged k-NN purity across all hyperparameter
  configurations (``hyperparam_seed_averaged.csv``).
* **improvement** — ``best_knn - default_knn``.

No embeddings or metrics are recomputed here; this only reads existing
seed-averaged result tables.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SEED_AVERAGED_FILE = "hyperparam_seed_averaged.csv"
DEFAULT_RESULTS_FILE = "cath_hierarchy_results.csv"
DETAIL_OUTPUT_FILE = "best_vs_default_knn.csv"
SUMMARY_OUTPUT_FILE = "best_vs_default_summary.csv"

# (column key in result tables, human-readable hierarchy level)
LEVELS: list[tuple[str, str]] = [
    ("cath_class", "Class"),
    ("architecture", "Architecture"),
    ("topology", "Topology"),
    ("homology", "Homology"),
]
_REFERENCE_METHODS = {"Original Embedding", "Random Baseline"}


def compute_best_vs_default(
    seedavg_df: pd.DataFrame, default_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the best-vs-default tuning gains.

    Parameters
    ----------
    seedavg_df:
        Hyperparameter seed-averaged results (one row per method × HP config)
        with ``khp_{level}_mean`` columns.  If it contains the baseline config
        (rows where ``param_name == "baseline"``), the default scores are read
        from there and *default_df* is not needed.
    default_df:
        Optional seed-mode aggregated results (one row per method) used as the
        default source when the baseline config is absent from *seedavg_df*
        (older sweeps).  Reference rows are ignored.

    Returns
    -------
    ``(detail_df, summary_df)`` where *detail_df* has columns ``method``,
    ``hierarchy_level``, ``default_knn``, ``best_knn``, ``improvement`` (one row
    per method × level) and *summary_df* has ``method``, ``max_improvement``,
    ``hierarchy_level_of_max_improvement`` (one row per method).
    """
    mean_cols = [f"khp_{key}_mean" for key, _ in LEVELS]

    # "best" is the max over ALL configurations (baseline included), so it can
    # never fall below the default → improvement is the realisable tuning gain.
    best = seedavg_df.groupby("method", sort=False)[mean_cols].max()

    # "default" preferentially from the in-table baseline config; else seed-mode.
    if (
        "param_name" in seedavg_df.columns
        and (seedavg_df["param_name"] == "baseline").any()
    ):
        default = (
            seedavg_df[seedavg_df["param_name"] == "baseline"]
            .drop_duplicates("method", keep="first")
            .set_index("method")
        )
    elif default_df is not None:
        default = default_df[~default_df["method"].isin(_REFERENCE_METHODS)].set_index(
            "method"
        )
    else:
        raise ValueError(
            "No baseline config in seedavg_df and no default_df provided — "
            "cannot determine default scores."
        )

    rows: list[dict] = []
    for method in best.index:
        if method not in default.index:
            logger.warning("Method %s has no default/baseline row — skipping", method)
            continue
        for key, label in LEVELS:
            col = f"khp_{key}_mean"
            d = float(default.loc[method, col])
            b = float(best.loc[method, col])
            rows.append(
                {
                    "method": method,
                    "hierarchy_level": label,
                    "default_knn": d,
                    "best_knn": b,
                    "improvement": b - d,
                }
            )

    detail_df = pd.DataFrame(rows)

    # max improvement per method (idxmax keeps the first level on ties)
    idx = detail_df.groupby("method", sort=False)["improvement"].idxmax()
    summary_df = (
        detail_df.loc[idx, ["method", "improvement", "hierarchy_level"]]
        .rename(
            columns={
                "improvement": "max_improvement",
                "hierarchy_level": "hierarchy_level_of_max_improvement",
            }
        )
        .reset_index(drop=True)
    )
    return detail_df, summary_df


def run_best_vs_default_summary(
    results_dir: Path, default_results_csv: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read existing seed-averaged tables, write both output CSVs.

    Parameters
    ----------
    results_dir:
        Directory holding ``hyperparam_seed_averaged.csv`` and where the two
        output CSVs are written.
    default_results_csv:
        Path to the seed-mode results (default config).  Defaults to
        ``<results_dir>/../cath_hierarchy_results.csv``.

    Returns
    -------
    ``(detail_df, summary_df)``.
    """
    results_dir = Path(results_dir)
    seedavg_path = results_dir / SEED_AVERAGED_FILE
    if not seedavg_path.exists():
        raise FileNotFoundError(
            f"Seed-averaged results not found: {seedavg_path}. "
            "Run the hyperparameter evaluation first (--mode hyperparam)."
        )

    seedavg_df = pd.read_csv(seedavg_path)

    # The seed-mode results are only needed as a fallback for older sweeps that
    # don't contain the baseline config. Newer sweeps include it in-table.
    has_baseline = (
        "param_name" in seedavg_df.columns
        and (seedavg_df["param_name"] == "baseline").any()
    )
    if default_results_csv is None:
        default_results_csv = results_dir.parent / DEFAULT_RESULTS_FILE
    default_results_csv = Path(default_results_csv)
    default_df = None
    if default_results_csv.exists():
        default_df = pd.read_csv(default_results_csv)
    elif not has_baseline:
        raise FileNotFoundError(
            f"No baseline config in {seedavg_path} and no seed-mode results at "
            f"{default_results_csv}. Run the hyperparameter evaluation "
            "(--mode hyperparam) or the seed evaluation (--mode seed) first."
        )

    detail_df, summary_df = compute_best_vs_default(seedavg_df, default_df)

    detail_path = results_dir / DETAIL_OUTPUT_FILE
    summary_path = results_dir / SUMMARY_OUTPUT_FILE
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    logger.info("Best-vs-default detail  → %s", detail_path)
    logger.info("Best-vs-default summary → %s", summary_path)
    return detail_df, summary_df
