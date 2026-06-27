"""Hierarchy-ordering preservation summary for the CATH evaluation.

A lightweight post-processing step over the **seed-averaged** hyperparameter
results.  A single hyperparameter configuration is said to *preserve the CATH
hierarchy ordering* when its seed-averaged k-NN purities satisfy::

    knn_Class > knn_Architecture > knn_Topology > knn_Homology

For each DR method we then report what fraction of its configurations preserve
this ordering.  No embeddings or metrics are recomputed here — this only reads
the already-aggregated ``hyperparam_seed_averaged.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SEED_AVERAGED_FILE = "hyperparam_seed_averaged.csv"
OUTPUT_FILE = "hierarchy_preservation.csv"


def compute_hierarchy_preservation(seedavg_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize hierarchy-ordering preservation per DR method.

    Parameters
    ----------
    seedavg_df:
        Seed-averaged results (one row per method × HP configuration) with the
        columns ``khp_cath_class_mean``, ``khp_architecture_mean``,
        ``khp_topology_mean`` and ``khp_homology_mean``.

    Returns
    -------
    DataFrame with columns ``method``, ``num_configurations``,
    ``num_preserved`` and ``hierarchy_preservation_percent``, one row per method
    (method order preserved from the input).
    """
    df = seedavg_df.copy()
    df["hierarchy_preserved"] = (
        (df["khp_cath_class_mean"] > df["khp_architecture_mean"])
        & (df["khp_architecture_mean"] > df["khp_topology_mean"])
        & (df["khp_topology_mean"] > df["khp_homology_mean"])
    )

    summary = (
        df.groupby("method", sort=False)["hierarchy_preserved"]
        .agg(num_configurations="size", num_preserved="sum")
        .reset_index()
    )
    summary["num_preserved"] = summary["num_preserved"].astype(int)
    summary["hierarchy_preservation_percent"] = (
        100.0 * summary["num_preserved"] / summary["num_configurations"]
    )
    return summary


def run_hierarchy_preservation_summary(results_dir: Path) -> pd.DataFrame:
    """Read the seed-averaged CSV in *results_dir* and write the summary CSV.

    Returns the summary DataFrame.
    """
    results_dir = Path(results_dir)
    seedavg_path = results_dir / SEED_AVERAGED_FILE
    if not seedavg_path.exists():
        raise FileNotFoundError(
            f"Seed-averaged results not found: {seedavg_path}. "
            "Run the hyperparameter evaluation first (--mode hyperparam)."
        )

    seedavg_df = pd.read_csv(seedavg_path)
    summary = compute_hierarchy_preservation(seedavg_df)

    out_path = results_dir / OUTPUT_FILE
    summary.to_csv(out_path, index=False)
    logger.info("Hierarchy-preservation summary → %s", out_path)
    return summary
