"""Hyperparameter-robustness evaluation for CATH hierarchy preservation.

Research question
-----------------
How sensitive are the CATH hierarchy k-NN purity scores (Class, Architecture,
Topology, Homology) to the choice of dimensionality-reduction hyperparameters?

This is a *parallel* workflow to the seed-robustness evaluation in
:mod:`~protspace.benchmark.cath_hierarchy.run` — it does **not** replace it.
It reuses the exact same metric, adaptive-k, baseline and label-loading code,
and draws the hyperparameter grids verbatim from
:mod:`protspace.benchmark.robustness.config`.

Experiment design
-----------------
For each DR method we evaluate the **full Cartesian product** of the
hyperparameter grids declared in ``METHOD_CONFIGS[method]["hyperparam_experiments"]``
(e.g. UMAP: ``n_neighbors × min_dist``; PaCMAP: ``n_neighbors × mn_ratio × fp_ratio``).
Each combination starts from the method's ``baseline`` configuration, overrides
only the participating hyperparameters, and is run **once** using the baseline
``random_state`` — there is **no seed sweep and no averaging across seeds**.

Each combination therefore yields a single KHP value per CATH level.  Across the
combinations we report mean ± std per method (the std now reflects spread across
hyperparameter combinations, used for the plot error bars).

Three result files are produced:

* ``hyperparam_raw.csv``            — one row per (method, HP combination).
* ``hyperparam_seed_averaged.csv``  — one row per (method, HP combination).
  (Filename kept for downstream compatibility; with single-run combinations each
  ``khp_{level}_mean`` is the run value and ``khp_{level}_std`` is 0.)
* ``hyperparam_aggregated.csv``     — one row per method (plus references);
  same schema as the seed workflow so :func:`plot_khp_results` can render it.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from protspace.benchmark.cath_hierarchy.hierarchy_preservation import (
    compute_hierarchy_preservation,
)
from protspace.benchmark.cath_hierarchy.metrics import (
    LEVEL_KEYS,
    compute_adaptive_k,
    compute_group_stats,
    compute_khp_baselines,
    knn_hierarchy_purity_adaptive,
)
from protspace.benchmark.cath_hierarchy.run import (
    _make_baseline_row,
    _make_reference_row,
    _print_group_stats,
    load_labels,
)
from protspace.benchmark.harness import benchmark_method
from protspace.benchmark.robustness.config import METHOD_CONFIGS, get_method_config
from protspace.utils.constants import DimensionReductionConfig

logger = logging.getLogger(__name__)


def _iter_hyperparam_combinations(method: str):
    """Yield hyperparameter-override dicts for a method's Cartesian grid.

    Produces the full Cartesian product of the value lists declared in
    ``METHOD_CONFIGS[method]["hyperparam_experiments"]`` (e.g. for UMAP this is
    ``n_neighbors × min_dist``).  Each yielded dict maps participating parameter
    names to one value of the combination; non-participating parameters keep
    their baseline values.
    """
    experiments = get_method_config(method)["hyperparam_experiments"]
    names = list(experiments.keys())
    value_lists = [experiments[n] for n in names]
    for combo in itertools.product(*value_lists):
        yield dict(zip(names, combo, strict=True))


def _combo_label(overrides: dict) -> str:
    """Human-readable label for an HP combination, e.g. ``n_neighbors=3;min_dist=0.0``."""
    if not overrides:
        return "baseline=default"
    return ";".join(f"{k}={v}" for k, v in overrides.items())


def _slug(label: str) -> str:
    """Filesystem-safe slug for a config label."""
    return label.replace("/", "_")


def _evaluate_combo(
    method: str,
    overrides: dict,
    is_baseline: bool,
    baseline_cfg: dict,
    embeddings: np.ndarray,
    cath_labels,
    k_per_level: dict[str, int],
) -> dict:
    """Run a single HP combination **once** with the baseline ``random_state``.

    Starts from ``baseline_cfg`` and overrides only the parameters in
    ``overrides``.  No seed sweep and no averaging — the baseline's
    ``random_state`` (or the :class:`DimensionReductionConfig` default when the
    baseline does not set one) is used verbatim.

    Returns a dict with ``"raw"`` (one row), ``"seed_averaged"`` (one row, kept
    under that key for downstream compatibility) and ``"config_mean"`` (level →
    KHP, for cross-combination aggregation).
    """
    config_label = _combo_label(overrides)
    cfg_dict = dict(baseline_cfg)
    cfg_dict.update(overrides)  # only the participating hyperparameters change
    dr_config = DimensionReductionConfig(**cfg_dict)
    random_state = cfg_dict.get("random_state", DimensionReductionConfig().random_state)

    result = benchmark_method(
        embeddings=embeddings,
        method=method,
        config=dr_config,
        normalize=False,
        metric_functions=None,
    )
    khp = knn_hierarchy_purity_adaptive(result.projection, cath_labels, k_per_level)

    param_name = "baseline" if is_baseline else "combo"
    raw_row: dict = {
        "method": method,
        "param_name": param_name,
        "param_value": config_label,
        "config": config_label,
        "random_state": random_state,
        **overrides,
        **{f"khp_{lvl}": khp[f"khp_{lvl}"] for lvl in LEVEL_KEYS},
    }

    config_mean: dict[str, float] = {
        lvl: float(khp[f"khp_{lvl}"]) for lvl in LEVEL_KEYS
    }
    srow: dict = {
        "method": method,
        "param_name": param_name,
        "param_value": config_label,
        "config": config_label,
        "n_runs": 1,
        **overrides,
    }
    for lvl in LEVEL_KEYS:
        srow[f"khp_{lvl}_mean"] = config_mean[lvl]
        srow[f"khp_{lvl}_std"] = 0.0  # single run → no within-combination spread

    return {"raw": [raw_row], "seed_averaged": srow, "config_mean": config_mean}


def _evaluate_method(
    method: str,
    embeddings: np.ndarray,
    cath_labels,
    k_per_level: dict[str, int],
    baselines: dict[str, float],
    cache_dir: Path | None = None,
) -> dict:
    """Run the full Cartesian HP sweep for one method, caching each combo to disk.

    Each hyperparameter combination is run once with the baseline
    ``random_state``.  If ``cache_dir`` is given, each combination's result is
    written to ``cache_dir/{method}/{combo_slug}.json`` as soon as it completes
    and reused on subsequent runs (resumable at combination granularity).

    Returns a dict with keys ``"raw"`` (list of per-combination rows),
    ``"seed_averaged"`` (list of per-combination rows) and ``"aggregated"`` (a
    single method row, mean ± std across HP combinations).
    """
    if method in METHOD_CONFIGS:
        cfg = get_method_config(method)
        baseline_cfg = cfg["baseline"]
        grid_params = list(cfg["hyperparam_experiments"].keys())
        # The baseline combination may itself appear in the Cartesian grid (the
        # grids now include the baseline values).  Emit one explicitly-labelled
        # baseline row (param_name="baseline", needed for best-vs-default) and
        # skip the identical Cartesian combo so it is not run/counted twice.
        baseline_combo = {p: baseline_cfg.get(p) for p in grid_params}
        combos = [({}, True)]
        for overrides in _iter_hyperparam_combinations(method):
            if overrides == baseline_combo:
                continue
            combos.append((overrides, False))
    else:
        # No hyperparameter grid (e.g. PCA, deterministic): baseline only.
        combos = [({}, True)]
        baseline_cfg = {}
    logger.info(
        "Method %s — %d configurations (baseline + Cartesian product), 1 run each",
        method,
        len(combos),
    )

    method_cache = cache_dir / method if cache_dir is not None else None
    if method_cache is not None:
        method_cache.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    seedavg_rows: list[dict] = []
    per_config_means: list[dict[str, float]] = []

    for overrides, is_baseline in combos:
        label = _combo_label(overrides)
        cfile = (
            method_cache / f"{_slug(label)}.json" if method_cache is not None else None
        )
        if cfile is not None and cfile.exists():
            logger.info("  %s — cached", label)
            cres = json.loads(cfile.read_text())
        else:
            logger.info("  %s — running 1 config", label)
            cres = _evaluate_combo(
                method,
                overrides,
                is_baseline,
                baseline_cfg,
                embeddings,
                cath_labels,
                k_per_level,
            )
            if cfile is not None:
                cfile.write_text(json.dumps(cres))

        raw_rows.extend(cres["raw"])
        seedavg_rows.append(cres["seed_averaged"])
        per_config_means.append(cres["config_mean"])

    # --- aggregate across HP combinations (mean ± std) ---
    agg_row: dict = {"method": method, "n_configs": len(per_config_means)}
    for lvl in LEVEL_KEYS:
        cfg_vals = np.array([c[lvl] for c in per_config_means])
        agg_row[f"khp_{lvl}_mean"] = float(np.mean(cfg_vals))
        agg_row[f"khp_{lvl}_std"] = float(
            np.std(cfg_vals, ddof=1) if len(cfg_vals) > 1 else 0.0
        )
        agg_row[f"baseline_{lvl}"] = baselines[f"baseline_{lvl}"]
        agg_row[f"k_{lvl}"] = k_per_level[lvl]

    return {"raw": raw_rows, "seed_averaged": seedavg_rows, "aggregated": agg_row}


def run_cath_hyperparam_evaluation(
    h5_path: Path,
    output_dir: Path,
    methods: list[str] | None = None,
    max_proteins: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate hyperparameter sensitivity of CATH hierarchy KHP scores.

    Runs the full Cartesian product of each method's hyperparameter grid, each
    combination once with the baseline ``random_state`` (no seed sweep).

    Parameters
    ----------
    h5_path:
        Path to ``data/cath_s40/prot_t5.h5``.
    output_dir:
        Directory for the three result CSVs (and, via the CLI, plots).
    methods:
        DR methods to evaluate.  Defaults to PCA (baseline only) plus every
        method that has a hyperparameter grid in the robustness config
        (``umap``, ``tsne``, ``pacmap``, ``localmap``, ``mds``).
    max_proteins:
        Subsample to this many proteins for faster testing.

    Returns
    -------
    dict with keys ``"raw"``, ``"seed_averaged"``, ``"aggregated"`` and
    ``"hierarchy_preservation"`` mapping to the corresponding DataFrames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        # PCA first (deterministic, no HP grid → no error bar), then the
        # methods that do have hyperparameter grids in the robustness config.
        methods = ["pca", *METHOD_CONFIGS.keys()]

    # ------------------------------------------------------------------
    # 1. Load embeddings + labels (shared with the seed workflow)
    # ------------------------------------------------------------------
    cath_labels = load_labels(h5_path, max_proteins=max_proteins)
    embeddings = cath_labels.embeddings

    # ------------------------------------------------------------------
    # 2. Adaptive k, random baseline, Original Embedding reference
    # ------------------------------------------------------------------
    group_stats = compute_group_stats(cath_labels)
    k_per_level = compute_adaptive_k(cath_labels)
    _print_group_stats(group_stats, k_per_level)

    baselines = compute_khp_baselines(cath_labels)

    logger.info(
        "Computing KHP in original %d-dim embedding space …", embeddings.shape[1]
    )
    orig_khp = knn_hierarchy_purity_adaptive(embeddings, cath_labels, k_per_level)

    # ------------------------------------------------------------------
    # 3. Hyperparameter sweep (resumable — cached per method and per config)
    # ------------------------------------------------------------------
    partial_dir = output_dir / "_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    seedavg_rows: list[dict] = []
    agg_rows: list[dict] = []

    for method in methods:
        pfile = partial_dir / f"{method}.json"
        if pfile.exists():
            logger.info("Method %s — loading cached partial result", method)
            partial = json.loads(pfile.read_text())
        else:
            # cache_dir enables per-combination resumption within the method
            partial = _evaluate_method(
                method,
                embeddings,
                cath_labels,
                k_per_level,
                baselines,
                cache_dir=partial_dir,
            )
            pfile.write_text(json.dumps(partial))
            logger.info("Method %s — cached partial result → %s", method, pfile)

        raw_rows.extend(partial["raw"])
        seedavg_rows.extend(partial["seed_averaged"])
        agg_rows.append(partial["aggregated"])

    # ------------------------------------------------------------------
    # 4. Assemble + save
    # ------------------------------------------------------------------
    raw_df = pd.DataFrame(raw_rows)
    seedavg_df = pd.DataFrame(seedavg_rows)

    # Plot-ready aggregated DF: references first/last, same schema as run.py
    orig_row = _make_reference_row(
        "Original Embedding", orig_khp, baselines, k_per_level, n_seeds=1
    )
    baseline_row = _make_baseline_row(baselines, k_per_level)
    agg_df = pd.DataFrame([orig_row] + agg_rows + [baseline_row])

    raw_path = output_dir / "hyperparam_raw.csv"
    seedavg_path = output_dir / "hyperparam_seed_averaged.csv"
    agg_path = output_dir / "hyperparam_aggregated.csv"
    raw_df.to_csv(raw_path, index=False)
    seedavg_df.to_csv(seedavg_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    logger.info("Raw results            → %s", raw_path)
    logger.info("Seed-averaged results  → %s", seedavg_path)
    logger.info("HP-aggregated results  → %s", agg_path)

    # Hierarchy-ordering preservation summary (post-processing of seed-averaged)
    preservation_df = compute_hierarchy_preservation(seedavg_df)
    preservation_path = output_dir / "hierarchy_preservation.csv"
    preservation_df.to_csv(preservation_path, index=False)
    logger.info("Hierarchy preservation → %s", preservation_path)

    _print_table(agg_df, k_per_level)
    _print_preservation(preservation_df)

    return {
        "raw": raw_df,
        "seed_averaged": seedavg_df,
        "aggregated": agg_df,
        "hierarchy_preservation": preservation_df,
    }


def _print_table(df: pd.DataFrame, k_per_level: dict[str, int]) -> None:
    """Console summary of the HP-aggregated table (mean ± std across configs)."""

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
    print("  CATH Hierarchy Preservation — k-NN Purity (mean ± std across HP configs)")
    print(
        f"  k: Homology={k_per_level['homology']}  Topology={k_per_level['topology']}  "
        f"Architecture={k_per_level['architecture']}  Class={k_per_level['cath_class']}"
    )
    print("=" * w)
    print(
        f"  {'Method':<22}  {'Homology(H)':>18}  {'Topology(T)':>18}  "
        f"{'Arch.(A)':>18}  {'Class(C)':>18}"
    )
    print("-" * w)
    for _, row in df.iterrows():
        method = str(row["method"])
        if method == "Random Baseline":
            print("─" * w)
        print(
            f"  {method:<22}  "
            f"{_fmt(row['khp_homology_mean'])} {_fmts(row['khp_homology_std'])}  "
            f"{_fmt(row['khp_topology_mean'])} {_fmts(row['khp_topology_std'])}  "
            f"{_fmt(row['khp_architecture_mean'])} {_fmts(row['khp_architecture_std'])}  "
            f"{_fmt(row['khp_cath_class_mean'])} {_fmts(row['khp_cath_class_std'])}"
        )
    print("=" * w)
    print(
        "  Error bars (±) are the std across hyperparameter configurations.\n"
        "  References (Original Embedding, Random Baseline) have no HP variation.\n"
    )


def _print_preservation(df: pd.DataFrame) -> None:
    """Console summary of hierarchy-ordering preservation per method."""
    w = 64
    print("=" * w)
    print("  Hierarchy-Ordering Preservation per DR Method")
    print("  (config preserves ordering iff Class > Arch > Topo > Homo)")
    print("=" * w)
    print(f"  {'Method':<14}  {'#Configs':>8}  {'#Preserved':>10}  {'Percent':>8}")
    print("-" * w)
    for _, row in df.iterrows():
        print(
            f"  {str(row['method']):<14}  {int(row['num_configurations']):>8}  "
            f"{int(row['num_preserved']):>10}  "
            f"{row['hierarchy_preservation_percent']:>7.1f}%"
        )
    print("=" * w + "\n")
