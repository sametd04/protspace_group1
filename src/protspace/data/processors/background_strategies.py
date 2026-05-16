"""ρPCA background-selection strategies.

Each strategy is a pure function with signature

    strategy(*, pool_embeddings, pool_headers, pool_annotations,
             target_embeddings, target_headers, target_annotations,
             rng, **kwargs) -> tuple[ndarray, dict]

returning (X_background, diagnostic_metadata). All five strategies live here
so the pipeline orchestration code stays minimal.

References
----------
Carilli, Jackson & Pachter 2025  (paper 1, ρPCA)
Jackson, Carilli & Pachter 2026  (paper 2, k-ρPCA)

Design notes
------------
* Strategies operate on (pool, target) — never on a subsample of the target
  itself. The deprecated auto-split policies (random/uniform/outlier) have
  been removed because they violate the ρPCA premise: Σ_B must estimate a
  *different* distribution than Σ_T, otherwise the Rayleigh quotient is
  dominated by sampling and regularisation noise (see Carilli et al. 2025,
  Fig 1A; ProtSpace docs/contrastive.md).
* The pool can be either an external HDF5 (canonical, Mode 1) or the input
  itself (for `complement`, where target ⊂ input). The pipeline resolves
  this distinction before calling these functions.
* All strategies return the same metadata schema so diagnostics in run.log
  are uniform.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --- Strategy names (exported for the Literal type in constants) -----------

POOL = "pool"
COMPLEMENT = "complement"
LENGTH_MATCHED = "length_matched"
STRATIFIED = "stratified"
MIXED = "mixed"

STRATEGY_NAMES = (POOL, COMPLEMENT, LENGTH_MATCHED, STRATIFIED, MIXED)


@dataclass
class BackgroundSpec:
    """Result of a background-construction strategy.

    Attributes
    ----------
    X_background : (n_B, d) ndarray
        The background embedding matrix.
    X_target : (n_T, d) ndarray | None
        The target embedding matrix. None means "use the pipeline's input
        unchanged"; not-None means the strategy sliced the target out of a
        larger input matrix (only `complement` does this).
    source : str
        Strategy name; logged into run.log diagnostics.
    n_target : int
    n_background : int
    details : dict
        Strategy-specific diagnostic info (selected n per stratum, bin
        boundaries, target-value list, …). Logged but not otherwise used.
    """
    X_background: np.ndarray
    X_target: np.ndarray | None
    source: str
    n_target: int
    n_background: int
    details: dict[str, Any]


# ===========================================================================
# Strategy 1 — pool: background = the entire external pool, target unchanged.
# ===========================================================================

def strategy_pool(
    *,
    pool_embeddings: np.ndarray,
    pool_headers: list[str],
    target_embeddings: np.ndarray,
    target_headers: list[str],
    exclude_target_from_pool: bool = True,
    **_: Any,
) -> BackgroundSpec:
    """Use the full pool as background. Optionally remove target identifiers
    from the pool first (only matters when the pool was assembled to include
    the target, e.g. SwissProt-as-pool for a target that lives in SwissProt).
    """
    if exclude_target_from_pool:
        target_set = set(target_headers)
        keep = [i for i, h in enumerate(pool_headers) if h not in target_set]
        n_overlap = len(pool_headers) - len(keep)
        if n_overlap > 0:
            logger.info(
                "Strategy 'pool': removing %d target identifiers from pool "
                "(%d → %d background samples).",
                n_overlap, len(pool_headers), len(keep),
            )
        X_B = pool_embeddings[keep]
    else:
        n_overlap = 0
        X_B = pool_embeddings

    return BackgroundSpec(
        X_background=X_B,
        X_target=None,
        source=POOL,
        n_target=len(target_headers),
        n_background=X_B.shape[0],
        details={"n_overlap_excluded": n_overlap},
    )


# ===========================================================================
# Strategy 2 — complement: target = annotation-defined subset of input;
# background = input \ target.
# ===========================================================================

def strategy_complement(
    *,
    pool_embeddings: np.ndarray,
    pool_headers: list[str],
    pool_annotations: pd.DataFrame,
    target_annotation: str,
    target_values: list[str],
    **_: Any,
) -> BackgroundSpec:
    """Split a single input into target and background by annotation column.

    The "target" rows are those with `pool_annotations[target_annotation]`
    matching any value in `target_values`. The "background" rows are
    everything else.

    No external pool needed; the input *is* the pool. This is the natural
    "3FTx vs everything else" contrast when you load all of ToxProt.
    """
    if target_annotation not in pool_annotations.columns:
        raise ValueError(
            f"Annotation column {target_annotation!r} not found. "
            f"Available: {sorted(pool_annotations.columns)}"
        )

    target_values_set = {str(v) for v in target_values}
    column = pool_annotations[target_annotation].astype(str)
    target_mask = column.isin(target_values_set).to_numpy()

    n_target = int(target_mask.sum())
    n_background = int((~target_mask).sum())

    if n_target < 2:
        raise ValueError(
            f"Strategy 'complement': target_values={target_values!r} in column "
            f"{target_annotation!r} matched only {n_target} samples; need >= 2."
        )
    if n_background < 2:
        raise ValueError(
            f"Strategy 'complement': complement (input \\ target) has only "
            f"{n_background} samples; need >= 2. Check target_values."
        )

    X_T = pool_embeddings[target_mask]
    X_B = pool_embeddings[~target_mask]

    # Distribution of values in the background — useful for sanity-checking.
    bg_counts = column[~target_mask].value_counts().head(10).to_dict()

    logger.info(
        "Strategy 'complement': target=%d (%s in %s), background=%d (rest).",
        n_target, sorted(target_values_set), target_annotation, n_background,
    )

    return BackgroundSpec(
        X_background=X_B,
        X_target=X_T,
        source=COMPLEMENT,
        n_target=n_target,
        n_background=n_background,
        details={
            "target_annotation": target_annotation,
            "target_values": sorted(target_values_set),
            "background_top_values": bg_counts,
        },
    )


# ===========================================================================
# Strategy 3 — length_matched: sample from pool so length distribution
# matches the target's.
# ===========================================================================

def strategy_length_matched(
    *,
    pool_embeddings: np.ndarray,
    pool_headers: list[str],
    pool_lengths: np.ndarray,
    target_headers: list[str],
    target_lengths: np.ndarray,
    rng: np.random.Generator,
    samples_per_target: int = 3,
    n_bins: int = 20,
    exclude_target_from_pool: bool = True,
    **_: Any,
) -> BackgroundSpec:
    """Construct a length-matched background by histogram-matching.

    Method: bin both target and pool by length using shared bin edges
    derived from the target's quantiles. For each target sample (or for
    each bin proportionally), draw `samples_per_target` pool samples from
    the same bin without replacement.

    If a bin has fewer pool samples than requested, take all of them and
    emit a warning. This is the standard "frequency-matched control" used
    in case-control studies.
    """
    target_set = set(target_headers)
    if exclude_target_from_pool:
        keep = np.array(
            [i for i, h in enumerate(pool_headers) if h not in target_set],
            dtype=int,
        )
        pool_embeddings = pool_embeddings[keep]
        pool_lengths = pool_lengths[keep]
        pool_headers = [pool_headers[i] for i in keep]

    if pool_embeddings.shape[0] < 2:
        raise ValueError(
            "Strategy 'length_matched': pool has < 2 samples after excluding "
            "target identifiers."
        )

    # Bin edges from the target's empirical distribution so every target bin
    # is non-empty by construction. quantile bin edges adapt to skew.
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(target_lengths, quantiles)
    # Numerical guard: ensure strictly increasing edges (skewed integer
    # lengths can produce duplicates at low/high quantiles).
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1e-9
    bin_edges[0] -= 1e-9
    bin_edges[-1] += 1e-9

    target_bins = np.digitize(target_lengths, bin_edges[1:-1])
    pool_bins = np.digitize(pool_lengths, bin_edges[1:-1])

    target_counts = np.bincount(target_bins, minlength=n_bins)

    chosen: list[int] = []
    per_bin_used: list[int] = []
    per_bin_requested: list[int] = []
    underfilled_bins = 0

    for b in range(n_bins):
        requested = int(target_counts[b] * samples_per_target)
        per_bin_requested.append(requested)
        if requested == 0:
            per_bin_used.append(0)
            continue
        pool_idx_in_bin = np.where(pool_bins == b)[0]
        if pool_idx_in_bin.size == 0:
            per_bin_used.append(0)
            underfilled_bins += 1
            continue
        n_take = min(requested, pool_idx_in_bin.size)
        if n_take < requested:
            underfilled_bins += 1
        picked = rng.choice(pool_idx_in_bin, size=n_take, replace=False)
        chosen.extend(picked.tolist())
        per_bin_used.append(n_take)

    chosen_arr = np.array(sorted(set(chosen)), dtype=int)
    X_B = pool_embeddings[chosen_arr]

    if underfilled_bins > 0:
        logger.warning(
            "Strategy 'length_matched': %d / %d length bins were under-filled "
            "(pool didn't have enough samples in that length range). The "
            "background distribution will be slightly shifted away from the "
            "target's.",
            underfilled_bins, n_bins,
        )

    if X_B.shape[0] < 2:
        raise ValueError(
            f"Strategy 'length_matched': only {X_B.shape[0]} samples matched; "
            "need >= 2. Pool is too small or length ranges are too disjoint."
        )

    # KS-style summary of how close target and background distributions are.
    from scipy.stats import ks_2samp
    ks_stat, ks_p = ks_2samp(target_lengths, pool_lengths[chosen_arr])

    logger.info(
        "Strategy 'length_matched': %d target × %d samples/target → %d "
        "background samples (KS d=%.3f, p=%.2g).",
        len(target_headers), samples_per_target, X_B.shape[0], ks_stat, ks_p,
    )

    return BackgroundSpec(
        X_background=X_B,
        X_target=None,
        source=LENGTH_MATCHED,
        n_target=len(target_headers),
        n_background=int(X_B.shape[0]),
        details={
            "samples_per_target": samples_per_target,
            "n_bins": n_bins,
            "underfilled_bins": underfilled_bins,
            "ks_d_after_matching": float(ks_stat),
            "ks_p_after_matching": float(ks_p),
        },
    )


# ===========================================================================
# Strategy 4 — stratified: sample from pool proportionally by one or more
# nuisance annotation columns.
# ===========================================================================

def strategy_stratified(
    *,
    pool_embeddings: np.ndarray,
    pool_headers: list[str],
    pool_annotations: pd.DataFrame,
    target_headers: list[str],
    target_annotations: pd.DataFrame,
    stratify_by: list[str],
    rng: np.random.Generator,
    samples_per_target: int = 3,
    exclude_target_from_pool: bool = True,
    **_: Any,
) -> BackgroundSpec:
    """Sample the background so the joint distribution of `stratify_by`
    columns matches the target.

    For each stratum (a unique combination of values across stratify_by
    columns), sample `samples_per_target × target_count(stratum)` pool
    samples from the same stratum. Strata with no pool representation are
    skipped with a warning.
    """
    for col in stratify_by:
        if col not in pool_annotations.columns:
            raise ValueError(
                f"Strategy 'stratified': annotation column {col!r} not found "
                f"in pool. Available: {sorted(pool_annotations.columns)}"
            )
        if col not in target_annotations.columns:
            raise ValueError(
                f"Strategy 'stratified': annotation column {col!r} not found "
                f"in target. Available: {sorted(target_annotations.columns)}"
            )

    target_set = set(target_headers)
    if exclude_target_from_pool:
        keep = np.array(
            [i for i, h in enumerate(pool_headers) if h not in target_set],
            dtype=int,
        )
        pool_embeddings = pool_embeddings[keep]
        pool_annotations = pool_annotations.iloc[keep].reset_index(drop=True)

    # Join columns to form stratum keys.
    target_strata = (
        target_annotations[stratify_by].astype(str).agg("|".join, axis=1)
    )
    pool_strata = (
        pool_annotations[stratify_by].astype(str).agg("|".join, axis=1)
    )
    target_counts = target_strata.value_counts()

    chosen_global: list[int] = []
    matched_strata = 0
    unmatched_strata: list[str] = []
    underfilled: list[tuple[str, int, int]] = []

    for stratum, t_count in target_counts.items():
        requested = int(t_count * samples_per_target)
        if requested == 0:
            continue
        pool_idx = np.where(pool_strata.to_numpy() == stratum)[0]
        if pool_idx.size == 0:
            unmatched_strata.append(stratum)
            continue
        matched_strata += 1
        n_take = min(requested, pool_idx.size)
        if n_take < requested:
            underfilled.append((stratum, requested, int(pool_idx.size)))
        picked = rng.choice(pool_idx, size=n_take, replace=False)
        chosen_global.extend(picked.tolist())

    if not chosen_global:
        raise ValueError(
            "Strategy 'stratified': no pool samples matched any target "
            "stratum. Target and pool may use disjoint label sets in "
            f"{stratify_by}."
        )

    chosen_arr = np.array(sorted(set(chosen_global)), dtype=int)
    X_B = pool_embeddings[chosen_arr]

    if unmatched_strata:
        logger.warning(
            "Strategy 'stratified': %d target strata had no pool match (e.g. "
            "%s). Background will under-represent these strata.",
            len(unmatched_strata), unmatched_strata[:3],
        )
    if underfilled:
        logger.warning(
            "Strategy 'stratified': %d strata were under-filled (e.g. %s "
            "wanted %d got %d).",
            len(underfilled), *underfilled[0],
        )

    logger.info(
        "Strategy 'stratified' on %s: matched %d/%d strata, %d target × "
        "%d samples/target → %d background samples.",
        stratify_by, matched_strata, len(target_counts),
        len(target_headers), samples_per_target, X_B.shape[0],
    )

    return BackgroundSpec(
        X_background=X_B,
        X_target=None,
        source=STRATIFIED,
        n_target=len(target_headers),
        n_background=int(X_B.shape[0]),
        details={
            "stratify_by": list(stratify_by),
            "samples_per_target": samples_per_target,
            "matched_strata": matched_strata,
            "total_target_strata": int(len(target_counts)),
            "unmatched_strata_sample": unmatched_strata[:5],
            "underfilled_strata_count": len(underfilled),
        },
    )


# ===========================================================================
# Strategy 5 — mixed: stratify by N annotation columns AND/OR length bins.
# ===========================================================================

def strategy_mixed(
    *,
    pool_embeddings: np.ndarray,
    pool_headers: list[str],
    pool_annotations: pd.DataFrame,
    pool_lengths: np.ndarray | None,
    target_headers: list[str],
    target_annotations: pd.DataFrame,
    target_lengths: np.ndarray | None,
    stratify_by: list[str],
    match_length: bool,
    rng: np.random.Generator,
    samples_per_target: int = 3,
    n_length_bins: int = 10,
    exclude_target_from_pool: bool = True,
    **_: Any,
) -> BackgroundSpec:
    """Stratify by annotation columns AND length bins simultaneously.

    Length is treated as an additional stratification dimension: target
    length values are binned by quantile, pool lengths are assigned to the
    same bins, and the bin index is appended to the stratum key.
    """
    if not stratify_by and not match_length:
        raise ValueError(
            "Strategy 'mixed' requires at least one of `stratify_by` or "
            "`match_length=True`. Got neither."
        )
    if match_length and (pool_lengths is None or target_lengths is None):
        raise ValueError(
            "Strategy 'mixed' with `match_length=True` requires sequence "
            "lengths from a FASTA (-f) or annotation column."
        )

    # Build target/pool stratum keys.
    if stratify_by:
        target_keys = (
            target_annotations[stratify_by].astype(str).agg("|".join, axis=1)
        )
        pool_keys = (
            pool_annotations[stratify_by].astype(str).agg("|".join, axis=1)
        )
    else:
        target_keys = pd.Series([""] * len(target_headers), dtype=str)
        pool_keys = pd.Series([""] * len(pool_headers), dtype=str)

    if match_length:
        quantiles = np.linspace(0.0, 1.0, n_length_bins + 1)
        bin_edges = np.quantile(target_lengths, quantiles)
        for i in range(1, len(bin_edges)):
            if bin_edges[i] <= bin_edges[i - 1]:
                bin_edges[i] = bin_edges[i - 1] + 1e-9
        bin_edges[0] -= 1e-9
        bin_edges[-1] += 1e-9
        target_lbin = np.digitize(target_lengths, bin_edges[1:-1])
        pool_lbin = np.digitize(pool_lengths, bin_edges[1:-1])
        target_keys = target_keys + "|L" + pd.Series(target_lbin.astype(str))
        pool_keys = pool_keys + "|L" + pd.Series(pool_lbin.astype(str))

    target_set = set(target_headers)
    if exclude_target_from_pool:
        keep = np.array(
            [i for i, h in enumerate(pool_headers) if h not in target_set],
            dtype=int,
        )
        pool_embeddings = pool_embeddings[keep]
        pool_keys = pool_keys.iloc[keep].reset_index(drop=True)

    target_counts = target_keys.value_counts()
    chosen_global: list[int] = []
    matched_strata = 0
    unmatched_strata: list[str] = []

    for stratum, t_count in target_counts.items():
        requested = int(t_count * samples_per_target)
        if requested == 0:
            continue
        pool_idx = np.where(pool_keys.to_numpy() == stratum)[0]
        if pool_idx.size == 0:
            unmatched_strata.append(stratum)
            continue
        matched_strata += 1
        n_take = min(requested, pool_idx.size)
        picked = rng.choice(pool_idx, size=n_take, replace=False)
        chosen_global.extend(picked.tolist())

    if not chosen_global:
        raise ValueError(
            "Strategy 'mixed': no pool samples matched any joint stratum. "
            "Constraints may be too tight; try reducing n_length_bins or "
            "stratify_by."
        )

    chosen_arr = np.array(sorted(set(chosen_global)), dtype=int)
    X_B = pool_embeddings[chosen_arr]

    if unmatched_strata:
        logger.warning(
            "Strategy 'mixed': %d / %d joint strata had no pool match.",
            len(unmatched_strata), len(target_counts),
        )

    logger.info(
        "Strategy 'mixed' (stratify_by=%s, match_length=%s): matched %d/%d "
        "strata → %d background samples.",
        stratify_by or "[]", match_length, matched_strata,
        len(target_counts), X_B.shape[0],
    )

    return BackgroundSpec(
        X_background=X_B,
        X_target=None,
        source=MIXED,
        n_target=len(target_headers),
        n_background=int(X_B.shape[0]),
        details={
            "stratify_by": list(stratify_by),
            "match_length": bool(match_length),
            "n_length_bins": n_length_bins if match_length else None,
            "samples_per_target": samples_per_target,
            "matched_strata": matched_strata,
            "total_target_strata": int(len(target_counts)),
            "unmatched_strata_sample": unmatched_strata[:5],
        },
    )


# ===========================================================================
# Dispatch
# ===========================================================================

_STRATEGIES = {
    POOL: strategy_pool,
    COMPLEMENT: strategy_complement,
    LENGTH_MATCHED: strategy_length_matched,
    STRATIFIED: strategy_stratified,
    MIXED: strategy_mixed,
}


def build_background(name: str, **kwargs) -> BackgroundSpec:
    """Top-level dispatch. Resolves the strategy by name."""
    if name not in _STRATEGIES:
        raise ValueError(
            f"Unknown background strategy {name!r}. "
            f"Available: {sorted(_STRATEGIES)}"
        )
    return _STRATEGIES[name](**kwargs)