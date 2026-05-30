"""ρPCA background-selection strategies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

POOL = "pool"
COMPLEMENT = "complement"
LENGTH_MATCHED = "length_matched"
STRATIFIED = "stratified"
MIXED = "mixed"
ISOLATE = "isolate"

STRATEGY_NAMES = (POOL, COMPLEMENT, LENGTH_MATCHED, STRATIFIED, MIXED, ISOLATE)

@dataclass
class BackgroundSpec:
    X_background: np.ndarray
    X_target: np.ndarray | None
    source: str
    n_target: int
    n_background: int
    details: dict[str, Any]

def _align_pool(pool_headers, pool_annotations):
    """Ensure annotations match the header order and length exactly."""
    if 'identifier' not in pool_annotations.columns:
        raise ValueError("Annotation dataframe missing 'identifier' column.")
    # Reindex annotations to match headers exactly
    return pool_annotations.set_index('identifier').reindex(pool_headers).reset_index()

def strategy_pool(*, pool_embeddings: np.ndarray, pool_headers: list[str], target_embeddings: np.ndarray, target_headers: list[str], exclude_target_from_pool: bool = True, **_: Any) -> BackgroundSpec:
    if exclude_target_from_pool:
        target_set = set(target_headers)
        keep = [i for i, h in enumerate(pool_headers) if h not in target_set]
        X_B = pool_embeddings[keep]
    else:
        X_B = pool_embeddings
    return BackgroundSpec(X_background=X_B, X_target=None, source=POOL, n_target=len(target_headers), n_background=X_B.shape[0], details={})

def strategy_complement(*, pool_embeddings: np.ndarray, pool_headers: list[str], pool_annotations: pd.DataFrame, target_annotation: str, target_values: list[str], **_: Any) -> BackgroundSpec:
    annot = _align_pool(pool_headers, pool_annotations)
    if target_annotation not in annot.columns: raise ValueError(f"Annotation {target_annotation!r} not found.")
    target_values_set = {str(v) for v in target_values}
    column = annot[target_annotation].astype(str)
    target_mask = column.isin(target_values_set).to_numpy()
    X_T, X_B = pool_embeddings[target_mask], pool_embeddings[~target_mask]
    return BackgroundSpec(X_background=X_B, X_target=X_T, source=COMPLEMENT, n_target=int(target_mask.sum()), n_background=int((~target_mask).sum()), details={})

def strategy_length_matched(*, pool_embeddings: np.ndarray, pool_headers: list[str], pool_lengths: np.ndarray, target_headers: list[str], target_lengths: np.ndarray, rng: np.random.Generator, samples_per_target: int = 3, n_bins: int = 20, exclude_target_from_pool: bool = True, **_: Any) -> BackgroundSpec:
    target_set = set(target_headers)
    if exclude_target_from_pool:
        keep = np.array([i for i, h in enumerate(pool_headers) if h not in target_set], dtype=int)
        pool_embeddings, pool_lengths = pool_embeddings[keep], pool_lengths[keep]
    
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    bin_edges = np.quantile(target_lengths, quantiles)
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i - 1]: bin_edges[i] = bin_edges[i - 1] + 1e-9
    bin_edges[0] -= 1e-9; bin_edges[-1] += 1e-9
    
    target_bins = np.digitize(target_lengths, bin_edges[1:-1])
    pool_bins = np.digitize(pool_lengths, bin_edges[1:-1])
    target_counts = np.bincount(target_bins, minlength=n_bins)
    
    chosen = []
    for b in range(n_bins):
        requested = int(target_counts[b] * samples_per_target)
        if requested == 0: continue
        pool_idx = np.where(pool_bins == b)[0]
        n_take = min(requested, pool_idx.size)
        chosen.extend(rng.choice(pool_idx, size=n_take, replace=False).tolist())
    
    X_B = pool_embeddings[np.array(sorted(set(chosen)), dtype=int)]
    return BackgroundSpec(X_background=X_B, X_target=None, source=LENGTH_MATCHED, n_target=len(target_headers), n_background=int(X_B.shape[0]), details={})

def strategy_stratified(*, pool_embeddings: np.ndarray, pool_headers: list[str], pool_annotations: pd.DataFrame, target_headers: list[str], target_annotations: pd.DataFrame, stratify_by: list[str], rng: np.random.Generator, samples_per_target: int = 3, exclude_target_from_pool: bool = True, **_: Any) -> BackgroundSpec:
    annot = _align_pool(pool_headers, pool_annotations)
    for col in stratify_by:
        if col not in target_annotations.columns: raise ValueError(f"Annotation {col!r} not found in target.")
        if col not in annot.columns: raise ValueError(f"Annotation {col!r} not found in pool.")
        
    if exclude_target_from_pool:
        target_set = set(target_headers)
        keep = np.array([i for i, h in enumerate(pool_headers) if h not in target_set], dtype=int)
        pool_embeddings = pool_embeddings[keep]
        annot = annot.iloc[keep].reset_index(drop=True)
        
    target_strata = target_annotations[stratify_by].astype(str).agg("|".join, axis=1)
    pool_strata = annot[stratify_by].astype(str).agg("|".join, axis=1)
    target_counts = target_strata.value_counts()
    
    chosen = []
    for stratum, t_count in target_counts.items():
        requested = int(t_count * samples_per_target)
        if requested == 0: continue
        pool_idx = np.where(pool_strata.to_numpy() == stratum)[0]
        n_take = min(requested, pool_idx.size)
        chosen.extend(rng.choice(pool_idx, size=n_take, replace=False).tolist())
        
    X_B = pool_embeddings[np.array(sorted(set(chosen)), dtype=int)]
    return BackgroundSpec(X_background=X_B, X_target=None, source=STRATIFIED, n_target=len(target_headers), n_background=int(X_B.shape[0]), details={})

def strategy_mixed(*, pool_embeddings: np.ndarray, pool_headers: list[str], pool_annotations: pd.DataFrame, pool_lengths: np.ndarray | None, target_headers: list[str], target_annotations: pd.DataFrame, target_lengths: np.ndarray | None, stratify_by: list[str], match_length: bool, rng: np.random.Generator, samples_per_target: int = 3, n_length_bins: int = 10, exclude_target_from_pool: bool = True, **_: Any) -> BackgroundSpec:
    annot = _align_pool(pool_headers, pool_annotations)
    if not stratify_by and not match_length: raise ValueError("Requires stratify_by or match_length.")
    target_keys = target_annotations[stratify_by].astype(str).agg("|".join, axis=1) if stratify_by else pd.Series([""] * len(target_headers), dtype=str)
    pool_keys = annot[stratify_by].astype(str).agg("|".join, axis=1) if stratify_by else pd.Series([""] * len(pool_headers), dtype=str)
    
    if match_length:
        quantiles = np.linspace(0.0, 1.0, n_length_bins + 1)
        bin_edges = np.quantile(target_lengths, quantiles)
        for i in range(1, len(bin_edges)):
            if bin_edges[i] <= bin_edges[i - 1]: bin_edges[i] = bin_edges[i - 1] + 1e-9
        bin_edges[0] -= 1e-9; bin_edges[-1] += 1e-9
        target_keys = target_keys + "|L" + pd.Series(np.digitize(target_lengths, bin_edges[1:-1]).astype(str))
        pool_keys = pool_keys + "|L" + pd.Series(np.digitize(pool_lengths, bin_edges[1:-1]).astype(str))
        
    if exclude_target_from_pool:
        target_set = set(target_headers)
        keep = np.array([i for i, h in enumerate(pool_headers) if h not in target_set], dtype=int)
        pool_embeddings = pool_embeddings[keep]
        pool_keys = pool_keys.iloc[keep].reset_index(drop=True)
        
    chosen = []
    for stratum, t_count in target_keys.value_counts().items():
        requested = int(t_count * samples_per_target)
        if requested == 0: continue
        pool_idx = np.where(pool_keys.to_numpy() == stratum)[0]
        n_take = min(requested, pool_idx.size)
        chosen.extend(rng.choice(pool_idx, size=n_take, replace=False).tolist())
        
    X_B = pool_embeddings[np.array(sorted(set(chosen)), dtype=int)]
    return BackgroundSpec(X_background=X_B, X_target=None, source=MIXED, n_target=len(target_headers), n_background=int(X_B.shape[0]), details={})

def strategy_isolate(
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
    annot = _align_pool(pool_headers, pool_annotations)
    for col in stratify_by:
        if col not in target_annotations.columns: raise ValueError(f"Artifact {col!r} not found in target.")
        if col not in annot.columns: raise ValueError(f"Artifact {col!r} not found in pool.")
        if len(target_annotations[col].dropna().unique()) < 2:
            raise ValueError(f"Artifact {col!r} has no variance in target.")

    if exclude_target_from_pool:
        target_set = set(target_headers)
        keep = np.array([i for i, h in enumerate(pool_headers) if h not in target_set], dtype=int)
        pool_embeddings = pool_embeddings[keep]
        annot = annot.iloc[keep].reset_index(drop=True)

    pool_strata = annot[stratify_by].astype(str).agg("|".join, axis=1)
    unique_strata = pool_strata.unique()
    target_total = len(target_headers) * samples_per_target
    per_stratum_target = max(2, target_total // len(unique_strata))

    stratum_counts = pool_strata.value_counts()
    n_take_per_stratum = min(per_stratum_target, stratum_counts.min())

    if n_take_per_stratum < 2:
        raise ValueError("Pool lacks enough samples to isolate.")

    chosen_global = []
    for stratum in unique_strata:
        pool_idx = np.where(pool_strata.to_numpy() == stratum)[0]
        picked = rng.choice(pool_idx, size=n_take_per_stratum, replace=False)
        chosen_global.extend(picked.tolist())

    X_B = pool_embeddings[np.array(sorted(set(chosen_global)), dtype=int)]
    return BackgroundSpec(X_B, None, ISOLATE, len(target_headers), int(X_B.shape[0]), {})

_STRATEGIES = {
    POOL: strategy_pool,
    COMPLEMENT: strategy_complement,
    LENGTH_MATCHED: strategy_length_matched,
    STRATIFIED: strategy_stratified,
    MIXED: strategy_mixed,
    ISOLATE: strategy_isolate,
}

def build_background(name: str, **kwargs) -> BackgroundSpec:
    if name not in _STRATEGIES:
        raise ValueError(f"Unknown background strategy {name!r}.")
    return _STRATEGIES[name](**kwargs)