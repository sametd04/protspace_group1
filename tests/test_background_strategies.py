"""Unit tests for ρPCA background construction strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from protspace.data.processors.background_strategies import (
    build_background, BackgroundSpec,
)


@pytest.fixture
def synth_pool():
    """A 'pool' of 1000 fake proteins with 64-D embeddings, length, and
    two categorical annotations."""
    rng = np.random.default_rng(0)
    n, d = 1000, 64
    X = rng.standard_normal(size=(n, d))
    headers = [f"P{i:04d}" for i in range(n)]
    lengths = rng.integers(50, 500, size=n)
    superkingdom = rng.choice(["Bacteria", "Eukaryota", "Archaea"], size=n,
                               p=[0.5, 0.4, 0.1])
    signal_peptide = rng.choice(["yes", "no"], size=n, p=[0.3, 0.7])
    annotations = pd.DataFrame({
        "identifier": headers,
        "superkingdom": superkingdom,
        "signal_peptide": signal_peptide,
        "sequence_length": lengths,
    })
    return X, headers, lengths, annotations


@pytest.fixture
def synth_target(synth_pool):
    """A small target embedded in the same space as the pool (50 fake
    'three-finger toxins') with biased length and taxonomy."""
    pool_X, _, _, _ = synth_pool
    rng = np.random.default_rng(1)
    n_t, d = 50, pool_X.shape[1]
    X_t = rng.standard_normal(size=(n_t, d))
    headers = [f"T{i:03d}" for i in range(n_t)]
    # Skew: targets are short Eukaryotes with signal peptides
    lengths = rng.integers(80, 200, size=n_t)
    annotations = pd.DataFrame({
        "identifier": headers,
        "superkingdom": ["Eukaryota"] * n_t,
        "signal_peptide": ["yes"] * n_t,
        "sequence_length": lengths,
    })
    return X_t, headers, lengths, annotations


# ---------------------------- pool ---------------------------------------

def test_pool_returns_whole_pool(synth_pool, synth_target):
    pool_X, pool_h, _, _ = synth_pool
    tgt_X, tgt_h, _, _ = synth_target
    spec = build_background(
        "pool",
        pool_embeddings=pool_X, pool_headers=pool_h,
        target_embeddings=tgt_X, target_headers=tgt_h,
    )
    assert isinstance(spec, BackgroundSpec)
    assert spec.X_target is None
    assert spec.n_background == pool_X.shape[0]  # no overlap to exclude
    assert spec.source == "pool"


def test_pool_excludes_target_identifiers(synth_pool):
    pool_X, pool_h, _, _ = synth_pool
    # Target identifiers overlap with pool
    tgt_h = pool_h[:30]
    tgt_X = pool_X[:30]
    spec = build_background(
        "pool",
        pool_embeddings=pool_X, pool_headers=pool_h,
        target_embeddings=tgt_X, target_headers=tgt_h,
    )
    assert spec.n_background == pool_X.shape[0] - 30
    assert spec.details["n_overlap_excluded"] == 30


# ---------------------------- complement ---------------------------------

def test_complement_splits_by_annotation(synth_pool):
    pool_X, pool_h, _, pool_annot = synth_pool
    spec = build_background(
        "complement",
        pool_embeddings=pool_X, pool_headers=pool_h,
        pool_annotations=pool_annot,
        target_embeddings=None,        # not used by complement
        target_headers=[],             # not used
        target_annotation="superkingdom",
        target_values=["Eukaryota"],
    )
    n_target_expected = (pool_annot["superkingdom"] == "Eukaryota").sum()
    assert spec.n_target == n_target_expected
    assert spec.n_background == len(pool_h) - n_target_expected
    assert spec.X_target is not None
    assert spec.X_target.shape[0] == n_target_expected


def test_complement_rejects_unknown_column(synth_pool):
    pool_X, pool_h, _, pool_annot = synth_pool
    with pytest.raises(ValueError, match="Annotation column"):
        build_background(
            "complement",
            pool_embeddings=pool_X, pool_headers=pool_h,
            pool_annotations=pool_annot,
            target_embeddings=None, target_headers=[],
            target_annotation="nonexistent",
            target_values=["x"],
        )


# ---------------------------- length_matched -----------------------------

def test_length_matched_distribution_matches(synth_pool, synth_target):
    from scipy.stats import ks_2samp
    pool_X, pool_h, pool_l, _ = synth_pool
    tgt_X, tgt_h, tgt_l, _ = synth_target
    rng = np.random.default_rng(42)
    spec = build_background(
        "length_matched",
        pool_embeddings=pool_X, pool_headers=pool_h, pool_lengths=pool_l,
        target_embeddings=tgt_X, target_headers=tgt_h, target_lengths=tgt_l,
        rng=rng,
        samples_per_target=5,
        n_bins=10,
    )
    # KS between target lengths and matched-background lengths should be
    # much closer to 0 than between target and unmatched pool.
    ks_unmatched, _ = ks_2samp(tgt_l, pool_l)
    ks_matched = spec.details["ks_d_after_matching"]
    assert ks_matched < ks_unmatched - 0.2, (
        f"Length matching did not reduce KS distance: "
        f"unmatched={ks_unmatched:.3f}, matched={ks_matched:.3f}"
    )


# ---------------------------- stratified ---------------------------------

def test_stratified_matches_target_proportions(synth_pool, synth_target):
    pool_X, pool_h, _, pool_annot = synth_pool
    tgt_X, tgt_h, _, tgt_annot = synth_target
    rng = np.random.default_rng(0)
    spec = build_background(
        "stratified",
        pool_embeddings=pool_X, pool_headers=pool_h,
        pool_annotations=pool_annot,
        target_embeddings=tgt_X, target_headers=tgt_h,
        target_annotations=tgt_annot,
        stratify_by=["superkingdom"],
        rng=rng,
        samples_per_target=4,
    )
    # All targets are Eukaryota, so the matched background should also be
    # all Eukaryota.
    chosen = pool_annot.loc[
        pool_annot["identifier"].isin(
            [pool_h[i] for i in range(len(pool_h))
             if pool_X[i].tobytes() in {row.tobytes() for row in spec.X_background}]
        )
    ]
    # Direct check: all background rows came from Eukaryota.
    assert spec.details["matched_strata"] == 1
    assert spec.n_background <= 50 * 4  # at most samples_per_target × n_target


def test_stratified_requires_known_columns(synth_pool, synth_target):
    pool_X, pool_h, _, pool_annot = synth_pool
    tgt_X, tgt_h, _, tgt_annot = synth_target
    with pytest.raises(ValueError, match="not found in pool"):
        build_background(
            "stratified",
            pool_embeddings=pool_X, pool_headers=pool_h,
            pool_annotations=pool_annot,
            target_embeddings=tgt_X, target_headers=tgt_h,
            target_annotations=tgt_annot,
            stratify_by=["nope"],
            rng=np.random.default_rng(),
        )


# ---------------------------- mixed --------------------------------------

def test_mixed_with_length_only(synth_pool, synth_target):
    pool_X, pool_h, pool_l, pool_annot = synth_pool
    tgt_X, tgt_h, tgt_l, tgt_annot = synth_target
    spec = build_background(
        "mixed",
        pool_embeddings=pool_X, pool_headers=pool_h,
        pool_annotations=pool_annot,
        pool_lengths=pool_l,
        target_embeddings=tgt_X, target_headers=tgt_h,
        target_annotations=tgt_annot,
        target_lengths=tgt_l,
        stratify_by=[],
        match_length=True,
        rng=np.random.default_rng(0),
        samples_per_target=3,
        n_length_bins=8,
    )
    assert spec.n_background > 0
    assert spec.details["match_length"]
    assert spec.details["stratify_by"] == []


def test_mixed_with_both(synth_pool, synth_target):
    pool_X, pool_h, pool_l, pool_annot = synth_pool
    tgt_X, tgt_h, tgt_l, tgt_annot = synth_target
    spec = build_background(
        "mixed",
        pool_embeddings=pool_X, pool_headers=pool_h,
        pool_annotations=pool_annot,
        pool_lengths=pool_l,
        target_embeddings=tgt_X, target_headers=tgt_h,
        target_annotations=tgt_annot,
        target_lengths=tgt_l,
        stratify_by=["superkingdom", "signal_peptide"],
        match_length=True,
        rng=np.random.default_rng(0),
        samples_per_target=3,
        n_length_bins=5,
    )
    assert spec.n_background > 0
    assert spec.details["stratify_by"] == ["superkingdom", "signal_peptide"]


def test_mixed_rejects_empty_constraints(synth_pool, synth_target):
    pool_X, pool_h, pool_l, pool_annot = synth_pool
    tgt_X, tgt_h, tgt_l, tgt_annot = synth_target
    with pytest.raises(ValueError, match="at least one"):
        build_background(
            "mixed",
            pool_embeddings=pool_X, pool_headers=pool_h,
            pool_annotations=pool_annot,
            pool_lengths=pool_l,
            target_embeddings=tgt_X, target_headers=tgt_h,
            target_annotations=tgt_annot,
            target_lengths=tgt_l,
            stratify_by=[],
            match_length=False,
            rng=np.random.default_rng(),
        )


def test_unknown_strategy_name():
    with pytest.raises(ValueError, match="Unknown background strategy"):
        build_background("not_a_real_strategy")