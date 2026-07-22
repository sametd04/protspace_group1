"""Tests for the ρPCA annotation-nuisance background: order-invariant cross-fit
folds, the scale=1.0 / cross_fit=1 defaults, auto type detection, minimal-spec
robustness, and the log1p→identity fallback for negative continuous values."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from protspace.utils.annotation_nuisance_background import (
    NuisanceSpec,
    _default_missing_policy,
    build_annotation_nuisance_background,
    build_continuous_design,
    estimate_effect_block,
    infer_annotation_type,
    parse_nuisance_spec,
    ridge_predict_effect,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _spec(raw, global_ridge_alpha="auto"):
    return parse_nuisance_spec(
        raw, global_cross_fit=1, global_ridge_alpha=global_ridge_alpha, random_state=42,
    )


# --------------------------------------------------------------------------- #
# Order-invariant cross-fit folds
# --------------------------------------------------------------------------- #
def test_ridge_predict_effect_order_invariant():
    """Out-of-fold predictions must not depend on the input row order when the
    identifiers are supplied (folds are assigned by canonical id order)."""
    rng = _rng(1)
    n, p, d = 12, 2, 3
    Phi = rng.normal(size=(n, p))
    Y = rng.normal(size=(n, d))
    ids = [f"P{i:03d}" for i in range(n)]

    base, _, _ = ridge_predict_effect(
        Phi, Y, alpha=10.0, cross_fit=3, random_state=42,
        fit_intercept=True, fold_ids=ids,
    )

    perm = rng.permutation(n)
    permd, _, _ = ridge_predict_effect(
        Phi[perm], Y[perm], alpha=10.0, cross_fit=3, random_state=42,
        fit_intercept=True, fold_ids=[ids[i] for i in perm],
    )

    # permd is aligned to the permuted input; base[perm] is the same rows in that order
    assert np.allclose(permd, base[perm], atol=1e-9)


def test_ridge_predict_effect_positional_folds_are_order_sensitive():
    """Sanity: without fold_ids the folds are positional, so a permutation changes
    the out-of-fold predictions — this is exactly what the id-based fix removes."""
    rng = _rng(2)
    n = 12
    Phi = rng.normal(size=(n, 2))
    Y = rng.normal(size=(n, 3))

    base, _, _ = ridge_predict_effect(
        Phi, Y, alpha=10.0, cross_fit=3, random_state=42, fit_intercept=True,
    )
    perm = rng.permutation(n)
    permd, _, _ = ridge_predict_effect(
        Phi[perm], Y[perm], alpha=10.0, cross_fit=3, random_state=42, fit_intercept=True,
    )
    assert not np.allclose(permd, base[perm], atol=1e-6)


def test_estimate_effect_block_order_invariant():
    """The full effect block (per-identifier) must be invariant to row order."""
    rng = _rng(3)
    n, d = 16, 5
    X = rng.normal(size=(n, d))
    grp = np.where(np.arange(n) % 2 == 0, "yes", "no")
    ids = [f"Q{i:03d}" for i in range(n)]
    spec = parse_nuisance_spec(
        "grp:type=binary;missing=zero;cross_fit=4",
        global_cross_fit=1, global_ridge_alpha=10.0, random_state=42,
    )
    ann = pd.DataFrame({"grp": grp})

    block = estimate_effect_block(ids, X, ann, spec)
    by_id = dict(zip(block.ids, block.effects, strict=True))

    perm = rng.permutation(n)
    block_p = estimate_effect_block(
        [ids[i] for i in perm], X[perm], ann.iloc[perm].reset_index(drop=True), spec,
    )
    by_id_p = dict(zip(block_p.ids, block_p.effects, strict=True))

    assert set(by_id) == set(by_id_p)
    for k in by_id:
        assert np.allclose(by_id[k], by_id_p[k], atol=1e-9)


# --------------------------------------------------------------------------- #
# Defaults: scale=1.0, cross_fit=1
# --------------------------------------------------------------------------- #
def test_default_scale_is_one():
    assert NuisanceSpec(raw="x", name="x", column="x").scale == 1.0
    spec = parse_nuisance_spec(
        "x:type=binary", global_cross_fit=1, global_ridge_alpha=10.0, random_state=42,
    )
    assert spec.scale == 1.0


def test_explicit_scale_overrides_default():
    spec = parse_nuisance_spec(
        "x:type=binary;scale=0.5", global_cross_fit=1, global_ridge_alpha=10.0,
        random_state=42,
    )
    assert spec.scale == 0.5


def test_default_cross_fit_is_one():
    from protspace.cli.prepare import prepare as prepare_cmd
    from protspace.data.processors.pipeline import ReducerParams

    assert ReducerParams().nuisance_cross_fit == 1
    sig = inspect.signature(prepare_cmd)
    assert sig.parameters["nuisance_cross_fit"].default == 1


# --------------------------------------------------------------------------- #
# Auto type detection + minimal / bare specs
# --------------------------------------------------------------------------- #
def test_auto_type_detection():
    assert infer_annotation_type(pd.Series(["yes", "no", "yes", "no", "yes"]))[0] == "binary"
    assert infer_annotation_type(pd.Series(["A", "B", "C", "A", "B", "C", "A"]))[0] == "categorical"
    assert infer_annotation_type(pd.Series(list(range(1, 20))))[0] == "continuous"
    assert infer_annotation_type(pd.Series(["a;b", "b;c", "a", "c;a", "b"]))[0] == "multilabel"


def test_bare_and_minimal_specs_parse():
    # No colon at all -> column only, type auto, new defaults.
    s = _spec("grp")
    assert s.column == "grp"
    assert s.nuisance_type == "auto"
    assert s.scale == 1.0
    assert s.missing == "auto"
    # Type given, everything else default.
    assert _spec("sp_in_embedding:type=binary").nuisance_type == "binary"


def test_default_missing_policy():
    assert _default_missing_policy("binary", "auto") == "category"
    assert _default_missing_policy("categorical", "auto") == "category"
    assert _default_missing_policy("continuous", "auto") == "drop"
    assert _default_missing_policy("multilabel", "auto") == "empty"
    # Explicit overrides auto.
    assert _default_missing_policy("binary", "zero") == "zero"


@pytest.mark.parametrize(
    "raw,expected_type",
    [
        ("sp", "binary"),
        ("sp:type=binary", "binary"),
        ("grp:type=categorical", "categorical"),
        ("length:type=continuous", "continuous"),
        ("grp", "categorical"),
    ],
)
def test_minimal_spec_end_to_end(raw, expected_type):
    """Only `--nuisance SPEC` (nothing else) must build a valid background."""
    rng = _rng(7)
    n, d = 60, 12
    X = rng.normal(size=(n, d))
    ids = [f"P{i:03d}" for i in range(n)]
    ann = pd.DataFrame({
        "identifier": ids,
        "sp": np.where(np.arange(n) % 2 == 0, "yes", "no"),
        "grp": rng.choice(["A", "B", "C"], size=n),
        "length": rng.integers(10, 300, size=n),
    })
    res = build_annotation_nuisance_background(
        ids=ids, X=X, annotations=ann, nuisance_specs=[raw], target_name="prot_t5",
    )
    B = res.background
    assert B.shape == (2 * n, d)          # signed [+E, -E] block
    assert np.isfinite(B).all()
    det = res.block_details[0]["estimator_details"]
    assert det["resolved_type"] == expected_type


# --------------------------------------------------------------------------- #
# log1p → identity fallback for negative continuous values
# --------------------------------------------------------------------------- #
def test_log1p_negative_fallback_keeps_rows():
    # Explicit transform=log1p isolates the negative→identity fallback.
    neg = pd.Series([-5.0, -2.0, 0.0, 3.0, 8.0, 12.0, -1.0, 20.0, -9.0, 4.0, 7.0, 15.0])
    d = build_continuous_design(neg, _spec("charge:type=continuous;transform=log1p"), name="charge")
    assert int(d.row_mask.sum()) == len(neg)         # no rows dropped for negativity
    assert d.details["transform"] == "none"          # effective
    assert d.details["transform_requested"] == "log1p"
    assert any("negative" in w for w in d.warnings)


def test_log1p_kept_for_positive_data():
    pos = pd.Series([1.0, 5.0, 8.0, 12.0, 33.0, 55.0, 88.0, 120.0, 7.0, 64.0, 29.0, 101.0])
    d = build_continuous_design(pos, _spec("length:type=continuous;transform=log1p"), name="length")
    assert int(d.row_mask.sum()) == len(pos)
    assert d.details["transform"] == "log1p"
    assert not any("negative" in w for w in d.warnings)


# --------------------------------------------------------------------------- #
# Auto-tuned ridge_alpha (Phase 1)
# --------------------------------------------------------------------------- #
def test_ridge_alpha_default_is_auto():
    assert _spec("x:type=binary").ridge_alpha == "auto"
    assert _spec("x:type=binary;ridge_alpha=0.5").ridge_alpha == 0.5
    from protspace.cli.prepare import prepare as prepare_cmd
    from protspace.data.processors.pipeline import ReducerParams
    assert ReducerParams().nuisance_ridge_alpha is None
    assert inspect.signature(prepare_cmd).parameters["nuisance_ridge_alpha"].default is None


def _small_background(spec, seed=5):
    rng = _rng(seed)
    n, d = 90, 16
    X = rng.normal(size=(n, d))
    ids = [f"P{i:03d}" for i in range(n)]
    ann = pd.DataFrame({"identifier": ids, "grp": rng.choice(list("ABCDE"), size=n),
                        "length": rng.integers(1, 300, size=n).astype(float)})
    res = build_annotation_nuisance_background(
        ids=ids, X=X, annotations=ann, nuisance_specs=[spec], target_name="prot_t5",
    )
    return res.block_details[0]["estimator_details"]["fit_details"]


def test_ridge_alpha_autotune_selects_from_grid():
    from protspace.utils.annotation_nuisance_background import RIDGE_ALPHA_GRID
    fit = _small_background("grp:type=categorical")
    assert fit["ridge_alpha_source"].startswith("auto")
    assert fit["ridge_alpha"] in set(np.round(RIDGE_ALPHA_GRID, 10))


def test_explicit_ridge_alpha_overrides():
    fit = _small_background("grp:type=categorical;ridge_alpha=123")
    assert fit["ridge_alpha_source"] == "explicit"
    assert fit["ridge_alpha"] == 123.0


def test_autotune_deterministic():
    a = _small_background("grp:type=categorical")["ridge_alpha"]
    b = _small_background("grp:type=categorical")["ridge_alpha"]
    assert a == b


def test_autotune_effect_block_order_invariant():
    rng = _rng(11)
    n, d = 40, 6
    X = rng.normal(size=(n, d))
    grp = np.where(np.arange(n) % 2 == 0, "yes", "no")
    ids = [f"Q{i:03d}" for i in range(n)]
    spec = _spec("grp:type=binary;cross_fit=4")   # ridge_alpha defaults to auto
    ann = pd.DataFrame({"grp": grp})
    b0 = estimate_effect_block(ids, X, ann, spec)
    by0 = dict(zip(b0.ids, b0.effects, strict=True))
    perm = rng.permutation(n)
    b1 = estimate_effect_block([ids[i] for i in perm], X[perm],
                               ann.iloc[perm].reset_index(drop=True), spec)
    by1 = dict(zip(b1.ids, b1.effects, strict=True))
    for k in by0:
        assert np.allclose(by0[k], by1[k], atol=1e-9)


# --------------------------------------------------------------------------- #
# Continuous transform / n_knots auto-selection (Phase 2)
# --------------------------------------------------------------------------- #
def test_continuous_hparam_autoselect_records_choice():
    fit = _small_background("length:type=continuous")   # transform/n_knots auto
    # runs end-to-end and reports an auto-tuned alpha
    assert fit["ridge_alpha_source"].startswith("auto")


def test_continuous_explicit_disables_search():
    d = build_continuous_design(
        pd.Series(np.linspace(1, 100, 40)),
        _spec("length:type=continuous;transform=none;n_knots=6"), name="length",
    )
    assert d.details["transform"] == "none"
    assert d.details["n_knots"] == 6
    assert "hparam_selection" not in d.details


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
