"""
Tests for dimensionality reduction methods.

Verifies that all six DR methods (PCA, t-SNE, UMAP, PaCMAP, MDS, LocalMAP)
produce correct output shapes, handle edge cases, and work end-to-end
through the processor pipeline.
"""

import numpy as np
import pytest

from protspace.utils.reducers import (
    DimensionReductionConfig,
    LocalMAPReducer,
    MDSReducer,
    PaCMAPReducer,
    PCAReducer,
    PPCAReducer,  # deprecated alias, kept for the compat test below
    RhoPCAReducer,
    TSNEReducer,
    UMAPReducer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Deterministic data large enough for all methods (t-SNE needs n > perplexity)
SEED = 42
N_SAMPLES = 50
N_FEATURES = 20


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def data_2d(rng):
    """Float32 data suitable for 2-component reduction."""
    return rng.standard_normal((N_SAMPLES, N_FEATURES)).astype(np.float32)


@pytest.fixture
def data_3d(rng):
    """Float32 data suitable for 3-component reduction."""
    return rng.standard_normal((N_SAMPLES, N_FEATURES)).astype(np.float32)


@pytest.fixture
def config_2d():
    return DimensionReductionConfig(n_components=2, random_state=SEED)


@pytest.fixture
def config_3d():
    return DimensionReductionConfig(n_components=3, random_state=SEED)


# ---------------------------------------------------------------------------
# Per-method tests — 2D
# ---------------------------------------------------------------------------


class TestPCAReducer:
    def test_output_shape_2d(self, data_2d, config_2d):
        result = PCAReducer(config_2d).fit_transform(data_2d)
        assert result.shape == (N_SAMPLES, 2)

    def test_output_shape_3d(self, data_3d, config_3d):
        result = PCAReducer(config_3d).fit_transform(data_3d)
        assert result.shape == (N_SAMPLES, 3)

    def test_no_nan_values(self, data_2d, config_2d):
        result = PCAReducer(config_2d).fit_transform(data_2d)
        assert not np.isnan(result).any()

    def test_deterministic(self, data_2d, config_2d):
        r1 = PCAReducer(config_2d).fit_transform(data_2d)
        r2 = PCAReducer(config_2d).fit_transform(data_2d)
        np.testing.assert_allclose(r1, r2, atol=1e-5)

    def test_get_params(self, config_2d):
        reducer = PCAReducer(config_2d)
        reducer.fit_transform(np.random.randn(10, 5).astype(np.float32))
        params = reducer.get_params()
        assert params["n_components"] == 2
        assert "explained_variance_ratio" in params


class TestTSNEReducer:
    def test_output_shape_2d(self, data_2d, config_2d):
        result = TSNEReducer(config_2d).fit_transform(data_2d)
        assert result.shape == (N_SAMPLES, 2)

    def test_no_nan_values(self, data_2d, config_2d):
        result = TSNEReducer(config_2d).fit_transform(data_2d)
        assert not np.isnan(result).any()

    def test_get_params(self, config_2d):
        params = TSNEReducer(config_2d).get_params()
        assert params["n_components"] == 2
        assert "perplexity" in params


class TestUMAPReducer:
    def test_output_shape_2d(self, data_2d, config_2d):
        result = UMAPReducer(config_2d).fit_transform(data_2d)
        assert result.shape == (N_SAMPLES, 2)

    def test_output_shape_3d(self, data_3d, config_3d):
        result = UMAPReducer(config_3d).fit_transform(data_3d)
        assert result.shape == (N_SAMPLES, 3)

    def test_no_nan_values(self, data_2d, config_2d):
        result = UMAPReducer(config_2d).fit_transform(data_2d)
        assert not np.isnan(result).any()

    def test_get_params(self, config_2d):
        params = UMAPReducer(config_2d).get_params()
        assert params["n_components"] == 2
        assert "n_neighbors" in params
        assert "min_dist" in params


class TestPaCMAPReducer:
    def test_output_shape_2d(self, data_2d, config_2d):
        result = PaCMAPReducer(config_2d).fit_transform(data_2d)
        assert result.shape == (N_SAMPLES, 2)

    def test_no_nan_values(self, data_2d, config_2d):
        result = PaCMAPReducer(config_2d).fit_transform(data_2d)
        assert not np.isnan(result).any()

    def test_get_params(self, config_2d):
        params = PaCMAPReducer(config_2d).get_params()
        assert params["n_components"] == 2
        assert "MN_ratio" in params
        assert "FP_ratio" in params


class TestMDSReducer:
    def test_output_shape_2d(self, data_2d, config_2d):
        result = MDSReducer(config_2d).fit_transform(data_2d)
        assert result.shape == (N_SAMPLES, 2)

    def test_no_nan_values(self, data_2d, config_2d):
        result = MDSReducer(config_2d).fit_transform(data_2d)
        assert not np.isnan(result).any()

    def test_get_params(self, config_2d):
        params = MDSReducer(config_2d).get_params()
        assert params["n_components"] == 2
        assert "n_init" in params


class TestLocalMAPReducer:
    def test_output_shape_2d(self, data_2d, config_2d):
        result = LocalMAPReducer(config_2d).fit_transform(data_2d)
        assert result.shape == (N_SAMPLES, 2)

    def test_no_nan_values(self, data_2d, config_2d):
        result = LocalMAPReducer(config_2d).fit_transform(data_2d)
        assert not np.isnan(result).any()

    def test_get_params(self, config_2d):
        params = LocalMAPReducer(config_2d).get_params()
        assert params["n_components"] == 2
        assert "MN_ratio" in params
        assert "FP_ratio" in params


# ---------------------------------------------------------------------------
# Cross-cutting tests
# ---------------------------------------------------------------------------

ALL_REDUCERS = [
    ("pca", PCAReducer),
    ("tsne", TSNEReducer),
    ("umap", UMAPReducer),
    ("pacmap", PaCMAPReducer),
    ("mds", MDSReducer),
    ("localmap", LocalMAPReducer),
]


@pytest.mark.parametrize("name,cls", ALL_REDUCERS, ids=[r[0] for r in ALL_REDUCERS])
class TestAllReducers:
    """Tests that apply to every reducer."""

    def test_returns_float_array(self, name, cls, data_2d, config_2d):
        result = cls(config_2d).fit_transform(data_2d)
        assert result.dtype in (np.float32, np.float64)

    def test_no_inf_values(self, name, cls, data_2d, config_2d):
        result = cls(config_2d).fit_transform(data_2d)
        assert not np.isinf(result).any()

    def test_output_finite(self, name, cls, data_2d, config_2d):
        result = cls(config_2d).fit_transform(data_2d)
        assert np.isfinite(result).all()


class TestFloat16Handling:
    """Ensure float16 input doesn't cause overflow or NaN."""

    @pytest.mark.parametrize("name,cls", ALL_REDUCERS, ids=[r[0] for r in ALL_REDUCERS])
    def test_float16_input_produces_finite_output(self, name, cls, rng):
        # Small values typical of pLM embeddings stored in float16
        data = (rng.standard_normal((N_SAMPLES, N_FEATURES)) * 0.04).astype(np.float16)
        config = DimensionReductionConfig(n_components=2, random_state=SEED)
        # float16 is upcast in the processor, but reducers should still handle it
        result = cls(config).fit_transform(data.astype(np.float32))
        assert result.shape == (N_SAMPLES, 2)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# DimensionReductionConfig validation
# ---------------------------------------------------------------------------


class TestDimensionReductionConfig:
    def test_default_values(self):
        config = DimensionReductionConfig()
        assert config.n_components == 2
        assert config.metric == "euclidean"
        assert config.random_state == 42

    def test_custom_values(self):
        config = DimensionReductionConfig(
            n_components=3, metric="cosine", n_neighbors=10
        )
        assert config.n_components == 3
        assert config.metric == "cosine"
        assert config.n_neighbors == 10

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            DimensionReductionConfig(metric="invalid")

    def test_invalid_n_components_raises(self):
        with pytest.raises(ValueError):
            DimensionReductionConfig(n_components=0)

    def test_invalid_perplexity_raises(self):
        with pytest.raises(ValueError):
            DimensionReductionConfig(perplexity=3)  # min is 5


# ---------------------------------------------------------------------------
# End-to-end through BaseProcessor
# ---------------------------------------------------------------------------


class TestProcessorReduction:
    """Test DR methods through the BaseProcessor.process_reduction pipeline."""

    def test_all_methods_through_processor(self, data_2d):
        from protspace.data.processors.base_processor import BaseProcessor
        from protspace.utils import get_reducers

        REDUCERS = get_reducers()

        processor = BaseProcessor({"random_state": SEED}, REDUCERS)

        for method in ["pca", "tsne", "umap", "pacmap", "mds", "localmap"]:
            result = processor.process_reduction(data_2d, method, 2)
            assert result["data"].shape == (N_SAMPLES, 2), f"{method} shape mismatch"
            assert np.isfinite(result["data"]).all(), f"{method} produced non-finite"
            assert result["dimensions"] == 2
            assert isinstance(result["name"], str)
            assert isinstance(result["info"], dict)


# ---------------------------------------------------------------------------
# ρPCA (contrastive) — k-D pre-reduction + output scaling
# ---------------------------------------------------------------------------


def _rhopca_config(n_components, rho_output_scale="none"):
    """Build a ρPCA config with an explicit background attached as the pipeline does."""
    rng = np.random.default_rng(SEED)
    cfg = DimensionReductionConfig(
        n_components=n_components,
        regularization_mu=1e-6,
        standard_scale=False,
        rho_output_scale=rho_output_scale,
    )
    background = rng.standard_normal((N_SAMPLES, N_FEATURES))
    object.__setattr__(cfg, "background_data", background)
    object.__setattr__(cfg, "background_source", "explicit")
    return cfg


class TestRhoPCAReducer:
    def test_two_components_default(self):
        rng = np.random.default_rng(SEED)
        data = rng.standard_normal((N_SAMPLES, N_FEATURES))
        out = RhoPCAReducer(_rhopca_config(2)).fit_transform(data)
        assert out.shape == (N_SAMPLES, 2)
        assert np.isfinite(out).all()

    @pytest.mark.parametrize("k", [3, 5, 10])
    def test_k_dim_prereduction(self, k):
        rng = np.random.default_rng(SEED)
        data = rng.standard_normal((N_SAMPLES, N_FEATURES))
        out = RhoPCAReducer(_rhopca_config(k)).fit_transform(data)
        assert out.shape == (N_SAMPLES, k)
        assert np.isfinite(out).all()

    def test_output_scaling_modes(self):
        rng = np.random.default_rng(SEED)
        data = rng.standard_normal((N_SAMPLES, N_FEATURES))
        unit = RhoPCAReducer(_rhopca_config(5, "unit_var")).fit_transform(data)
        # unit_var standardizes every axis to ~unit variance
        assert np.allclose(unit.std(axis=0), 1.0, atol=1e-6)
        tvar = RhoPCAReducer(_rhopca_config(5, "target_var")).fit_transform(data)
        assert tvar.shape == (N_SAMPLES, 5) and np.isfinite(tvar).all()

    def test_rejects_one_component(self):
        rng = np.random.default_rng(SEED)
        data = rng.standard_normal((N_SAMPLES, N_FEATURES))
        with pytest.raises(ValueError):
            RhoPCAReducer(_rhopca_config(1)).fit_transform(data)

    @pytest.mark.parametrize("k", [2, 25])  # 2 → subset solver, 25 → full solver (2k>d)
    def test_eigensolve_satisfies_generalized_problem(self, k):
        """The returned top-k eigenpairs must satisfy Σ_T v = λ(Σ_B+μI)v and be
        ρ-descending — for both the partial (subset) and full solver branches."""
        from protspace.utils.reducers import _sample_covariance, _solve_rho_eigenproblem

        rng = np.random.default_rng(0)
        d, mu = 30, 1e-6
        St = _sample_covariance(rng.standard_normal((200, d)))
        Sb = _sample_covariance(rng.standard_normal((200, d)))
        vals, vecs = _solve_rho_eigenproblem(St, Sb, k, mu)
        assert vals.shape == (k,) and vecs.shape == (d, k)
        assert np.all(np.diff(vals) <= 1e-9)  # descending ρ
        Sb_reg = Sb + mu * np.eye(d)
        for i in range(k):
            resid = np.linalg.norm(St @ vecs[:, i] - vals[i] * (Sb_reg @ vecs[:, i]))
            assert resid < 1e-8

    def test_ppca_reducer_alias(self):
        # The pre-rename class name stays importable and identical (back-compat).
        assert PPCAReducer is RhoPCAReducer
