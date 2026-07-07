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
    TSNEReducer,
    UMAPReducer,
)

try:
    from protspace.utils.reducers import PPCAReducer

    HAS_PPCA = True
except ImportError:
    HAS_PPCA = False

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Deterministic data large enough for all methods (t-SNE needs n > perplexity)
SEED = 42
N_SAMPLES = 50
N_FEATURES = 20

# ρPCA-specific note: background data is attached externally to the frozen
# config via object.__setattr__. With a 15-sample background and 20 features,
# Σ_B is rank-deficient, so all ρPCA tests pass regularization_mu to
# stabilize the generalized eigenproblem.
PPCA_TEST_MU = 1e-3


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


@pytest.fixture
def config_2d_ppca(rng):
    """2D config with regularization and background_data attached."""
    cfg = DimensionReductionConfig(
        n_components=2,
        random_state=SEED,
        regularization_mu=PPCA_TEST_MU,
    )
    bg = rng.standard_normal((15, N_FEATURES)).astype(np.float32)
    object.__setattr__(cfg, "background_data", bg)
    return cfg


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
# ρPCA tests
# ---------------------------------------------------------------------------

ppca_skip = pytest.mark.skipif(not HAS_PPCA, reason="PPCAReducer not yet implemented")


def _make_ppca_config(rng, n_components=2, n_bg=15, regularization_mu=PPCA_TEST_MU):
    """Helper: create a DimensionReductionConfig with background_data attached."""
    cfg = DimensionReductionConfig(
        n_components=n_components,
        random_state=SEED,
        regularization_mu=regularization_mu,
    )
    bg = rng.standard_normal((n_bg, N_FEATURES)).astype(np.float32)
    object.__setattr__(cfg, "background_data", bg)
    return cfg


@ppca_skip
class TestPPCAReducer:
    @pytest.fixture
    def ppca_config_2d(self, rng):
        return _make_ppca_config(rng, n_components=2)

    @pytest.fixture
    def ppca_config_3d(self, rng):
        return _make_ppca_config(rng, n_components=3)

    def test_output_shape_2d(self, data_2d, ppca_config_2d):
        result = PPCAReducer(ppca_config_2d).fit_transform(data_2d)
        assert result.shape == (N_SAMPLES, 2)

    def test_output_shape_3d(self, data_3d, ppca_config_3d):
        result = PPCAReducer(ppca_config_3d).fit_transform(data_3d)
        assert result.shape == (N_SAMPLES, 3)

    def test_no_nan_values(self, data_2d, ppca_config_2d):
        result = PPCAReducer(ppca_config_2d).fit_transform(data_2d)
        assert not np.isnan(result).any()

    def test_deterministic(self, data_2d, ppca_config_2d):
        r1 = PPCAReducer(ppca_config_2d).fit_transform(data_2d)
        r2 = PPCAReducer(ppca_config_2d).fit_transform(data_2d)
        np.testing.assert_allclose(r1, r2, atol=1e-5)

    def test_get_params(self, data_2d, ppca_config_2d):
        reducer = PPCAReducer(ppca_config_2d)
        reducer.fit_transform(data_2d)
        params = reducer.get_params()
        assert params["n_components"] == 2
        assert "regularization_mu" in params
        assert "standard_scale" in params
        assert "eigenvalue_ratios" in params

    def test_missing_background_raises(self, data_2d):
        """Without background_data attached, PPCAReducer must raise ValueError."""
        config = DimensionReductionConfig(
            n_components=2,
            random_state=SEED,
            regularization_mu=PPCA_TEST_MU,
        )
        with pytest.raises(ValueError, match="background"):
            PPCAReducer(config).fit_transform(data_2d)

    def test_regularization(self, data_2d, rng):
        """Higher regularization should also produce a valid projection."""
        config = _make_ppca_config(rng, regularization_mu=0.1)
        result = PPCAReducer(config).fit_transform(data_2d)
        assert result.shape == (N_SAMPLES, 2)
        assert np.isfinite(result).all()

    def test_singular_background_raises_without_regularization(self, data_2d, rng):
        """Σ_B is singular when n_background <= n_features; without
        regularization, the generalized eigenproblem must fail."""
        config = _make_ppca_config(rng, regularization_mu=0.0)
        with pytest.raises(Exception):
            PPCAReducer(config).fit_transform(data_2d)

    def test_eigenvalue_ratios(self, data_2d, ppca_config_2d):
        reducer = PPCAReducer(ppca_config_2d)
        reducer.fit_transform(data_2d)
        ratios = reducer.get_params()["eigenvalue_ratios"]
        assert len(ratios) == 2
        assert all(r > 0 for r in ratios), "eigenvalue ratios must be positive"
        assert ratios[0] >= ratios[1], "eigenvalue ratios must be descending"


# ---------------------------------------------------------------------------
# Cross-cutting tests
# ---------------------------------------------------------------------------

# Reducers that work with the bare config_2d fixture (no special params needed).
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
    """Tests that apply to every non-ρPCA reducer (ρPCA needs regularization)."""

    def test_returns_float_array(self, name, cls, data_2d, config_2d):
        result = cls(config_2d).fit_transform(data_2d)
        assert result.dtype in (np.float32, np.float64)

    def test_no_inf_values(self, name, cls, data_2d, config_2d):
        result = cls(config_2d).fit_transform(data_2d)
        assert not np.isinf(result).any()

    def test_output_finite(self, name, cls, data_2d, config_2d):
        result = cls(config_2d).fit_transform(data_2d)
        assert np.isfinite(result).all()


# Cross-cutting tests for ρPCA, identical to TestAllReducers but with the
# regularized config. Kept as a separate class because the bare config_2d
# fixture's regularization_mu=0.0 would otherwise raise LinAlgError on the
# (50, 20) test data.
@ppca_skip
class TestPPCACrossCutting:
    def test_returns_float_array(self, data_2d, config_2d_ppca):
        result = PPCAReducer(config_2d_ppca).fit_transform(data_2d)
        assert result.dtype in (np.float32, np.float64)

    def test_no_inf_values(self, data_2d, config_2d_ppca):
        result = PPCAReducer(config_2d_ppca).fit_transform(data_2d)
        assert not np.isinf(result).any()

    def test_output_finite(self, data_2d, config_2d_ppca):
        result = PPCAReducer(config_2d_ppca).fit_transform(data_2d)
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

    @ppca_skip
    def test_float16_input_produces_finite_output_ppca(self, rng, config_2d_ppca):
        data = (rng.standard_normal((N_SAMPLES, N_FEATURES)) * 0.04).astype(np.float16)
        result = PPCAReducer(config_2d_ppca).fit_transform(data.astype(np.float32))
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

    @pytest.mark.skipif(not HAS_PPCA, reason="PPCAReducer not yet implemented")
    def test_ppca_through_processor(self, data_2d, rng):
        from protspace.data.processors.base_processor import BaseProcessor
        from protspace.utils import get_reducers

        REDUCERS = get_reducers()
        bg = rng.standard_normal((15, N_FEATURES)).astype(np.float32)

        processor = BaseProcessor(
            {
                "random_state": SEED,
                "regularization_mu": PPCA_TEST_MU,
                "background_data": bg,
            },
            REDUCERS,
        )
        result = processor.process_reduction(data_2d, "ppca", 2)
        assert result["data"].shape == (N_SAMPLES, 2)
        assert np.isfinite(result["data"]).all()
        assert result["dimensions"] == 2
        assert isinstance(result["name"], str)
        assert isinstance(result["info"], dict)
        assert "eigenvalue_ratios" in result["info"]