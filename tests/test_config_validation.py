"""Tests for DimensionReductionConfig validation."""

import pytest

from protspace.utils.reducers import DimensionReductionConfig


class TestConfigValidation:
    def test_default_is_valid(self):
        config = DimensionReductionConfig()
        assert config.n_components == 2

    def test_n_components_3(self):
        config = DimensionReductionConfig(n_components=3)
        assert config.n_components == 3

    def test_n_components_k_for_prereduction(self):
        # k>3 is allowed (ρPCA/PCA pre-reduction); the viewer 2/3-D limit is
        # enforced at the CLI parse layer, not in the config.
        assert DimensionReductionConfig(n_components=50).n_components == 50

    def test_invalid_n_components(self):
        with pytest.raises(ValueError, match="n_components"):
            DimensionReductionConfig(n_components=1)

    def test_rho_output_scale_valid(self):
        for scale in ("none", "target_var", "unit_var"):
            assert DimensionReductionConfig(rho_output_scale=scale).rho_output_scale == scale

    def test_rho_output_scale_invalid(self):
        with pytest.raises(ValueError, match="rho_output_scale"):
            DimensionReductionConfig(rho_output_scale="bogus")

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="metric"):
            DimensionReductionConfig(metric="invalid")

    def test_min_dist_boundary(self):
        DimensionReductionConfig(min_dist=0.0)  # gte=0, should pass
        DimensionReductionConfig(min_dist=1.0)  # lte=1, should pass

    def test_min_dist_out_of_range(self):
        with pytest.raises(ValueError, match="min_dist"):
            DimensionReductionConfig(min_dist=-0.1)
        with pytest.raises(ValueError, match="min_dist"):
            DimensionReductionConfig(min_dist=1.1)

    def test_perplexity_boundaries(self):
        DimensionReductionConfig(perplexity=5)  # gte=5
        DimensionReductionConfig(perplexity=50)  # lte=50

    def test_perplexity_out_of_range(self):
        with pytest.raises(ValueError, match="perplexity"):
            DimensionReductionConfig(perplexity=4)
        with pytest.raises(ValueError, match="perplexity"):
            DimensionReductionConfig(perplexity=51)

    def test_learning_rate_must_be_positive(self):
        with pytest.raises(ValueError, match="learning_rate"):
            DimensionReductionConfig(learning_rate=0)

    def test_n_neighbors_must_be_positive(self):
        with pytest.raises(ValueError, match="n_neighbors"):
            DimensionReductionConfig(n_neighbors=0)

    def test_negative_random_state(self):
        with pytest.raises(ValueError, match="random_state"):
            DimensionReductionConfig(random_state=-1)

    def test_eps_must_be_positive(self):
        with pytest.raises(ValueError, match="eps"):
            DimensionReductionConfig(eps=0)
