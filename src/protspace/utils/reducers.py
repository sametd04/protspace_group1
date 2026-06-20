import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Any, get_type_hints

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

# Re-export constants and config from lightweight module
from protspace.utils.constants import (  # noqa: F401
    LOCALMAP_NAME,
    MDS_NAME,
    METRIC_TYPES,
    PACMAP_NAME,
    PCA_NAME,
    PPCA_NAME,
    REDUCER_METHODS,
    TSNE_NAME,
    UMAP_NAME,
    DimensionReductionConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# annoy compatibility shim
# ---------------------------------------------------------------------------
_annoy_checked: bool = False


def _ensure_annoy_or_fallback() -> None:
    """Patch pacmap to use sklearn if annoy is broken. Only runs once."""
    global _annoy_checked
    if _annoy_checked:
        return
    _annoy_checked = True

    import subprocess
    import sys

    code = (
        "from annoy import AnnoyIndex; import random; random.seed(0); "
        "d=10; t=AnnoyIndex(d,'euclidean'); "
        "[t.add_item(i,[random.gauss(0,1) for _ in range(d)]) for i in range(50)]; "
        "t.build(5); "
        "exit(0 if len(t.get_nns_by_item(0,10))>=10 else 1)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], timeout=10, capture_output=True
        )
        if result.returncode == 0:
            return
    except Exception:
        pass

    import pacmap.pacmap as _pm
    from sklearn.neighbors import NearestNeighbors

    class _SklearnAnnoyIndex:
        def __init__(self, dim: int, metric: str = "euclidean"):
            self.dim = dim
            self.metric = metric
            self._items: list = []

        def add_item(self, i: int, vec) -> None:
            while len(self._items) <= i:
                self._items.append(None)
            self._items[i] = vec

        def set_seed(self, seed: int) -> None:
            pass

        def build(self, n_trees: int) -> None:
            self._data = np.array(self._items, dtype=np.float64)
            self._nn = NearestNeighbors(metric=self.metric, algorithm="auto")
            self._nn.fit(self._data)

        def get_nns_by_item(self, i: int, n: int) -> list[int]:
            k = min(n, len(self._data))
            _, idx = self._nn.kneighbors(self._data[i : i + 1], n_neighbors=k)
            return idx[0].tolist()

        def get_distance(self, i: int, j: int) -> float:
            return float(np.linalg.norm(self._data[i] - self._data[j]))

    _pm.AnnoyIndex = _SklearnAnnoyIndex  # type: ignore[attr-defined]
    logger.warning(
        "annoy is non-functional on this platform; "
        "using sklearn NearestNeighbors fallback for PaCMAP/LocalMAP"
    )

    def parameters_by_method(self, method: str) -> list[dict[str, Any]]:
        from pacmap import LocalMAP, PaCMAP
        from umap import UMAP

        method_map = {
            TSNE_NAME: TSNE,
            PCA_NAME: PCA,
            UMAP_NAME: UMAP,
            PACMAP_NAME: PaCMAP,
            MDS_NAME: MDS,
            LOCALMAP_NAME: LocalMAP,
        }

        if method not in method_map:
            return []

        def _get_parameter_desc_from_docstring(parameter: str, docstring: str) -> str:
            large_splits = []
            possible_split_variants = [
                f"{parameter} : ",
                f"{parameter}: ",
                f"{parameter}:",
                parameter,
            ]
            for split_variant in possible_split_variants:
                if split_variant in docstring:
                    large_splits = docstring.split(split_variant)
                    break
            if len(large_splits) == 0:
                return ""
            large_split = large_splits[0] if len(large_splits) == 1 else large_splits[1]
            param_split = (
                large_split.split("\n\n")[0]
                if "\n" in large_split
                else large_split.split("\n")[0]
            )
            param_split_cleaned = (
                param_split.replace("\n\n", "")
                .replace("\t", "")
                .replace("  ", " ")
                .replace("   ", " ")
                .strip()
            )
            return param_split_cleaned

        type_hints = get_type_hints(self.__class__)

        try:
            method_function = method_map[method]
            method_signature = inspect.signature(method_function)
            docstring = inspect.getdoc(method_function)
            method_parameters = list(method_signature.parameters.keys())
            lowercase_fields = {
                data_field.name.lower(): data_field for data_field in fields(self)
            }
            result = []
            for param in method_parameters:
                if method == MDS_NAME and param == "metric":
                    continue
                if method == UMAP_NAME and param == "learning_rate":
                    continue
                if method == LOCALMAP_NAME and param == "metric":
                    continue

                if param.lower() in lowercase_fields:
                    data_field = lowercase_fields[param.lower()]
                    field_type_hint = type_hints.get(data_field.name, Any)
                    field_type_name = getattr(
                        field_type_hint, "__name__", str(field_type_hint)
                    )
                    if hasattr(field_type_hint, "__args__"):
                        field_type_name = str(field_type_hint).replace("typing.", "")

                    doc_desc = _get_parameter_desc_from_docstring(
                        parameter=param, docstring=docstring
                    )
                    description = (
                        doc_desc
                        if doc_desc
                        else f"{data_field.name}: Config parameter. Default: {data_field.default}"
                    )

                    result.append(
                        {
                            "name": param.lower(),
                            "default": data_field.default,
                            "description": description,
                            "constraints": {
                                "type": field_type_name,
                                **data_field.metadata,
                            },
                        }
                    )
            return result
        except Exception as e:
            logger.error("Failed to extract parameter info: %s", e)
            return []


class DimensionReducer(ABC):
    """Abstract base class for dimension reduction methods."""

    def __init__(self, config: DimensionReductionConfig):
        self.config = config

    @abstractmethod
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        pass


class PCAReducer(DimensionReducer):
    """Principal Component Analysis reduction, preferring ARPACK solver."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        solver = "arpack"
        n_samples, n_annotations = data.shape
        k = self.config.n_components

        if k >= min(n_samples, n_annotations):
            logger.warning(
                f"PCA: n_components ({k}) >= min(shape) ({min(n_samples, n_annotations)}). "
                f"'arpack' solver unavailable, falling back to 'full' solver."
            )
            solver = "full"

        pca = PCA(n_components=k, svd_solver=solver)
        try:
            result = pca.fit_transform(data)
            self.explained_variance = pca.explained_variance_ratio_.tolist()
            self.used_solver = solver
            return result
        except Exception as e:
            logger.error(f"PCA failed using '{solver}' solver: {e}")
            raise

    def get_params(self) -> dict[str, Any]:
        params = {
            "n_components": self.config.n_components,
            "svd_solver": getattr(self, "used_solver", "arpack"),
        }
        if hasattr(self, "explained_variance"):
            params["explained_variance_ratio"] = self.explained_variance
        return params


class TSNEReducer(DimensionReducer):
    """t-SNE (t-Distributed Stochastic Neighbor Embedding) reduction."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return TSNE(
            n_components=self.config.n_components,
            perplexity=self.config.perplexity,
            learning_rate=self.config.learning_rate,
            metric=self.config.metric,
            random_state=self.config.random_state,
        ).fit_transform(data)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "perplexity": self.config.perplexity,
            "learning_rate": self.config.learning_rate,
            "metric": self.config.metric,
            "random_state": self.config.random_state,
        }


class UMAPReducer(DimensionReducer):
    """UMAP (Uniform Manifold Approximation and Projection) reduction."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        from umap import UMAP

        return UMAP(
            n_components=self.config.n_components,
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            random_state=self.config.random_state,
        ).fit_transform(data)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "n_neighbors": self.config.n_neighbors,
            "min_dist": self.config.min_dist,
            "metric": self.config.metric,
            "random_state": self.config.random_state,
        }


class PaCMAPReducer(DimensionReducer):
    """PaCMAP (Pairwise Controlled Manifold Approximation) reduction."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        from pacmap import PaCMAP

        _ensure_annoy_or_fallback()
        return PaCMAP(
            n_components=self.config.n_components,
            n_neighbors=self.config.n_neighbors,
            MN_ratio=self.config.mn_ratio,
            FP_ratio=self.config.fp_ratio,
            random_state=self.config.random_state,
        ).fit_transform(data)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "n_neighbors": self.config.n_neighbors,
            "MN_ratio": self.config.mn_ratio,
            "FP_ratio": self.config.fp_ratio,
            "random_state": self.config.random_state,
        }


class LocalMAPReducer(DimensionReducer):
    """LocalMAP (Local Manifold Approximation) reduction."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        from pacmap import LocalMAP

        _ensure_annoy_or_fallback()
        return LocalMAP(
            n_components=self.config.n_components,
            n_neighbors=self.config.n_neighbors,
            MN_ratio=self.config.mn_ratio,
            FP_ratio=self.config.fp_ratio,
            random_state=self.config.random_state,
        ).fit_transform(data, init="pca")

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "n_neighbors": self.config.n_neighbors,
            "MN_ratio": self.config.mn_ratio,
            "FP_ratio": self.config.fp_ratio,
            "random_state": self.config.random_state,
        }


class MDSReducer(DimensionReducer):
    """Multidimensional Scaling reduction."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return MDS(
            n_components=self.config.n_components,
            metric=True,
            n_init=self.config.n_init,
            max_iter=self.config.max_iter,
            eps=self.config.eps,
            random_state=self.config.random_state,
            dissimilarity=("precomputed" if self.config.precomputed else "euclidean"),
        ).fit_transform(data)

    def get_params(self) -> dict[str, Any]:
        return {
            "n_components": self.config.n_components,
            "n_init": self.config.n_init,
            "max_iter": self.config.max_iter,
            "eps": self.config.eps,
            "random_state": self.config.random_state,
        }


# =============================================================================
# ρPCA  (rhoPCA — Carilli, Jackson & Pachter 2025)
#
# Reference: Carilli, M., Jackson, K., & Pachter, L. (2025).
#   "The Rayleigh Quotient and Contrastive Principal Component Analysis I."
#   bioRxiv 2025.11.19.689125. https://github.com/pachterlab/rhopca
#
# Mathematical objective
# ----------------------
# Given target data X_T ∈ R^{n_T × d} and background data X_B ∈ R^{n_B × d},
# ρPCA finds directions v that maximize the Rayleigh quotient
#
#     R(v) = (v^T Σ_T v) / (v^T Σ_B v)
#
# where Σ_T and Σ_B are the per-set sample covariance matrices computed on
# separately standard-scaled data. The maximizers are the top eigenvectors
# of the generalized eigenproblem
#
#     Σ_T v = λ Σ_B v
#
# solved via scipy.linalg.eigh(Σ_T, Σ_B). Per-set standard-scaling (mean 0,
# unit variance) makes the covariance matrices correlation matrices and is
# the convention used by the rhopca reference implementation; it is
# essential for PLM embeddings whose dimensions have wildly heterogeneous
# scale.
#
# Background sources
# ------------------
# Two modes:
#
#   1. External (preferred, matches paper): user supplies a separate
#      background dataset via --ppca-background. The reducer receives it
#      through cfg.background_data and uses it directly.
#
#   2. Auto-split (no external background): partition the input data into
#      target = full input and background = subset chosen by strategy:
#      "random", "uniform", or "outlier". This is a heuristic; results
#      are usually inferior to a real background. The pipeline emits a
#      warning when no external background is provided.
#
# Regularization
# --------------
# Σ_B is singular whenever n_background ≤ d. For PLM embeddings (d ≥ 1024)
# this is the default regime. regularization_mu adds μI to Σ_B before
# solving. The default is 1e-3 — safe for typical PLM embeddings after
# standard-scaling.
# =============================================================================


class PPCAReducer(DimensionReducer):
    """ρPCA: contrastive dimension reduction via generalized eigendecomposition.

    Solves Σ_T v = λ Σ_B v for the top n_components eigenvectors.

    Background source is determined by config.background_strategy:
      - "external": use config.background_data (set by the pipeline from
                    --ppca-background). This is the canonical mode.
      - "random" / "uniform" / "outlier": auto-split policies on the input.

    Attributes after fit:
        eigenvalues_ : top eigenvalues, descending. Each is the Rayleigh
            quotient (target/background variance ratio) along its axis.
        eigenvectors_ : (d, n_components) generalized eigenvectors.
        background_indices_ : indices into the input array (auto-split only).
        background_source_ : "external" or one of the auto-split strategies.
    """

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        from scipy.linalg import LinAlgError, eigh

        cfg = self.config
        n_components = int(cfg.n_components)
        random_state = int(cfg.random_state)
        background_ratio = float(cfg.background_ratio)
        background_strategy = str(cfg.background_strategy)
        regularization_mu = float(cfg.regularization_mu)
        standard_scale = bool(cfg.standard_scale)

        # External background data is stashed on the config by the pipeline
        # under the attribute name `background_data`. It's not a dataclass
        # field because it can be a (potentially large) ndarray; we attach
        # it dynamically before the reducer is constructed.
        background_data = getattr(cfg, "background_data", None)

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"PPCA expects 2D input, got shape {data.shape}.")
        n_samples, n_features = data.shape

        if n_components > n_features:
            raise ValueError(
                f"n_components={n_components} > n_features={n_features}."
            )

        # --- Resolve target and background matrices ---
        if background_strategy == "external":
            if background_data is None:
                raise ValueError(
                    "background_strategy='external' requires a background "
                    "dataset, but config.background_data is None. Pass "
                    "--ppca-background <file.h5> on the CLI, or choose a "
                    "different background_strategy."
                )
            X_target = data
            X_background = np.asarray(background_data, dtype=np.float64)
            if X_background.ndim != 2:
                raise ValueError(
                    f"External background must be 2D, got shape "
                    f"{X_background.shape}."
                )
            if X_background.shape[1] != n_features:
                raise ValueError(
                    f"External background has {X_background.shape[1]} features "
                    f"but target has {n_features}. Both datasets must use the "
                    f"same embedding model (same dimensionality)."
                )
            self.background_indices_ = None
            self.background_source_ = "external"
        else:
            bg_idx = self._select_background_indices(
                data,
                ratio=background_ratio,
                strategy=background_strategy,
                random_state=random_state,
            )
            self.background_indices_ = bg_idx
            self.background_source_ = background_strategy
            if bg_idx.size < 2:
                raise ValueError(
                    f"Background set has {bg_idx.size} samples; need >= 2 "
                    f"for a non-degenerate covariance. Increase "
                    f"background_ratio or input size."
                )
            X_target = data
            X_background = data[bg_idx]

        if X_background.shape[0] < 2:
            raise ValueError(
                f"Background has {X_background.shape[0]} samples; need >= 2."
            )

        # --- Per-set standardization (matches Carilli/Jackson/Pachter convention) ---
        # Standard-scaling makes covariances into correlation matrices and
        # neutralizes per-dimension scale differences between target and
        # background. Essential for PLM embeddings.
        if standard_scale:
            X_target_p = self._standard_scale(X_target)
            X_background_p = self._standard_scale(X_background)
        else:
            X_target_p = X_target - X_target.mean(axis=0, keepdims=True)
            X_background_p = X_background - X_background.mean(axis=0, keepdims=True)

        # --- Sample covariance matrices ---
        sigma_target = self._compute_covariance(X_target_p)
        sigma_background = self._compute_covariance(X_background_p)

        # --- Tikhonov regularization ---
        if regularization_mu > 0.0:
            sigma_background = sigma_background + regularization_mu * np.eye(
                n_features, dtype=np.float64
            )

        # --- Solve generalized eigenproblem  Σ_T v = λ Σ_B v ---
        try:
            eigenvalues, eigenvectors = eigh(sigma_target, sigma_background)
        except (LinAlgError, ValueError) as err:
            deficiency = max(0, n_features - X_background.shape[0] + 1)
            raise LinAlgError(
                f"ρPCA generalized eigenproblem failed: Σ_B is not positive "
                f"definite. With n_background={X_background.shape[0]} and "
                f"n_features={n_features}, Σ_B is rank-deficient by at "
                f"least {deficiency} dimensions. Set regularization_mu to "
                f"a small positive value (e.g. 1e-3) to stabilize. "
                f"Original error: {err}"
            ) from err

        # eigh returns eigenvalues in ascending order; we want descending.
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        top_eigenvalues = eigenvalues[:n_components]
        top_eigenvectors = eigenvectors[:, :n_components]

        # --- Sign normalization (deterministic output) ---
        for k in range(n_components):
            j = int(np.argmax(np.abs(top_eigenvectors[:, k])))
            if top_eigenvectors[j, k] < 0:
                top_eigenvectors[:, k] = -top_eigenvectors[:, k]

        self.eigenvalues_ = top_eigenvalues
        self.eigenvectors_ = top_eigenvectors

        # --- Project the standardized target onto top eigenvectors ---
        projection = X_target_p @ top_eigenvectors
        return projection.astype(np.float64)

    def get_params(self) -> dict[str, Any]:
        """Return parameters and post-fit diagnostics for logging."""
        cfg = self.config
        params = {
            "n_components": int(cfg.n_components),
            "random_state": int(cfg.random_state),
            "background_ratio": float(cfg.background_ratio),
            "background_strategy": str(cfg.background_strategy),
            "regularization_mu": float(cfg.regularization_mu),
            "standard_scale": bool(cfg.standard_scale),
        }
        if hasattr(self, "background_source_"):
            params["background_source"] = self.background_source_
        if hasattr(self, "eigenvalues_"):
            params["eigenvalue_ratios"] = self.eigenvalues_.tolist()
        if (
            hasattr(self, "background_indices_")
            and self.background_indices_ is not None
        ):
            params["n_background_samples"] = int(self.background_indices_.size)
        return params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _standard_scale(X: np.ndarray) -> np.ndarray:
        """Per-set standardization: column mean 0, column variance 1.

        Constant columns (variance < 1e-12) are left at zero rather than
        amplifying numerical noise. This matches sklearn's StandardScaler
        behaviour with with_std=False on those columns.
        """
        mean = X.mean(axis=0, keepdims=True)
        centered = X - mean
        std = centered.std(axis=0, keepdims=True, ddof=1)
        # Avoid division by zero on constant features.
        std_safe = np.where(std < 1e-12, 1.0, std)
        return centered / std_safe

    @staticmethod
    def _compute_covariance(centered: np.ndarray) -> np.ndarray:
        """Bessel-corrected sample covariance of an already-centered matrix."""
        n = centered.shape[0]
        if n < 2:
            raise ValueError(
                f"Need at least 2 samples to compute covariance, got {n}."
            )
        cov = (centered.T @ centered) / (n - 1)
        return 0.5 * (cov + cov.T)

    @staticmethod
    def _select_background_indices(
        data: np.ndarray,
        *,
        ratio: float,
        strategy: str,
        random_state: int,
    ) -> np.ndarray:
        """Select indices of an auto-split background subset.

        Strategies:
          random  : uniform sample without replacement.
          uniform : k-means with k=n_background; pick the sample closest to
                    each centroid.
          outlier : score by mean distance to k=15 NN; pick the highest.
        """
        n_samples = data.shape[0]
        n_background = max(2, int(round(ratio * n_samples)))
        n_background = min(n_background, n_samples - 1)

        rng = np.random.default_rng(random_state)

        if strategy == "random":
            return rng.choice(n_samples, size=n_background, replace=False)

        if strategy == "uniform":
            from sklearn.cluster import KMeans
            from scipy.spatial.distance import cdist

            km = KMeans(
                n_clusters=n_background,
                n_init=4,
                random_state=random_state,
            ).fit(data)
            dists = cdist(km.cluster_centers_, data)
            closest = np.argmin(dists, axis=1)
            unique = np.unique(closest)
            if unique.size < n_background:
                remaining = np.setdiff1d(np.arange(n_samples), unique)
                extra = rng.choice(
                    remaining,
                    size=n_background - unique.size,
                    replace=False,
                )
                unique = np.concatenate([unique, extra])
            return unique[:n_background]

        if strategy == "outlier":
            from sklearn.neighbors import NearestNeighbors

            k = min(15, n_samples - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(data)
            distances, _ = nn.kneighbors(data)
            outlier_score = distances[:, 1:].mean(axis=1)
            order = np.argsort(outlier_score)[::-1]
            return order[:n_background]

        raise ValueError(
            f"Unknown background_strategy={strategy!r}. "
            f"Expected one of: external, random, uniform, outlier."
        )