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
# annoy compatibility shim — annoy can segfault or return empty results on
# certain platforms (notably macOS ARM64).  We detect that at first use and
# transparently swap in an sklearn-based replacement so PaCMAP / LocalMAP
# keep working everywhere.
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

    # Run the check in a subprocess so a segfault doesn't kill the main process
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

    # annoy is broken — swap in sklearn fallback
    import pacmap.pacmap as _pm
    from sklearn.neighbors import NearestNeighbors

    class _SklearnAnnoyIndex:
        """Drop-in AnnoyIndex replacement backed by sklearn NearestNeighbors."""

        def __init__(self, dim: int, metric: str = "euclidean"):
            self.dim = dim
            self.metric = metric
            self._items: list = []

        def add_item(self, i: int, vec) -> None:
            while len(self._items) <= i:
                self._items.append(None)
            self._items[i] = vec

        def set_seed(self, seed: int) -> None:
            pass  # determinism handled by sklearn

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

    # Constants and DimensionReductionConfig are imported from constants.py above

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
            # Create a dictionary of lowercase attribute names to their original names
            lowercase_fields = {
                data_field.name.lower(): data_field for data_field in fields(self)
            }
            result = []
            for param in method_parameters:
                # Exclude parameters not relevant for certain methods
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
                    if hasattr(
                        field_type_hint, "__args__"
                    ):  # Handle Literal, Union etc.
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
        """Transform data to lower dimensions."""
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Get parameters used for the reduction."""
        pass


class PCAReducer(DimensionReducer):
    """Principal Component Analysis reduction, preferring ARPACK solver."""

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        solver = "arpack"
        n_samples, n_annotations = data.shape
        k = self.config.n_components

        # ARPACK requires n_components < min(shape), fallback to full SVD otherwise
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
            self.used_solver = solver  # Store the solver that was actually used
            return result
        except Exception as e:
            logger.error(f"PCA failed using '{solver}' solver: {e}")
            raise

    def get_params(self) -> dict[str, Any]:
        """Get parameters used for the reduction."""
        params = {
            "n_components": self.config.n_components,
            # Report the solver used, default to 'arpack' if not set yet
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
# ρPCA reducer
# =============================================================================


def _solve_rho_eigenproblem(
    sigma_target: np.ndarray,
    sigma_background: np.ndarray,
    n_components: int,
    regularization_mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve Σ_T v = λ(Σ_B + μI)v and return the top eigenpairs."""
    from scipy.linalg import LinAlgError, eigh

    if sigma_target.ndim != 2 or sigma_background.ndim != 2:
        raise ValueError("ρPCA covariances must be 2D matrices.")
    if sigma_target.shape[0] != sigma_target.shape[1]:
        raise ValueError(f"Σ_T must be square, got {sigma_target.shape}.")
    if sigma_background.shape[0] != sigma_background.shape[1]:
        raise ValueError(f"Σ_B must be square, got {sigma_background.shape}.")
    if sigma_target.shape != sigma_background.shape:
        raise ValueError(
            f"Σ_T shape {sigma_target.shape} and Σ_B shape "
            f"{sigma_background.shape} differ."
        )

    d = sigma_background.shape[0]
    sigma_background_reg = sigma_background.astype(np.float64, copy=True)
    if regularization_mu > 0.0:
        sigma_background_reg += float(regularization_mu) * np.eye(d, dtype=np.float64)

    try:
        eigenvalues, eigenvectors = eigh(sigma_target, sigma_background_reg)
    except LinAlgError as err:
        raise LinAlgError(
            "ρPCA generalized eigenproblem failed: Σ_B is not positive "
            "definite even after Tikhonov regularization. Increase "
            f"regularization_mu (currently {regularization_mu}). "
            f"Original error: {err}"
        ) from err

    # Covariance matrices are PSD in theory, but generalized eigensolvers may
    # produce tiny negative round-off values. Keep finite, numerically non-negative
    # eigenpairs and sort descending by contrastive ratio.
    tol = 1e-12
    valid = np.isfinite(eigenvalues) & (eigenvalues > -tol)
    eigenvalues = np.maximum(eigenvalues[valid], 0.0)
    eigenvectors = eigenvectors[:, valid]
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if eigenvalues.size < n_components:
        raise ValueError(
            f"ρPCA produced only {eigenvalues.size} finite non-negative "
            f"generalized eigenvalues, but n_components={n_components} was "
            "requested. Increase regularization_mu, use a larger/non-degenerate "
            "background set, or request fewer components."
        )

    top_eigvals = eigenvalues[:n_components]
    top_eigvecs = eigenvectors[:, :n_components]

    # Deterministic sign convention.
    for k in range(top_eigvecs.shape[1]):
        j = int(np.argmax(np.abs(top_eigvecs[:, k])))
        if top_eigvecs[j, k] < 0:
            top_eigvecs[:, k] = -top_eigvecs[:, k]

    return top_eigvals, top_eigvecs


def _standard_scale_columns(X: np.ndarray) -> np.ndarray:
    """Column-center and variance-scale a matrix; constant columns remain zero."""
    mean = X.mean(axis=0, keepdims=True)
    centered = X - mean
    std = centered.std(axis=0, keepdims=True, ddof=1)
    std_safe = np.where(std < 1e-12, 1.0, std)
    return centered / std_safe


def _sample_covariance(centered: np.ndarray) -> np.ndarray:
    """Bessel-corrected sample covariance of an already-centered matrix."""
    n = centered.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 samples to compute covariance, got {n}.")
    cov = (centered.T @ centered) / (n - 1)
    return 0.5 * (cov + cov.T)


def _project_full_input(
    data: np.ndarray,
    X_target: np.ndarray,
    top_eigvecs: np.ndarray,
    standard_scale: bool,
) -> np.ndarray:
    """Project all displayed rows into the target-space coordinate frame."""
    target_mean = X_target.mean(axis=0, keepdims=True)
    if standard_scale:
        target_std = X_target.std(axis=0, keepdims=True, ddof=1)
        target_std_safe = np.where(target_std < 1e-12, 1.0, target_std)
        data_projected = (data - target_mean) / target_std_safe
    else:
        data_projected = data - target_mean
    return (data_projected @ top_eigvecs).astype(np.float64)


class PPCAReducer(DimensionReducer):
    """ρPCA via a generalized eigenproblem with an explicit background.

    The reducer is deliberately small and mathematical: it receives the target
    embedding matrix plus a prepared ``background_data`` ndarray attached by the
    pipeline or project command. It does not fetch annotations and it does not
    construct nuisance backgrounds itself.
    """

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        cfg = self.config
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"ρPCA expects 2D input, got shape {data.shape}.")

        n_samples, n_features = data.shape
        n_components = int(cfg.n_components)
        if n_components not in {2, 3}:
            raise ValueError(f"ρPCA supports 2 or 3 components, got {n_components}.")
        if n_components > n_features:
            raise ValueError(
                f"n_components={n_components} > n_features={n_features}."
            )
        if n_samples < 2:
            raise ValueError(
                f"ρPCA input has {n_samples} sample(s); need at least 2."
            )

        X_target, X_background, bg_source = self._resolve_target_background(data, cfg)

        if X_target.shape[0] < 2:
            raise ValueError(
                f"ρPCA target has {X_target.shape[0]} sample(s); need at least 2 "
                "to estimate a covariance matrix."
            )
        if X_background.shape[0] < 2:
            raise ValueError(
                f"ρPCA background has {X_background.shape[0]} sample(s); need "
                "at least 2 to estimate a covariance matrix."
            )

        standard_scale = bool(cfg.standard_scale)
        if standard_scale:
            X_target_p = _standard_scale_columns(X_target)
            X_background_p = _standard_scale_columns(X_background)
        else:
            X_target_p = X_target - X_target.mean(axis=0, keepdims=True)
            X_background_p = X_background - X_background.mean(axis=0, keepdims=True)

        sigma_target = _sample_covariance(X_target_p)
        sigma_background = _sample_covariance(X_background_p)

        try:
            top_eigvals, top_eigvecs = _solve_rho_eigenproblem(
                sigma_target=sigma_target,
                sigma_background=sigma_background,
                n_components=n_components,
                regularization_mu=float(cfg.regularization_mu),
            )
        except np.linalg.LinAlgError as err:
            deficiency = max(0, n_features - X_background.shape[0] + 1)
            raise type(err)(
                f"{err}\n  n_background={X_background.shape[0]}, "
                f"n_features={n_features}, deficiency ≥ {deficiency}. "
                "Try a larger regularization_mu or more background samples."
            ) from err

        self.eigenvalues_ = top_eigvals
        self.eigenvectors_ = top_eigvecs
        self.background_source_ = bg_source
        self.n_background_samples_ = int(X_background.shape[0])

        return _project_full_input(
            data=data,
            X_target=X_target,
            top_eigvecs=top_eigvecs,
            standard_scale=standard_scale,
        )

    def _resolve_target_background(self, data: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray, str]:
        background_data = getattr(cfg, "background_data", None)
        if background_data is None:
            raise ValueError(
                "ρPCA requires a background. Provide --ppca-background or use "
                "protspace prepare with --nuisance."
            )

        X_background = np.asarray(background_data, dtype=np.float64)
        if X_background.ndim != 2:
            raise ValueError(f"ρPCA background must be 2D, got {X_background.shape}.")
        if X_background.shape[1] != data.shape[1]:
            raise ValueError(
                f"ρPCA background has {X_background.shape[1]} features but input "
                f"has {data.shape[1]}. Use the same embedding model for target "
                "and background."
            )
        if not np.isfinite(X_background).all():
            raise ValueError("ρPCA background contains NaN or infinite values.")

        target_data = getattr(cfg, "target_data", None)
        if target_data is not None:
            X_target = np.asarray(target_data, dtype=np.float64)
            if X_target.ndim != 2 or X_target.shape[1] != data.shape[1]:
                raise ValueError(
                    f"ρPCA target_data shape {X_target.shape} is incompatible "
                    f"with input shape {data.shape}."
                )
            if not np.isfinite(X_target).all():
                raise ValueError("ρPCA target_data contains NaN or infinite values.")
        else:
            X_target = data

        source = str(getattr(cfg, "background_source", "external"))
        return X_target, X_background, source

    def get_params(self) -> dict[str, Any]:
        cfg = self.config
        params: dict[str, Any] = {
            "n_components": int(cfg.n_components),
            "random_state": int(cfg.random_state),
            "regularization_mu": float(cfg.regularization_mu),
            "standard_scale": bool(cfg.standard_scale),
        }
        if hasattr(self, "background_source_"):
            params["background_source"] = self.background_source_
        if hasattr(self, "n_background_samples_"):
            params["n_background_samples"] = self.n_background_samples_
        if hasattr(self, "eigenvalues_"):
            params["eigenvalue_ratios"] = self.eigenvalues_.tolist()
        details = getattr(cfg, "background_details", None)
        if details:
            params["background_details"] = details
        return params
