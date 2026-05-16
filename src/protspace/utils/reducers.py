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
    KPPCA_NAME,
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
# Shared eigenproblem solver for both ρPCA and k-ρPCA.
# =============================================================================

def _solve_rho_eigenproblem(
    sigma_target: np.ndarray,
    sigma_background: np.ndarray,
    n_components: int,
    regularization_mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve Σ_T v = λ Σ_B v and return the top `n_components` eigenpairs.

    Adds μI to Σ_B for Tikhonov regularization, solves with scipy.linalg.eigh,
    then keeps eigenpairs with positive finite eigenvalues (rhopca reference
    convention; see rhopca/utils/misc.py::generalized_eigen). The returned
    eigenvectors are sign-normalized so that, for each axis, the entry of
    largest magnitude is positive.

    Parameters
    ----------
    sigma_target : (d, d) float64
    sigma_background : (d, d) float64
    n_components : int
    regularization_mu : float, ≥ 0

    Returns
    -------
    eigvals : (k,) float64, descending. k ≤ n_components — fewer if some
        eigenvalues are filtered out as non-positive.
    eigvecs : (d, k) float64
    """
    from scipy.linalg import LinAlgError, eigh

    d = sigma_background.shape[0]
    if regularization_mu > 0.0:
        sigma_background = sigma_background + regularization_mu * np.eye(d, dtype=np.float64)

    try:
        eigenvalues, eigenvectors = eigh(sigma_target, sigma_background)
    except LinAlgError as err:
        raise LinAlgError(
            "ρPCA generalized eigenproblem failed: Σ_B is not positive "
            "definite even after Tikhonov regularization. Increase "
            f"regularization_mu (currently {regularization_mu}). "
            f"Original error: {err}"
        ) from err

    # Filter to positive finite eigenvalues (rhopca convention) and sort desc.
    valid = (eigenvalues > 0) & np.isfinite(eigenvalues)
    eigenvalues = eigenvalues[valid]
    eigenvectors = eigenvectors[:, valid]
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if eigenvalues.size < n_components:
        logger.warning(
            "ρPCA: only %d positive eigenvalues available, requested %d. "
            "Returning %d components.",
            eigenvalues.size, n_components, eigenvalues.size,
        )

    top_eigvals = eigenvalues[:n_components]
    top_eigvecs = eigenvectors[:, :n_components]

    # Sign normalization for deterministic output.
    for k in range(top_eigvecs.shape[1]):
        j = int(np.argmax(np.abs(top_eigvecs[:, k])))
        if top_eigvecs[j, k] < 0:
            top_eigvecs[:, k] = -top_eigvecs[:, k]

    return top_eigvals, top_eigvecs



def _standard_scale_columns(X: np.ndarray) -> np.ndarray:
    """Per-set standardization: column mean 0, column variance 1. Constant
    columns are left at zero to avoid amplifying noise."""
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

    Solves Σ_T v = λ Σ_B v for the top n_components eigenvectors. See
    Carilli, Jackson & Pachter 2025 (bioRxiv 2025.11.19.689125) for the
    objective; matches the rhopca reference implementation
    (https://github.com/pachterlab/rhopca) with the following choices:

      * Per-set standard-scaling (paper convention, scale_variance=True)
      * Bessel-corrected covariance (bias=False)
      * Fixed Tikhonov μ (default 1e-3); rhopca defaults to a trace-scaled
        heuristic μ = 1e-6 · tr(Σ_B) / d, which can be too small for PLM
        embeddings where n_B < d. The fixed default is safer in that regime.
      * Eigenvalues filtered to positive finite values before truncation.
    """

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        cfg = self.config

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"PPCA expects 2D input, got shape {data.shape}.")
        n_samples, n_features = data.shape

        if int(cfg.n_components) > n_features:
            raise ValueError(
                f"n_components={cfg.n_components} > n_features={n_features}."
            )

        X_target, X_background, bg_idx, bg_source = self._resolve_target_background(
            data, cfg
        )

        # Per-set standardization (paper convention).
        if cfg.standard_scale:
            X_target_p = _standard_scale_columns(X_target)
            X_background_p = _standard_scale_columns(X_background)
        else:
            X_target_p = X_target - X_target.mean(axis=0, keepdims=True)
            X_background_p = X_background - X_background.mean(axis=0, keepdims=True)

        sigma_target = _sample_covariance(X_target_p)
        sigma_background = _sample_covariance(X_background_p)

        try:
            top_eigvals, top_eigvecs = _solve_rho_eigenproblem(
                sigma_target, sigma_background,
                n_components=int(cfg.n_components),
                regularization_mu=float(cfg.regularization_mu),
            )
        except np.linalg.LinAlgError as err:
            deficiency = max(0, n_features - X_background.shape[0] + 1)
            raise type(err)(
                f"{err}\n  n_background={X_background.shape[0]}, "
                f"n_features={n_features}, deficiency ≥ {deficiency}. "
                f"Try a larger regularization_mu or more background samples."
            ) from err

        self.eigenvalues_ = top_eigvals
        self.eigenvectors_ = top_eigvecs
        self.background_source_ = bg_source

        # Project the standardized target onto top eigenvectors.
        return (X_target_p @ top_eigvecs).astype(np.float64)
    

    def _resolve_target_background(
            self, data: np.ndarray, cfg
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
            """Pick the background matrix the pipeline prepared and stashed on cfg.

            The pipeline owns all the strategy logic now; the reducer's job is to
            receive a ready-made X_target and X_background pair and solve the
            eigenproblem on them.
            """
            background_data = getattr(cfg, "background_data", None)
            if background_data is None:
                raise ValueError(
                    "ρPCA requires a prepared background. The pipeline should have "
                    "attached cfg.background_data via build_background(); this is a "
                    "configuration error. Pass --ppca-background and "
                    "--ppca-strategy, or contact the developers."
                )

            X_B = np.asarray(background_data, dtype=np.float64)
            if X_B.ndim != 2:
                raise ValueError(f"Background must be 2D, got {X_B.shape}.")
            if X_B.shape[1] != data.shape[1]:
                raise ValueError(
                    f"Background has {X_B.shape[1]} features but target has "
                    f"{data.shape[1]}. Same embedding model required."
                )
            source = getattr(cfg, "background_source", "pool")
            return data, X_B, None, source 

    def get_params(self) -> dict[str, Any]:
        cfg = self.config
        params = {
            "n_components": int(cfg.n_components),
            "random_state": int(cfg.random_state),
            "background_strategy": str(cfg.background_strategy),
            "regularization_mu": float(cfg.regularization_mu),
            "standard_scale": bool(cfg.standard_scale),
        }
        if hasattr(self, "background_source_"):
            params["background_source"] = self.background_source_
        if hasattr(self, "eigenvalues_"):
            params["eigenvalue_ratios"] = self.eigenvalues_.tolist()
        # The pipeline attaches background_details as a side-channel dict.
        details = getattr(cfg, "background_details", None)
        if details:
            params["background_details"] = details
        n_bg = getattr(cfg, "background_n_samples", None)
        if n_bg is not None:
            params["n_background_samples"] = int(n_bg)
        return params
    

# =============================================================================
# k-ρPCA  (kernel-weighted contrastive PCA — Jackson, Carilli & Pachter 2026)
#
# Reference: Jackson, K., Carilli, M., & Pachter, L. (2026).
#   "The Rayleigh Quotient and Contrastive Principal Component Analysis II."
#   bioRxiv 2026.04.08.717236. https://github.com/pachterlab/rhopca
#
# Objective
# ---------
# Given centered target X_T ∈ R^{n_T × d}, background X_B ∈ R^{n_B × d}, and
# kernel matrix K ∈ R^{n_T × n_T} encoding sample-similarity weights between
# target samples, k-ρPCA maximises
#
#     R(v) = v^T Σ_T^K v / v^T Σ_B v
#
# with
#     Σ_T^K = (1 / (n_T - 1)) X_T^T K X_T   (kernel-weighted target covariance)
#     Σ_B   = (1 / (n_B - 1)) X_B^T X_B     (standard background covariance)
#
# When K = I, k-ρPCA reduces exactly to ρPCA. In paper 2, K typically encodes
# spatial proximity (Visium spot coordinates). For protein embeddings we
# expose three kernel sources:
#
#   embedding   K_ij = κ(x_i, x_j) — pairwise kernel on standardised target
#               embeddings. Up-weights neighbours in pLM space → emphasises
#               local rather than global axes of variation.
#   similarity  K_ij = κ(d(seq_i, seq_j)) — MMseqs2-derived sequence similarity.
#               Up-weights homologues. Requires precomputed similarity matrix.
#   precomputed K is loaded directly from disk.
#
# Kernel functions
# ----------------
#   gaussian          K_ij = exp(-d_ij^2 / (2 h^2)),  h = sqrt(median(d)) by default
#   inverse_distance  K_ij = 1 / (d_ij + ε)
#   linear            K = I (sanity check; reduces k-ρPCA to ρPCA)
# =============================================================================


class KPPCAReducer(PPCAReducer):
    """k-ρPCA: kernel-weighted contrastive DR.

    Inherits background resolution and the eigenproblem solver from
    PPCAReducer; overrides only how Σ_T is computed.

    Attributes after fit (in addition to PPCAReducer attrs):
        kernel_source_       : "embedding" | "similarity" | "precomputed"
        kernel_              : kernel function name
        kernel_bandwidth_    : actual bandwidth used (may be auto-resolved)
        kernel_n_samples_    : n_T (size of K)
    """

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        cfg = self.config

        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"KPPCA expects 2D input, got shape {data.shape}.")
        n_samples, n_features = data.shape
        if int(cfg.n_components) > n_features:
            raise ValueError(
                f"n_components={cfg.n_components} > n_features={n_features}."
            )

        X_target, X_background, bg_idx, bg_source = self._resolve_target_background(
            data, cfg
        )

        # Standardize per-set (paper convention).
        if cfg.standard_scale:
            X_target_p = _standard_scale_columns(X_target)
            X_background_p = _standard_scale_columns(X_background)
        else:
            X_target_p = X_target - X_target.mean(axis=0, keepdims=True)
            X_background_p = X_background - X_background.mean(axis=0, keepdims=True)

        # Build the n_T × n_T kernel matrix K.
        K, kernel_meta = self._build_kernel_matrix(X_target_p, cfg)

        # Σ_T^K = X_T^T K X_T / (n_T - 1); X_T already centered.
        n_t = X_target_p.shape[0]
        sigma_target = (X_target_p.T @ K @ X_target_p) / (n_t - 1)
        sigma_target = 0.5 * (sigma_target + sigma_target.T)

        # Σ_B is standard (kernel-weighted background is opt-in and uncommon).
        if cfg.background_kernel:
            logger.warning(
                "background_kernel=True is unusual for k-ρPCA on protein "
                "embeddings; the background is assumed to lack the structure "
                "encoded by K. Using identity kernel for Σ_B."
            )
        sigma_background = _sample_covariance(X_background_p)

        try:
            top_eigvals, top_eigvecs = _solve_rho_eigenproblem(
                sigma_target, sigma_background,
                n_components=int(cfg.n_components),
                regularization_mu=float(cfg.regularization_mu),
            )
        except np.linalg.LinAlgError as err:
            deficiency = max(0, n_features - X_background.shape[0] + 1)
            raise type(err)(
                f"{err}\n  n_background={X_background.shape[0]}, "
                f"n_features={n_features}, deficiency ≥ {deficiency}."
            ) from err

        self.eigenvalues_ = top_eigvals
        self.eigenvectors_ = top_eigvecs
        self.background_source_ = bg_source
        self.kernel_source_ = kernel_meta["source"]
        self.kernel_ = kernel_meta["kernel"]
        self.kernel_bandwidth_ = kernel_meta["bandwidth"]
        self.kernel_n_samples_ = n_t

        return (X_target_p @ top_eigvecs).astype(np.float64)


    def _build_kernel_matrix(
        self, X_target_std: np.ndarray, cfg
    ) -> tuple[np.ndarray, dict]:
        """Construct the n_T × n_T kernel matrix K.

        Returns (K, metadata_dict). Metadata is recorded in get_params() for
        diagnostics in run.log.
        """
        from scipy.spatial.distance import pdist, squareform

        source = str(cfg.kernel_source)
        kernel = str(cfg.kernel)
        bandwidth_req = float(cfg.kernel_bandwidth)

        # Linear kernel: K = I, k-ρPCA reduces to ρPCA. Useful as a sanity check.
        if kernel == "linear":
            n_t = X_target_std.shape[0]
            return np.eye(n_t, dtype=np.float64), {
                "source": source, "kernel": "linear", "bandwidth": float("nan"),
            }

        # Build the pairwise distance vector based on source.
        if source == "embedding":
            distances = pdist(X_target_std, metric="euclidean")
        elif source == "similarity":
            sim = getattr(cfg, "kernel_similarity_matrix", None)
            if sim is None:
                raise ValueError(
                    "kernel_source='similarity' requires a precomputed "
                    "similarity matrix in the embedding bundle. Pass "
                    "--similarity to protspace prepare so MMseqs2 is run, "
                    "or use kernel_source='embedding' or 'precomputed'."
                )
            sim = np.asarray(sim, dtype=np.float64)
            if sim.shape != (X_target_std.shape[0], X_target_std.shape[0]):
                raise ValueError(
                    f"Similarity matrix shape {sim.shape} does not match "
                    f"target size {X_target_std.shape[0]}."
                )
            sim_clipped = np.clip(sim, 0.0, 1.0)
            np.fill_diagonal(sim_clipped, 1.0)
            distance_full = 1.0 - sim_clipped
            distances = squareform(distance_full, checks=False)
        elif source == "precomputed":
            K_raw = getattr(cfg, "kernel_precomputed_matrix", None)
            if K_raw is None:
                raise ValueError(
                    "kernel_source='precomputed' requires a kernel matrix via "
                    "--kppca-kernel-path."
                )
            K = np.asarray(K_raw, dtype=np.float64)
            if K.shape != (X_target_std.shape[0], X_target_std.shape[0]):
                raise ValueError(
                    f"Precomputed kernel shape {K.shape} does not match "
                    f"target size {X_target_std.shape[0]}."
                )
            K = 0.5 * (K + K.T)
            return K, {
                "source": "precomputed", "kernel": "precomputed",
                "bandwidth": float("nan"),
            }
        else:
            raise ValueError(f"Unknown kernel_source: {source!r}")

        # Resolve bandwidth and apply kernel function.
        if kernel == "gaussian":
            if bandwidth_req > 0.0:
                bandwidth = bandwidth_req
            else:
                # rhopca reference heuristic: sqrt(median(distances)).
                bandwidth = float(np.sqrt(np.median(distances)))
                if bandwidth <= 0.0:
                    raise ValueError(
                        "Auto bandwidth resolved to 0 — pairwise distances are "
                        "all zero (degenerate target). Set --kppca-kernel-bandwidth."
                    )
            weights = np.exp(-(distances ** 2) / (2.0 * bandwidth ** 2))
        elif kernel == "inverse_distance":
            weights = 1.0 / (distances + 1e-6)
            bandwidth = float("nan")
        else:
            raise ValueError(f"Unknown kernel: {kernel!r}")

        K = squareform(weights, checks=False)
        np.fill_diagonal(K, 1.0)
        return K, {"source": source, "kernel": kernel, "bandwidth": bandwidth}

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update({
            "kernel": str(self.config.kernel),
            "kernel_source": str(self.config.kernel_source),
            "kernel_bandwidth_requested": float(self.config.kernel_bandwidth),
            "background_kernel": bool(self.config.background_kernel),
        })
        if hasattr(self, "kernel_bandwidth_"):
            params["kernel_bandwidth_used"] = self.kernel_bandwidth_
        if hasattr(self, "kernel_n_samples_"):
            params["kernel_n_samples"] = int(self.kernel_n_samples_)
        return params