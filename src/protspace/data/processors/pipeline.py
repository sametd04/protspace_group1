"""Unified reduction pipeline — replaces LocalProcessor and UniProtQueryProcessor.

Composes: loaders → annotation fetch → dimensionality reduction → output.
"""

import hashlib
import json
import logging
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from protspace.data.loaders import EmbeddingSet
from protspace.data.loaders.embedding_set import (
    format_param_suffix,
    format_projection_name,
)
from protspace.data.processors.base_processor import BaseProcessor
from protspace.utils import get_reducers
from protspace.utils.constants import MDS_NAME, METHOD_ALIASES, RHOPCA_NAME

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReducerParams:
    """User-configurable dimensionality reduction parameters."""

    metric: str = "euclidean"
    random_state: int = 42
    n_neighbors: int = 25
    min_dist: float = 0.1
    perplexity: float = 30.0
    learning_rate: float = 200.0
    mn_ratio: float = 0.5
    fp_ratio: float = 2.0
    n_init: int = 4
    max_iter: int = 300
    eps: float = 1e-6

    # ρPCA. Exactly two final pathways are supported:
    #   --rhopca-background: use an existing explicit background HDF5
    #   --nuisance: build an annotation-defined background inside prepare
    regularization_mu: float = 1e-6
    standard_scale: bool = False
    rho_output_scale: str = "none"
    rhopca_background_path: str = ""
    nuisance_specs: tuple[str, ...] = ()
    nuisance_ridge_alpha: float | None = None  # None → auto-tune per nuisance (CV-GCV)
    nuisance_cross_fit: int = 1
    nuisance_block_normalization: str = "none"
    nuisance_write_background: bool = True


@dataclass(frozen=True)
class MethodSpec:
    """A single DR method with its dimension count and parameter overrides.

    A method may be *chained*: ``pre`` holds ordered pre-reduction stages whose
    output feeds this (final) stage. E.g. ``rhopca50>umap2`` parses to
    ``MethodSpec("umap", 2, pre=(MethodSpec("rhopca", 50),))``. A bare method has
    an empty ``pre``. Only the final stage becomes a viewer projection.
    """

    method: str  # e.g. "umap"
    dims: int  # e.g. 2
    overrides: tuple[tuple[str, int | float | str], ...] = ()
    pre: tuple["MethodSpec", ...] = ()

    def _stage_str(self) -> str:
        base = f"{self.method}{self.dims}"
        if self.overrides:
            params = ";".join(f"{k}={v}" for k, v in self.overrides)
            return f"{base}:{params}"
        return base

    def __str__(self) -> str:
        if self.pre:
            return ">".join(p._stage_str() for p in self.pre) + ">" + self._stage_str()
        return self._stage_str()

    @property
    def overrides_dict(self) -> dict[str, int | float | str]:
        return dict(self.overrides)


@dataclass
class PipelineConfig:
    """Configuration for a ReductionPipeline run."""

    methods: list[MethodSpec]
    output_path: Path
    bundled: bool = True
    keep_tmp: bool = False
    no_scores: bool = False
    refetch_stages: frozenset[str] = field(default_factory=frozenset)
    annotations: list[str] | None = None
    intermediate_dir: Path | None = None
    reducer_params: ReducerParams = field(default_factory=ReducerParams)


# Valid override parameter names (from ReducerParams fields)
_VALID_OVERRIDE_KEYS = {f.name for f in fields(ReducerParams)}
# Field types for coercion
_FIELD_TYPES = {f.name: f.type for f in fields(ReducerParams)}


def _coerce_value(key: str, raw: str) -> int | float | str:
    """Coerce a string value to the appropriate type for the given parameter."""
    expected = _FIELD_TYPES.get(key)
    if expected is int:
        return int(raw)
    if expected is float:
        return float(raw)
    return raw


def _parse_stage(stage_spec: str) -> MethodSpec:
    """Parse one (unchained) stage like 'umap2' or 'rhopca50:rho_output_scale=target_var'."""
    if ":" in stage_spec:
        base, params_str = stage_spec.split(":", 1)
    else:
        base, params_str = stage_spec, ""

    method = "".join(filter(str.isalpha, base))
    digits = "".join(filter(str.isdigit, base))
    if not method or not digits:
        raise ValueError(f"Invalid method spec '{stage_spec}'. Expected e.g. 'umap2'.")
    # Normalize deprecated method tokens (e.g. 'ppca' → 'rhopca') to canonical.
    method = METHOD_ALIASES.get(method, method)
    dims = int(digits)

    overrides = {}
    if params_str:
        for pair in params_str.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"Invalid parameter format '{pair}' in '{stage_spec}'. "
                    f"Expected key=value."
                )
            key, val = pair.split("=", 1)
            key = key.strip()
            if key not in _VALID_OVERRIDE_KEYS:
                raise ValueError(
                    f"Unknown parameter '{key}' in '{stage_spec}'. "
                    f"Valid parameters: {', '.join(sorted(_VALID_OVERRIDE_KEYS))}"
                )
            overrides[key] = _coerce_value(key, val.strip())

    return MethodSpec(method=method, dims=dims, overrides=tuple(sorted(overrides.items())))


def parse_method_spec(method_spec: str) -> MethodSpec:
    """Parse a method spec, optionally chained with '>' pre-reduction stages.

    Examples:
        'pca2'              → MethodSpec('pca', 2)
        'umap2:n_neighbors=50;min_dist=0.1' → MethodSpec('umap', 2, overrides=...)
        'rhopca50>umap2'      → MethodSpec('umap', 2, pre=(MethodSpec('rhopca', 50),))

    The final (rightmost) stage is the viewer projection and must be 2- or 3-D.
    Earlier stages are pre-reductions and may have any dimension ≥ 2.
    """
    stages = [_parse_stage(s.strip()) for s in method_spec.split(">") if s.strip()]
    if not stages:
        raise ValueError(f"Empty method spec: {method_spec!r}")

    final = stages[-1]
    if final.dims not in {2, 3}:
        raise ValueError(
            f"Final projection '{final._stage_str()}' must be 2- or 3-D for the "
            f"viewer (got {final.dims}); k>3 is only allowed for pre-reduction "
            f"stages, e.g. '{final.method}{final.dims}>umap2'."
        )
    for pre in stages[:-1]:
        if pre.dims < 2:
            raise ValueError(f"Pre-reduction stage '{pre._stage_str()}' needs dims ≥ 2.")
    if len(stages) == 1:
        return final
    return replace(final, pre=tuple(stages[:-1]))


def chain_descriptor(spec: MethodSpec) -> str:
    """Short 'via …' descriptor of a chain's pre-stages, for projection names."""
    if not spec.pre:
        return ""
    return "via " + ">".join(f"{p.method}{p.dims}" for p in spec.pre)


def parse_methods_arg(raw: list[str]) -> list[MethodSpec]:
    """Parse repeatable -m arguments into a deduplicated MethodSpec list.

    Each element may be comma-separated: "pca2,umap2:n_neighbors=50"
    Semicolons separate parameters within a method override.
    """
    specs: list[MethodSpec] = []
    seen: set[MethodSpec] = set()
    for item in raw:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            spec = parse_method_spec(part)
            if spec not in seen:
                seen.add(spec)
                specs.append(spec)
    return specs


def disambiguation_suffix(spec: MethodSpec, method_counts: Counter) -> str:
    """Return a parameter suffix for projection name disambiguation.

    When the same (method, dims) pair appears multiple times in a run AND the
    given spec carries parameter overrides, return the abbreviated parameter
    string (e.g. "n=50, d=0.1"). Otherwise return "".

    A plain spec sitting alongside an override spec returns "" — the override
    spec alone carries the disambiguating suffix, and the plain spec keeps the
    default name (e.g. "ProtT5 — UMAP 2").
    """
    if method_counts[(spec.method, spec.dims)] > 1 and spec.overrides:
        return format_param_suffix(spec.overrides_dict)
    return ""


def _run_with_overridden_config(
    base: BaseProcessor,
    effective_params: dict[str, Any],
    method: str,
    dims: int,
    data: Any,
) -> dict[str, Any]:
    """Run base.process_reduction with effective_params, restoring the prior
    base.config afterwards.

    Centralizes the save/restore pattern so a leaked `precomputed` flag (or
    any other temporary key) cannot survive across reduction calls.
    """
    saved = base.config
    base.config = effective_params
    try:
        return base.process_reduction(data, method, dims)
    finally:
        base.config = saved




def _json_safe_projection_param(value: Any) -> Any:
    """Return a JSON-safe cache-key representation of projection params."""
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        h = hashlib.sha256(arr.view(np.uint8)).hexdigest()[:16]
        return {"ndarray_shape": list(arr.shape), "ndarray_dtype": str(arr.dtype), "sha256": h}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe_projection_param(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe_projection_param(v) for v in value]
    return value


def _cacheable_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert params, including ρPCA ndarray side channels, to cache-safe form."""
    return {str(k): _json_safe_projection_param(v) for k, v in params.items()}


def load_rhopca_background_h5(path: str | Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Load an explicit ρPCA background HDF5 as a matrix.

    The loader intentionally reuses ProtSpace's HDF5 conventions by forcing a
    name override, so background HDF5 files do not need a `model_name` attribute.
    """
    from protspace.data.loaders import load_h5

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ρPCA background not found: {path}")
    bg_set = load_h5([path], name_override="rhopca_background")
    B = np.asarray(bg_set.data, dtype=np.float64)
    if B.ndim != 2:
        raise ValueError(f"ρPCA background must be 2D, got {B.shape}.")
    if B.shape[0] < 2:
        raise ValueError(f"ρPCA background needs at least two rows, got {B.shape[0]}.")
    if not np.isfinite(B).all():
        raise ValueError("ρPCA background contains NaN or infinite values.")
    details = {
        "mode": "explicit",
        "background_path": str(path),
        "n_background": int(B.shape[0]),
        "n_features": int(B.shape[1]),
    }
    return B, list(bg_set.headers), details


def _safe_path_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return safe or "embedding"


class ReductionPipeline:
    """Unified pipeline: load → annotate → reduce → output.

    This class orchestrates the full data preparation workflow, replacing
    both LocalProcessor and UniProtQueryProcessor with a single composable
    pipeline that works with any input source via EmbeddingSet.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        reducer_dict = asdict(config.reducer_params)
        self.base = BaseProcessor(reducer_dict, get_reducers())
        self._rhopca_background_cache: dict[str, tuple[np.ndarray, list[str], dict[str, Any]]] = {}
        # Per-run memo of annotation-defined nuisance backgrounds, so several ρPCA
        # projections sharing one nuisance/embedding (e.g. rhopca2 and rhopca3, or a
        # viewer projection plus a rhopca-k pre-reduction) build it (and write its
        # diagnostics) only once.
        self._nuisance_background_cache: dict[tuple, dict[str, Any]] = {}
        # Per-run memo of pre-reduction outputs, so several downstream methods on
        # the same pre-reduction (e.g. rhopca50>umap2 and rhopca50>pca2) reuse it.
        self._prereduce_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}

    def run(self, embedding_sets: list[EmbeddingSet]) -> Path:
        """Execute the full pipeline.

        Args:
            embedding_sets: One or more EmbeddingSets to process.

        Returns:
            Path to the output file/directory.
        """
        if not embedding_sets:
            raise ValueError("At least one EmbeddingSet is required.")

        # Merge same-name embedding sets (union their proteins)
        from protspace.data.loaders.embedding_set import merge_same_name_sets

        embedding_sets = merge_same_name_sets(embedding_sets)

        # Validate all sets share the same headers (or compute intersection)
        all_headers = self._validate_headers(embedding_sets)

        # Fetch annotations (pass embedding sets so FASTA sequences can be reused)
        metadata = self._fetch_annotations(all_headers, embedding_sets)

        # Apply score stripping
        if self.config.no_scores:
            from protspace.data.annotations.scores import strip_scores_from_df

            metadata = strip_scores_from_df(metadata)

        # Build full metadata with all headers
        full_metadata = pd.DataFrame({"identifier": all_headers})
        if len(metadata.columns) > 1:
            metadata = metadata.astype(str)
            id_col = metadata.columns[0]
            if id_col != "identifier":
                metadata = metadata.rename(columns={id_col: "identifier"})
            full_metadata = full_metadata.merge(
                metadata.drop_duplicates("identifier"),
                on="identifier",
                how="left",
            )
        metadata = full_metadata

        # DR: each embedding set × each method
        all_reductions = self._run_reductions(embedding_sets, metadata)

        # Create and save output
        output = self.base.create_output(metadata, all_reductions, all_headers)
        self.base.save_output(
            output, self.config.output_path, bundled=self.config.bundled
        )

        logger.info(
            f"Processed {len(all_headers)} proteins, "
            f"{len(embedding_sets)} embedding(s), "
            f"{len(all_reductions)} projection(s)"
        )
        logger.info(f"Output saved to: {self.config.output_path}")

        # Clean up intermediate dir if not keeping
        if (
            not self.config.keep_tmp
            and self.config.intermediate_dir
            and self.config.intermediate_dir.exists()
        ):
            shutil.rmtree(self.config.intermediate_dir)

        return self.config.output_path

    @staticmethod
    def _extract_sequences(embedding_sets: list[EmbeddingSet]) -> dict[str, str]:
        """Extract protein sequences from FASTA files referenced by embedding sets."""
        sequences = {}
        for emb_set in embedding_sets:
            if emb_set.fasta_path and Path(emb_set.fasta_path).exists():
                from protspace.data.io.fasta import parse_fasta
                from protspace.data.loaders.h5 import parse_identifier

                raw = parse_fasta(Path(emb_set.fasta_path))
                sequences.update({parse_identifier(h): s for h, s in raw.items()})
        return sequences

    def _validate_headers(self, embedding_sets: list[EmbeddingSet]) -> list[str]:
        """Ensure all embedding sets share the same identifiers.

        If they differ, compute intersection and warn.
        """
        if len(embedding_sets) == 1:
            return embedding_sets[0].headers

        sets = [set(es.headers) for es in embedding_sets]
        common = sets[0]
        for s in sets[1:]:
            common = common & s

        if not common:
            raise ValueError(
                "No common protein identifiers found across embedding sets."
            )

        # Check if any set lost identifiers
        for es in embedding_sets:
            diff = set(es.headers) - common
            if diff:
                logger.warning(
                    f"Embedding '{es.name}': dropping {len(diff)} proteins "
                    f"not present in all sets."
                )

        # Use the order from the first set, filtered to common
        common_headers = [h for h in embedding_sets[0].headers if h in common]

        # Re-order data in each set to match common_headers
        for es in embedding_sets:
            if es.headers != common_headers:
                idx_map = {h: i for i, h in enumerate(es.headers)}
                indices = [idx_map[h] for h in common_headers]
                es.data = es.data[indices]
                es.headers = common_headers

        return common_headers

    def _fetch_annotations(
        self, headers: list[str], embedding_sets: list[EmbeddingSet] = None
    ) -> pd.DataFrame:
        """Fetch annotations from APIs with incremental caching support."""
        from protspace.data.annotations.manager import ProteinAnnotationManager

        # Extract sequences from FASTA files (if available) to avoid re-fetching
        sequences = self._extract_sequences(embedding_sets) if embedding_sets else {}

        annotation_names, csv_path = self._resolve_annotation_names()

        # Load user CSV if provided
        csv_df = None
        if csv_path:
            logger.info(f"Loading custom annotations from: {csv_path}")
            csv_df = pd.read_csv(
                csv_path,
                sep="\t" if csv_path.endswith(".tsv") else ",",
            )
            id_col = csv_df.columns[0]
            if id_col != "identifier":
                csv_df = csv_df.rename(columns={id_col: "identifier"})

        if annotation_names:
            from protspace.data.annotations.configuration import (
                AnnotationConfiguration,
            )

            annotations_list = AnnotationConfiguration(
                annotation_names
            ).user_annotations
        else:
            annotations_list = None

        # CSV-only: no API annotations requested
        if annotations_list is None and csv_df is not None:
            return csv_df

        keep_tmp = self.config.keep_tmp
        intermediate_dir = self.config.intermediate_dir
        refetch = self.config.refetch_stages
        _ANN_SOURCES = ("uniprot", "taxonomy", "interpro", "ted", "biocentral")
        refetching_annotations = bool(refetch & set(_ANN_SOURCES))

        if keep_tmp and intermediate_dir:
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            cache_path = intermediate_dir / "all_annotations.parquet"

            if cache_path.exists():
                cached_df = pd.read_parquet(cache_path)
                cached_annotations = set(cached_df.columns) - {"identifier"}

                if annotations_list is None:
                    from protspace.data.annotations.configuration import (
                        ANNOTATION_GROUPS,
                    )

                    required = set(ANNOTATION_GROUPS["default"])
                else:
                    required = set(annotations_list)

                missing = required - cached_annotations

                if not missing and not refetching_annotations:
                    logger.warning("Using cached annotations")
                    if annotations_list:
                        cols = ["identifier"] + [
                            f for f in annotations_list if f in cached_df.columns
                        ]
                        api_df = cached_df[cols]
                    else:
                        api_df = cached_df

                    # Warn if cached annotations are all empty
                    data_cols = [c for c in api_df.columns if c != "identifier"]
                    if data_cols:
                        non_empty = api_df[data_cols].apply(
                            lambda col: (col != "").any()
                        )
                        if not non_empty.any():
                            logger.warning(
                                "All cached annotations are empty. This may be "
                                "from a previous run with non-UniProt identifiers. "
                                "Use --refetch annotations to re-fetch, or provide "
                                "a FASTA file with -f."
                            )

                    return self._merge_csv(api_df, csv_df)

                from protspace.data.annotations.configuration import (
                    AnnotationConfiguration,
                )

                sources = AnnotationConfiguration.determine_sources_to_fetch(
                    cached_annotations, required
                )

                if refetching_annotations:
                    # Override with explicitly requested sources
                    sources = {src: src in refetch for src in _ANN_SOURCES}
                    refetched = [s for s in _ANN_SOURCES if sources[s]]
                    logger.info(f"--refetch: re-fetching {', '.join(refetched)}")
                    # Drop cached columns for refetched sources so manager
                    # re-fetches them
                    from protspace.data.annotations.configuration import (
                        AnnotationConfiguration as AnnCfg,
                    )

                    cols_to_drop = set()
                    for src in refetched:
                        cols_to_drop |= AnnCfg.categorize_annotations_by_source(
                            cached_annotations
                        ).get(src, set())
                    if cols_to_drop:
                        cached_df = cached_df.drop(
                            columns=[c for c in cols_to_drop if c in cached_df.columns]
                        )
                else:
                    logger.info(f"Missing annotations: {missing}")

                api_df = ProteinAnnotationManager(
                    headers=headers,
                    annotations=annotations_list,
                    output_path=cache_path,
                    sequences=sequences,
                    cached_data=cached_df,
                    sources_to_fetch=sources,
                ).to_pd()
                return self._merge_csv(api_df, csv_df)
            else:
                api_df = ProteinAnnotationManager(
                    headers=headers,
                    annotations=annotations_list,
                    output_path=cache_path,
                    sequences=sequences,
                ).to_pd()
                return self._merge_csv(api_df, csv_df)
        else:
            api_df = ProteinAnnotationManager(
                headers=headers,
                annotations=annotations_list,
                output_path=None,
                sequences=sequences,
            ).to_pd()
            return self._merge_csv(api_df, csv_df)

    def _resolve_annotation_names(self) -> tuple[list[str], str | None]:
        """Parse annotation arguments into annotation names and optional CSV path.

        Returns:
            Tuple of (annotation_names, csv_path_or_None)
        """
        if not self.config.annotations:
            return [], None

        names = []
        csv_path = None
        for item in self.config.annotations:
            item = item.strip()
            if not item:
                continue
            if item.endswith((".csv", ".tsv")):
                csv_path = item
            else:
                for part in item.split(","):
                    part = part.strip()
                    if part:
                        names.append(part)
        return names, csv_path

    @staticmethod
    def _merge_csv(api_df: pd.DataFrame, csv_df: pd.DataFrame | None) -> pd.DataFrame:
        """Merge user CSV annotations onto API annotations. CSV wins on collision."""
        if csv_df is None:
            return api_df

        merged = api_df.merge(
            csv_df.drop_duplicates("identifier"),
            on="identifier",
            how="left",
            suffixes=("_api", ""),
        )
        # Drop API-suffixed duplicates so CSV values win
        for col in list(merged.columns):
            if col.endswith("_api"):
                base = col.removesuffix("_api")
                if base in merged.columns:
                    merged = merged.drop(columns=[col])
                else:
                    merged = merged.rename(columns={col: base})
        return merged

    # --- Projection caching helpers ---

    def _projection_cache_path(
        self,
        embedding_name: str,
        method: str,
        dims: int,
        effective_params: dict[str, Any] | None = None,
    ) -> Path | None:
        cache_dir = self.config.intermediate_dir
        if not cache_dir or not self.config.keep_tmp:
            return None
        key_dict = {
            "embedding": embedding_name,
            "method": method,
            "dims": dims,
            "params": _cacheable_params(effective_params or asdict(self.config.reducer_params)),
        }
        key_json = json.dumps(key_dict, sort_keys=True, default=str)
        h = hashlib.sha256(key_json.encode()).hexdigest()[:12]
        return cache_dir / f"proj_{embedding_name}_{method}{dims}_{h}.npz"

    def _load_cached_projection(
        self,
        embedding_name: str,
        method: str,
        dims: int,
        effective_params: dict[str, Any] | None = None,
        param_suffix: str = "",
    ) -> dict[str, Any] | None:
        path = self._projection_cache_path(
            embedding_name, method, dims, effective_params
        )
        if (
            path is None
            or not path.exists()
            or "projections" in self.config.refetch_stages
        ):
            return None
        logger.info(
            "Using cached %s %d projection for '%s'",
            method.upper(),
            dims,
            embedding_name,
        )
        cached = np.load(path, allow_pickle=False)
        info = json.loads(str(cached["info"]))
        return {
            "name": format_projection_name(embedding_name, method, dims, param_suffix),
            "dimensions": dims,
            "info": info,
            "data": cached["data"],
        }

    def _save_projection_cache(
        self,
        embedding_name: str,
        method: str,
        dims: int,
        reduction: dict,
        effective_params: dict[str, Any] | None = None,
    ) -> None:
        path = self._projection_cache_path(
            embedding_name, method, dims, effective_params
        )
        if path is None:
            return
        np.savez(
            path, data=reduction["data"], info=np.array(json.dumps(reduction["info"]))
        )

    # --- ρPCA background helpers ---

    def _background_output_dir(self, emb_set: EmbeddingSet) -> Path | None:
        params = self.config.reducer_params
        if not params.nuisance_write_background:
            return None
        base = self.config.intermediate_dir
        if base is None:
            base = self.config.output_path if self.config.output_path.suffix == "" else self.config.output_path.parent
        return base / "rhopca_nuisance_backgrounds" / _safe_path_component(emb_set.name)

    def _load_explicit_rhopca_background(
        self, background_path: str
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        cache_key = str(Path(background_path).resolve())
        if cache_key not in self._rhopca_background_cache:
            self._rhopca_background_cache[cache_key] = load_rhopca_background_h5(background_path)
        return self._rhopca_background_cache[cache_key]

    def _prepare_rhopca_params(
        self,
        *,
        emb_set: EmbeddingSet,
        metadata: pd.DataFrame,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach the explicit background matrix required by RhoPCAReducer."""
        if emb_set.precomputed:
            raise ValueError("ρPCA cannot run on precomputed similarity/distance matrices.")

        background_path = str(params.get("rhopca_background_path", "") or "")
        nuisance_specs = tuple(params.get("nuisance_specs", ()) or ())

        if background_path and nuisance_specs:
            raise ValueError(
                "ρPCA received both --rhopca-background and --nuisance. Choose one "
                "background source for a given run."
            )

        if background_path:
            B, bg_headers, details = self._load_explicit_rhopca_background(background_path)
            if B.shape[1] != emb_set.data.shape[1]:
                raise ValueError(
                    f"ρPCA background {background_path!r} has {B.shape[1]} "
                    f"features but embedding '{emb_set.name}' has "
                    f"{emb_set.data.shape[1]}. Use the same embedding model."
                )
            return {
                **params,
                "background_data": B,
                "background_source": "explicit",
                "background_n_samples": int(B.shape[0]),
                "background_details": {
                    **details,
                    "background_headers_preview": bg_headers[:10],
                },
            }

        if nuisance_specs:
            # Memoize per (embedding set + everything that affects the background),
            # so multiple ρPCA projections in one run don't rebuild it or re-write
            # its diagnostics.
            cache_key = (
                emb_set.name,
                nuisance_specs,
                params.get("nuisance_ridge_alpha", None),
                int(params.get("nuisance_cross_fit", 1)),
                str(params.get("nuisance_block_normalization", "none")),
                float(params.get("regularization_mu", 1e-6)),
                int(params.get("random_state", 42)),
            )
            cached = self._nuisance_background_cache.get(cache_key)
            if cached is None:
                from protspace.utils.annotation_nuisance_background import (
                    build_annotation_nuisance_background,
                )

                result = build_annotation_nuisance_background(
                    ids=list(emb_set.headers),
                    X=np.asarray(emb_set.data, dtype=np.float64),
                    annotations=metadata,
                    nuisance_specs=list(nuisance_specs),
                    ridge_alpha=params.get("nuisance_ridge_alpha", None),
                    cross_fit=int(params.get("nuisance_cross_fit", 1)),
                    random_state=int(params.get("random_state", 42)),
                    block_normalization=str(params.get("nuisance_block_normalization", "none")),
                    target_name=emb_set.name,
                    regularization_mu=float(params.get("regularization_mu", 1e-6)),
                )
                out_dir = self._background_output_dir(emb_set)
                if out_dir is not None:
                    result.write(out_dir)
                    logger.info("Wrote ρPCA nuisance background diagnostics to %s", out_dir)
                cached = {
                    "background_data": result.background,
                    "background_source": "annotation_nuisance",
                    "background_n_samples": int(result.background.shape[0]),
                    "background_details": result.manifest,
                }
                self._nuisance_background_cache[cache_key] = cached

            return {**params, **cached}

        raise ValueError(
            "ρPCA requested but no background source was provided. Use either "
            "--rhopca-background background.h5 or, in protspace prepare, one or more "
            "--nuisance specifications."
        )


    # --- Dimensionality reduction ---

    def _run_prereduction(
        self,
        emb_set: EmbeddingSet,
        metadata: pd.DataFrame,
        pre_stages: tuple[MethodSpec, ...],
        global_params: dict[str, Any],
    ) -> np.ndarray:
        """Apply the ordered pre-reduction stages, returning the reduced matrix.

        A ρPCA pre-stage must be the first stage (it needs the raw 1024-D
        embedding to match its background). The result is memoized per run so
        several downstream methods can share one pre-reduction.
        """
        key = (emb_set.name, tuple(str(p) for p in pre_stages))
        if key in self._prereduce_cache:
            return self._prereduce_cache[key]

        data = emb_set.data
        for i, pre in enumerate(pre_stages):
            if pre.method not in self.base.reducers:
                raise ValueError(f"Unknown pre-reduction method: {pre.method}")
            if pre.method == RHOPCA_NAME and i != 0:
                raise ValueError(
                    "ρPCA pre-reduction must be the first stage — it needs the raw "
                    "embedding to match its background."
                )
            pre_params = {**global_params, **pre.overrides_dict}
            if pre.method == RHOPCA_NAME:
                pre_params = self._prepare_rhopca_params(
                    emb_set=emb_set, metadata=metadata, params=pre_params
                )
            logger.info(
                "Pre-reducing '%s' with %s %d", emb_set.name, pre.method.upper(), pre.dims
            )
            result = _run_with_overridden_config(
                self.base, pre_params, pre.method, pre.dims, data
            )
            data = np.asarray(result["data"])

        self._prereduce_cache[key] = data
        return data

    def _run_reductions(
        self, embedding_sets: list[EmbeddingSet], metadata: pd.DataFrame
    ) -> list[dict[str, Any]]:
        """Run dimensionality reduction on all embedding sets."""
        all_reductions = []
        cached_projections: list[str] = []  # e.g. "PCA 2 (prot_t5)"
        computed_count = 0

        # Pre-compute which (method, dims) pairs appear multiple times
        method_counts = Counter(
            (spec.method, spec.dims) for spec in self.config.methods
        )

        global_params = asdict(self.config.reducer_params)

        for emb_set in embedding_sets:
            if emb_set.precomputed:
                cached = self._load_cached_projection(
                    emb_set.name, MDS_NAME, 2, global_params
                )
                if cached:
                    all_reductions.append(cached)
                    cached_projections.append(f"MDS 2 ({emb_set.name})")
                    continue
                logger.info(f"Applying MDS 2 to '{emb_set.name}' (precomputed)")
                effective_params = {**global_params, "precomputed": True}
                reduction = _run_with_overridden_config(
                    self.base, effective_params, MDS_NAME, 2, emb_set.data
                )
                reduction["name"] = format_projection_name(emb_set.name, MDS_NAME, 2)
                all_reductions.append(reduction)
                self._save_projection_cache(
                    emb_set.name, MDS_NAME, 2, reduction, global_params
                )
                computed_count += 1
                continue

            for spec in self.config.methods:
                method, dims = spec.method, spec.dims

                if method not in self.base.reducers:
                    logger.warning(f"Unknown method: {method}. Skipping.")
                    continue

                # Resolve the input matrix: raw embedding, or a pre-reduction chain.
                if spec.pre:
                    input_data = self._run_prereduction(
                        emb_set, metadata, spec.pre, global_params
                    )
                else:
                    input_data = emb_set.data

                # Merge global defaults with per-method overrides
                effective_params = {**global_params, **spec.overrides_dict}

                if method == RHOPCA_NAME:
                    if spec.pre:
                        raise ValueError(
                            "ρPCA as the final stage after a pre-reduction is not "
                            "supported (its background is 1024-D). Use ρPCA as the "
                            "pre-reduction stage, e.g. 'rhopca50>umap2'."
                        )
                    effective_params = self._prepare_rhopca_params(
                        emb_set=emb_set,
                        metadata=metadata,
                        params=effective_params,
                    )

                # Build param suffix: pre-reduction descriptor + override disambiguation.
                param_suffix = disambiguation_suffix(spec, method_counts)
                chain_desc = chain_descriptor(spec)
                if chain_desc:
                    param_suffix = (
                        f"{chain_desc}, {param_suffix}" if param_suffix else chain_desc
                    )
                    # Distinguish cache keys of chains that share a final (method, dims).
                    effective_params = {
                        **effective_params,
                        "__prereduce__": [str(p) for p in spec.pre],
                    }

                cached = self._load_cached_projection(
                    emb_set.name, method, dims, effective_params, param_suffix
                )
                if cached:
                    all_reductions.append(cached)
                    cached_projections.append(
                        f"{method.upper()} {dims} ({emb_set.name})"
                    )
                    continue

                logger.info(f"Applying {method.upper()} {dims} to '{emb_set.name}'")
                reduction = _run_with_overridden_config(
                    self.base, effective_params, method, dims, input_data
                )

                reduction["name"] = format_projection_name(
                    emb_set.name, method, dims, param_suffix
                )
                all_reductions.append(reduction)
                self._save_projection_cache(
                    emb_set.name, method, dims, reduction, effective_params
                )
                computed_count += 1

        if cached_projections:
            logger.warning(
                "Using %d cached projection%s",
                len(cached_projections),
                "s" if len(cached_projections) != 1 else "",
            )

        return all_reductions
