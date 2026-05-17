"""Unified reduction pipeline — replaces LocalProcessor and UniProtQueryProcessor.

Composes: loaders → annotation fetch → dimensionality reduction → output.
"""

import hashlib
import json
import logging
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
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
from protspace.utils.constants import MDS_NAME, PPCA_NAME

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

    # ρPCA-specific parameters
    regularization_mu: float = 1e-3
    background_ratio: float = 0.3
    background_strategy: str = "outlier"
    standard_scale: bool = True
    # Path to external background HDF5. When set, takes precedence over
    # background_strategy and forces strategy="external" in the reducer.
    background_path: str = ""


@dataclass(frozen=True)
class MethodSpec:
    """A single DR method with its dimension count and parameter overrides."""

    method: str
    dims: int
    overrides: tuple[tuple[str, int | float | str | bool], ...] = ()

    def __str__(self) -> str:
        base = f"{self.method}{self.dims}"
        if self.overrides:
            params = ";".join(f"{k}={v}" for k, v in self.overrides)
            return f"{base}:{params}"
        return base

    @property
    def overrides_dict(self) -> dict[str, int | float | str | bool]:
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
_FIELD_TYPES = {f.name: f.type for f in fields(ReducerParams)}


def _coerce_value(key: str, raw: str) -> int | float | str | bool:
    """Coerce a string value to the appropriate type for the given parameter."""
    expected = _FIELD_TYPES.get(key)
    if expected is bool:
        # Accept "true"/"false"/"1"/"0" case-insensitively.
        if raw.lower() in ("true", "1", "yes"):
            return True
        if raw.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"Invalid boolean value {raw!r} for {key!r}")
    if expected is int:
        return int(raw)
    if expected is float:
        return float(raw)
    return raw


def parse_method_spec(method_spec: str) -> MethodSpec:
    """Parse a method spec string into a MethodSpec."""
    if ":" in method_spec:
        base, params_str = method_spec.split(":", 1)
    else:
        base, params_str = method_spec, ""

    method = "".join(filter(str.isalpha, base))
    dims = int("".join(filter(str.isdigit, base)))

    overrides = {}
    if params_str:
        for pair in params_str.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"Invalid parameter format '{pair}' in '{method_spec}'. "
                    f"Expected key=value."
                )
            key, val = pair.split("=", 1)
            key = key.strip()
            if key not in _VALID_OVERRIDE_KEYS:
                raise ValueError(
                    f"Unknown parameter '{key}' in '{method_spec}'. "
                    f"Valid parameters: {', '.join(sorted(_VALID_OVERRIDE_KEYS))}"
                )
            overrides[key] = _coerce_value(key, val.strip())

    return MethodSpec(
        method=method,
        dims=dims,
        overrides=tuple(sorted(overrides.items())),
    )


def parse_methods_arg(raw: list[str]) -> list[MethodSpec]:
    """Parse repeatable -m arguments into a deduplicated MethodSpec list."""
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
    """Return a parameter suffix for projection name disambiguation."""
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
    """Run base.process_reduction with effective_params, restoring base.config after."""
    saved = base.config
    base.config = effective_params
    try:
        return base.process_reduction(data, method, dims)
    finally:
        base.config = saved


def _load_background_h5(path: Path) -> np.ndarray:
    """Load all embedding vectors from an HDF5 file as a (n, d) ndarray.

    Identifier ordering is not preserved — only the vector matrix is needed
    for ρPCA's covariance computation.
    """
    from protspace.data.loaders import load_h5

    emb_set = load_h5([path])
    arr = np.asarray(emb_set.data, dtype=np.float64)
    logger.info(
        "Loaded ρPCA background: %d samples × %d features from %s",
        arr.shape[0],
        arr.shape[1],
        path,
    )
    return arr


class ReductionPipeline:
    """Unified pipeline: load → annotate → reduce → output."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        reducer_dict = asdict(config.reducer_params)
        self.base = BaseProcessor(reducer_dict, get_reducers())
        # Lazy cache for external ρPCA backgrounds, keyed by absolute path.
        # This supports both global --ppca-background and per-method
        # background_path overrides in method specs.
        self._background_cache: dict[str, np.ndarray] = {}

    def run(self, embedding_sets: list[EmbeddingSet]) -> Path:
        if not embedding_sets:
            raise ValueError("At least one EmbeddingSet is required.")

        from protspace.data.loaders.embedding_set import merge_same_name_sets

        embedding_sets = merge_same_name_sets(embedding_sets)
        all_headers = self._validate_headers(embedding_sets)
        metadata = self._fetch_annotations(all_headers, embedding_sets)

        if self.config.no_scores:
            from protspace.data.annotations.scores import strip_scores_from_df

            metadata = strip_scores_from_df(metadata)

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

        all_reductions = self._run_reductions(embedding_sets)

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

        if (
            not self.config.keep_tmp
            and self.config.intermediate_dir
            and self.config.intermediate_dir.exists()
        ):
            shutil.rmtree(self.config.intermediate_dir)

        return self.config.output_path

    @staticmethod
    def _extract_sequences(embedding_sets: list[EmbeddingSet]) -> dict[str, str]:
        sequences = {}
        for emb_set in embedding_sets:
            if emb_set.fasta_path and Path(emb_set.fasta_path).exists():
                from protspace.data.io.fasta import parse_fasta
                from protspace.data.loaders.h5 import parse_identifier

                raw = parse_fasta(Path(emb_set.fasta_path))
                sequences.update({parse_identifier(h): s for h, s in raw.items()})
        return sequences

    def _validate_headers(self, embedding_sets: list[EmbeddingSet]) -> list[str]:
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

        for es in embedding_sets:
            diff = set(es.headers) - common
            if diff:
                logger.warning(
                    f"Embedding '{es.name}': dropping {len(diff)} proteins "
                    f"not present in all sets."
                )

        common_headers = [h for h in embedding_sets[0].headers if h in common]

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
        from protspace.data.annotations.manager import ProteinAnnotationManager

        sequences = self._extract_sequences(embedding_sets) if embedding_sets else {}

        annotation_names, csv_path = self._resolve_annotation_names()

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
                    sources = {src: src in refetch for src in _ANN_SOURCES}
                    refetched = [s for s in _ANN_SOURCES if sources[s]]
                    logger.info(f"--refetch: re-fetching {', '.join(refetched)}")
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
        if csv_df is None:
            return api_df

        merged = api_df.merge(
            csv_df.drop_duplicates("identifier"),
            on="identifier",
            how="left",
            suffixes=("_api", ""),
        )
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
        # Exclude transient state (the loaded background array) from the
        # cache key — the path is already in effective_params.
        params_for_key = {
            k: v for k, v in (effective_params or {}).items() if k != "background_data"
        } or asdict(self.config.reducer_params)
        key_dict = {
            "embedding": embedding_name,
            "method": method,
            "dims": dims,
            "params": params_for_key,
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

    # --- ρPCA background loading ---

    def _get_background_data(self, path_str: str) -> np.ndarray | None:
        """Load an external ρPCA background (cached by path)."""
        if not path_str:
            return None
        key = str(Path(path_str).expanduser().resolve())
        if key not in self._background_cache:
            self._background_cache[key] = _load_background_h5(Path(path_str))
        return self._background_cache[key]

    def _prepare_ppca_params(
        self, effective_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve ρPCA-specific knobs in the effective params.

        If background_path is set (either globally or as a per-method
        override), force strategy="external" and load the background.
        Otherwise leave the auto-split strategy in place and emit a warning
        on first use.
        """
        params = dict(effective_params)
        path_str = params.get("background_path", "") or ""
        if path_str:
            params["background_strategy"] = "external"
            params["background_data"] = self._get_background_data(path_str)
        elif params.get("background_strategy") not in (
            "random",
            "uniform",
            "outlier",
        ):
            # Should not happen via the CLI, but guard the case where a
            # user manually passes background_strategy=external without a
            # background_path.
            params["background_strategy"] = "outlier"

        # background_path is a path string, not a reducer param; drop it.
        params.pop("background_path", None)
        return params

    # --- Dimensionality reduction ---

    def _run_reductions(
        self, embedding_sets: list[EmbeddingSet]
    ) -> list[dict[str, Any]]:
        all_reductions = []
        cached_projections: list[str] = []
        computed_count = 0

        method_counts = Counter(
            (spec.method, spec.dims) for spec in self.config.methods
        )

        global_params = asdict(self.config.reducer_params)
        ppca_in_run = any(spec.method == PPCA_NAME for spec in self.config.methods)
        ppca_warned = False

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

                effective_params = {**global_params, **spec.overrides_dict}

                # ρPCA-specific resolution.
                if method == PPCA_NAME:
                    effective_params = self._prepare_ppca_params(effective_params)
                    has_external = effective_params.get("background_data") is not None
                    if not has_external and not ppca_warned:
                        logger.warning(
                            "ρPCA: no external background provided via "
                            "--ppca-background; falling back to auto-split "
                            "strategy '%s' (background_ratio=%.2f). Results "
                            "are usually inferior to a true contrastive "
                            "background. Provide a biologically meaningful "
                            "background dataset for best results.",
                            effective_params.get("background_strategy"),
                            effective_params.get("background_ratio", 0.0),
                        )
                        ppca_warned = True

                param_suffix = disambiguation_suffix(spec, method_counts)

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
                    self.base, effective_params, method, dims, emb_set.data
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

        # Silence the unused-variable warning; ppca_in_run is informational
        # and may be used in future logging.
        _ = ppca_in_run

        return all_reductions
