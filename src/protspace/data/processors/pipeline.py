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

_CACHE_EXCLUDED_PARAM_KEYS = frozenset({
    "background_data",
    "target_data",
    "background_details",
})

_MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "<n/a>", "false", "0", "no"}


@dataclass(frozen=True)
class ReducerParams:
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
    regularization_mu: float = 1e-3
    standard_scale: bool = True

    # ρPCA background modes:
    #   explicit   -> external HDF5 file via background_path
    #   annotation -> rows from the current dataset selected by annotation
    #   derived    -> nuisance-only sequence segments embedded from the current dataset
    ppca_mode: str = "explicit"
    background_path: str = ""
    background_annotation: str = ""
    background_values: tuple[str, ...] = ()

    # Derived nuisance-only background parameters.
    derived_segment_column: str = ""
    derived_start_column: str = ""
    derived_end_column: str = ""
    derived_fixed_start: int = 0
    derived_fixed_end: int = 0
    derived_min_length: int = 5
    derived_embedder: str = ""

    # Paired-delta background parameters. The background rows are signed
    # deviations ±scale·(full_embedding - mature_embedding). This captures
    # both common and heterogeneous full-vs-mature embedding shifts.
    paired_full_path: str = ""
    paired_mature_path: str = ""
    paired_delta_scale: float = 0.5


@dataclass(frozen=True)
class MethodSpec:
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
    methods: list[MethodSpec]
    output_path: Path
    bundled: bool = True
    keep_tmp: bool = False
    no_scores: bool = False
    refetch_stages: frozenset[str] = field(default_factory=frozenset)
    annotations: list[str] | None = None
    intermediate_dir: Path | None = None
    reducer_params: ReducerParams = field(default_factory=ReducerParams)


_VALID_OVERRIDE_KEYS = {f.name for f in fields(ReducerParams)}
_FIELD_TYPES = {f.name: f.type for f in fields(ReducerParams)}


def _coerce_value(key: str, raw: str) -> int | float | str | bool:
    expected = _FIELD_TYPES.get(key)
    if expected is bool:
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
    if ":" in method_spec:
        base, params_str = method_spec.split(":", 1)
    else:
        base, params_str = method_spec, ""

    method = "".join(filter(str.isalpha, base))
    dims_text = "".join(filter(str.isdigit, base))
    if not method or not dims_text:
        raise ValueError(
            f"Invalid method spec {method_spec!r}. Expected e.g. pca2, umap2, ppca2."
        )
    dims = int(dims_text)
    overrides: dict[str, int | float | str | bool] = {}
    if params_str:
        for pair in params_str.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"Invalid parameter format '{pair}' in '{method_spec}'. Expected key=value."
                )
            key, val = pair.split("=", 1)
            key = key.strip()
            if key not in _VALID_OVERRIDE_KEYS:
                raise ValueError(f"Unknown parameter '{key}' in '{method_spec}'.")
            overrides[key] = _coerce_value(key, val.strip())

    return MethodSpec(method=method, dims=dims, overrides=tuple(sorted(overrides.items())))


def parse_methods_arg(raw: list[str]) -> list[MethodSpec]:
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
    saved = base.config
    base.config = effective_params
    try:
        return base.process_reduction(data, method, dims)
    finally:
        base.config = saved


def _load_background_h5(path: Path) -> tuple[np.ndarray, list[str]]:
    from protspace.data.loaders import load_h5

    # Provide name_override to bypass the missing 'model_name' attribute check.
    emb_set = load_h5([path], name_override="background")
    arr = np.asarray(emb_set.data, dtype=np.float64)
    headers = list(emb_set.headers)
    logger.info("Loaded explicit ρPCA background: %d × %d from %s", *arr.shape, path)
    return arr, headers


def _file_fingerprint(path: Path) -> dict[str, str | int]:
    """Return a lightweight cache fingerprint for a file-backed input.

    This prevents stale cached projections when a background file is changed
    in place but retains the same path.
    """
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def _is_truthy_annotation_value(value: Any) -> bool:
    return _normalize_value(value).lower() not in _MISSING_STRINGS


def _safe_identifier(identifier: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(identifier))


def _write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n")
            # Wrap for readability. Embedders should not care.
            for i in range(0, len(sequence), 80):
                handle.write(sequence[i : i + 80] + "\n")


def _hash_records(records: list[tuple[str, str]], *, embedder: str, params: dict[str, Any]) -> str:
    payload = {
        "embedder": embedder,
        "params": params,
        "records": records,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


class ReductionPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        reducer_dict = asdict(config.reducer_params)
        self.base = BaseProcessor(reducer_dict, get_reducers())
        self._background_cache: dict[str, tuple[np.ndarray, list[str]]] = {}
        self._derived_background_cache: dict[str, tuple[np.ndarray, list[str], dict[str, Any]]] = {}
        self._paired_delta_background_cache: dict[str, tuple[np.ndarray, list[str], dict[str, Any]]] = {}
        self._similarity_matrix_cache: np.ndarray | None = None
        self._sequences: dict[str, str] = {}

    def run(self, embedding_sets: list[EmbeddingSet]) -> Path:
        if not embedding_sets:
            raise ValueError("At least one EmbeddingSet is required.")
        from protspace.data.loaders.embedding_set import merge_same_name_sets

        embedding_sets = merge_same_name_sets(embedding_sets)
        all_headers = self._validate_headers(embedding_sets)
        self._sequences = self._extract_sequences(embedding_sets)
        metadata = self._fetch_annotations(all_headers, embedding_sets)

        if self.config.no_scores:
            from protspace.data.annotations.scores import strip_scores_from_df

            metadata = strip_scores_from_df(metadata)

        full_metadata = pd.DataFrame({"identifier": all_headers})
        if len(metadata.columns) > 1:
            metadata = metadata.astype(str)
            if "identifier" not in metadata.columns:
                id_col = metadata.columns[0]
                metadata = metadata.rename(columns={id_col: "identifier"})
            full_metadata = full_metadata.merge(
                metadata.drop_duplicates("identifier"), on="identifier", how="left"
            )
        metadata = full_metadata

        all_reductions = self._run_reductions(embedding_sets, metadata)

        output = self.base.create_output(metadata, all_reductions, all_headers)
        self.base.save_output(output, self.config.output_path, bundled=self.config.bundled)

        logger.info(
            f"Processed {len(all_headers)} proteins, {len(embedding_sets)} embedding(s), "
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
        sequences: dict[str, str] = {}
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
            raise ValueError("No common protein identifiers found across embedding sets.")
        for es in embedding_sets:
            diff = set(es.headers) - common
            if diff:
                logger.warning(
                    f"Embedding '{es.name}': dropping {len(diff)} proteins not present in all sets."
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
        self, headers: list[str], embedding_sets: list[EmbeddingSet] | None = None
    ) -> pd.DataFrame:
        from protspace.data.annotations.manager import ProteinAnnotationManager

        sequences = self._sequences if embedding_sets else {}
        annotation_names, csv_paths = self._resolve_annotation_names()

        csv_df = None
        if csv_paths:
            frames = []
            for path in csv_paths:
                sep = "\t" if path.endswith(".tsv") else ","
                df_one = pd.read_csv(path, sep=sep)
                if "identifier" not in df_one.columns:
                    id_col = df_one.columns[0]
                    df_one = df_one.rename(columns={id_col: "identifier"})
                df_one = df_one.drop_duplicates(subset="identifier", keep="last")
                frames.append(df_one)
            csv_df = frames[0]
            for next_df in frames[1:]:
                overlap = (set(csv_df.columns) & set(next_df.columns)) - {"identifier"}
                if overlap:
                    csv_df = csv_df.drop(columns=list(overlap))
                csv_df = csv_df.merge(next_df, on="identifier", how="outer")

        if annotation_names:
            from protspace.data.annotations.configuration import AnnotationConfiguration

            annotations_list = AnnotationConfiguration(annotation_names).user_annotations
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
                    from protspace.data.annotations.configuration import ANNOTATION_GROUPS

                    required = set(ANNOTATION_GROUPS["default"])
                else:
                    required = set(annotations_list)

                missing = required - cached_annotations

                if not missing and not refetching_annotations:
                    if annotations_list:
                        cols = ["identifier"] + [f for f in annotations_list if f in cached_df.columns]
                        api_df = cached_df[cols]
                    else:
                        api_df = cached_df
                    return self._merge_csv(api_df, csv_df)

                from protspace.data.annotations.configuration import AnnotationConfiguration

                sources = AnnotationConfiguration.determine_sources_to_fetch(
                    cached_annotations, required
                )

                if refetching_annotations:
                    sources = {src: src in refetch for src in _ANN_SOURCES}
                    refetched = [s for s in _ANN_SOURCES if sources[s]]
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

    def _resolve_annotation_names(self) -> tuple[list[str], list[str]]:
        if not self.config.annotations:
            return [], []
        names, csv_paths = [], []
        for item in self.config.annotations:
            item = item.strip()
            if not item:
                continue
            if item.endswith((".csv", ".tsv")):
                csv_paths.append(item)
            else:
                for part in item.split(","):
                    part = part.strip()
                    if part:
                        names.append(part)
        return names, csv_paths

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
        params_for_key = {
            k: v
            for k, v in (effective_params or {}).items()
            if k not in _CACHE_EXCLUDED_PARAM_KEYS
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
        path = self._projection_cache_path(embedding_name, method, dims, effective_params)
        if path is None or not path.exists() or "projections" in self.config.refetch_stages:
            return None
        logger.info("Using cached %s %d projection for '%s'", method.upper(), dims, embedding_name)
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
        path = self._projection_cache_path(embedding_name, method, dims, effective_params)
        if path is None:
            return
        np.savez(path, data=reduction["data"], info=np.array(json.dumps(reduction["info"])))

    def _get_explicit_background(self, background_path: str) -> tuple[np.ndarray, list[str]]:
        if not background_path:
            raise ValueError(
                "ρPCA explicit mode requires --ppca-background <background.h5>."
            )
        if background_path not in self._background_cache:
            self._background_cache[background_path] = _load_background_h5(Path(background_path))
        return self._background_cache[background_path]

    @staticmethod
    def _aligned_annotations(headers: list[str], annotations_df: pd.DataFrame) -> pd.DataFrame:
        if "identifier" not in annotations_df.columns:
            raise ValueError("Annotation table must contain an 'identifier' column.")
        return annotations_df.drop_duplicates("identifier").set_index("identifier").reindex(headers)

    def _annotation_mask(
        self,
        headers: list[str],
        annotations_df: pd.DataFrame,
        column: str,
        values: tuple[str, ...],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not column:
            raise ValueError(
                "ρPCA annotation mode requires --ppca-background-annotation <column>."
            )
        aligned = self._aligned_annotations(headers, annotations_df)
        if column not in aligned.columns:
            raise ValueError(
                f"Annotation column {column!r} not found. Available columns: "
                f"{', '.join(map(str, aligned.columns))}"
            )

        series = aligned[column]
        if values:
            allowed = {str(v).strip().lower() for v in values}
            mask = series.map(lambda v: _normalize_value(v).lower() in allowed).to_numpy(dtype=bool)
            selector = {"column": column, "values": list(values), "rule": "isin"}
        else:
            mask = series.map(_is_truthy_annotation_value).to_numpy(dtype=bool)
            selector = {"column": column, "values": [], "rule": "truthy_non_missing"}
        return mask, selector

    def _build_annotation_background(
        self,
        emb_set: EmbeddingSet,
        annotations_df: pd.DataFrame,
        params: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        column = str(params.get("background_annotation", ""))
        values = tuple(params.get("background_values", ()) or ())
        mask, selector = self._annotation_mask(emb_set.headers, annotations_df, column, values)
        n_selected = int(mask.sum())
        if n_selected < 2:
            raise ValueError(
                f"ρPCA annotation background selected {n_selected} sample(s); need at least 2. "
                "Check --ppca-background-annotation/--ppca-background-values."
            )
        X_B = np.asarray(emb_set.data[mask], dtype=np.float64)
        details = {
            "mode": "annotation",
            "selector": selector,
            "n_background": int(X_B.shape[0]),
            "n_target": int(emb_set.data.shape[0]),
            "note": "Background rows are selected from the current displayed dataset; target remains the full input.",
        }
        return X_B, details

    def _sequence_for_identifier(self, identifier: str, row: pd.Series) -> str:
        if identifier in self._sequences:
            return self._sequences[identifier]
        if "sequence" in row.index and _is_truthy_annotation_value(row["sequence"]):
            return _normalize_value(row["sequence"])
        return ""

    def _derived_records_from_annotations(
        self,
        emb_set: EmbeddingSet,
        annotations_df: pd.DataFrame,
        params: dict[str, Any],
    ) -> tuple[list[tuple[str, str]], dict[str, Any]]:
        aligned = self._aligned_annotations(emb_set.headers, annotations_df)
        headers = list(emb_set.headers)

        # Optional row selector. This is useful for cases where the CSV contains
        # both SP-positive and SP-negative proteins and derived segments should
        # only be made for SP-positive proteins.
        selector_details: dict[str, Any] | None = None
        candidate_mask = np.ones(len(headers), dtype=bool)
        if str(params.get("background_annotation", "")):
            candidate_mask, selector_details = self._annotation_mask(
                headers,
                annotations_df,
                str(params.get("background_annotation", "")),
                tuple(params.get("background_values", ()) or ()),
            )

        segment_col = str(params.get("derived_segment_column", ""))
        start_col = str(params.get("derived_start_column", ""))
        end_col = str(params.get("derived_end_column", ""))
        fixed_start = int(params.get("derived_fixed_start", 0))
        fixed_end = int(params.get("derived_fixed_end", 0))
        min_len = int(params.get("derived_min_length", 5))

        if segment_col and segment_col not in aligned.columns:
            raise ValueError(f"Derived segment column {segment_col!r} not found in annotations.")
        if start_col and start_col not in aligned.columns:
            raise ValueError(f"Derived start column {start_col!r} not found in annotations.")
        if end_col and end_col not in aligned.columns:
            raise ValueError(f"Derived end column {end_col!r} not found in annotations.")
        if not segment_col and not end_col and fixed_end <= fixed_start:
            raise ValueError(
                "ρPCA derived mode needs either --ppca-derived-segment-column, "
                "--ppca-derived-end-column, or --ppca-derived-fixed-end greater than "
                "--ppca-derived-fixed-start."
            )

        records: list[tuple[str, str]] = []
        skipped_missing_sequence = 0
        skipped_short = 0
        skipped_bad_coordinates = 0

        for i, identifier in enumerate(headers):
            if not candidate_mask[i]:
                continue
            row = aligned.iloc[i]
            segment = ""
            if segment_col:
                segment = _normalize_value(row.get(segment_col, ""))
            else:
                sequence = self._sequence_for_identifier(identifier, row)
                if not sequence:
                    skipped_missing_sequence += 1
                    continue
                try:
                    start = int(float(row.get(start_col))) if start_col else fixed_start
                    end = int(float(row.get(end_col))) if end_col else fixed_end
                except (TypeError, ValueError):
                    skipped_bad_coordinates += 1
                    continue
                if start < 0 or end <= start or start >= len(sequence):
                    skipped_bad_coordinates += 1
                    continue
                end = min(end, len(sequence))
                segment = sequence[start:end]

            segment = "".join(str(segment).split()).upper()
            if len(segment) < min_len:
                skipped_short += 1
                continue
            records.append((f"{_safe_identifier(identifier)}__derived_nuisance", segment))

        if len(records) < 2:
            raise ValueError(
                f"ρPCA derived mode produced only {len(records)} usable nuisance segment(s); "
                "need at least 2. Provide a segment column, valid start/end columns, "
                "or a FASTA via -f/--fasta, and check --ppca-derived-min-length."
            )

        details = {
            "mode": "derived",
            "selector": selector_details,
            "segment_column": segment_col,
            "start_column": start_col,
            "end_column": end_col,
            "fixed_start": fixed_start,
            "fixed_end": fixed_end,
            "min_length": min_len,
            "n_derived_segments": len(records),
            "skipped_missing_sequence": skipped_missing_sequence,
            "skipped_bad_coordinates": skipped_bad_coordinates,
            "skipped_short": skipped_short,
            "note": "Background is made from nuisance-only sequence segments and embedded separately; target embeddings are not modified.",
        }
        return records, details

    def _build_derived_background(
        self,
        emb_set: EmbeddingSet,
        annotations_df: pd.DataFrame,
        params: dict[str, Any],
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        records, details = self._derived_records_from_annotations(emb_set, annotations_df, params)
        embedder = str(params.get("derived_embedder", "") or emb_set.name)
        if not embedder:
            raise ValueError(
                "ρPCA derived mode could not infer an embedder. Use --ppca-derived-embedder."
            )

        hash_params = {
            "derived_segment_column": params.get("derived_segment_column", ""),
            "derived_start_column": params.get("derived_start_column", ""),
            "derived_end_column": params.get("derived_end_column", ""),
            "derived_fixed_start": params.get("derived_fixed_start", 0),
            "derived_fixed_end": params.get("derived_fixed_end", 0),
            "derived_min_length": params.get("derived_min_length", 5),
            "background_annotation": params.get("background_annotation", ""),
            "background_values": params.get("background_values", ()),
        }
        cache_key = _hash_records(records, embedder=embedder, params=hash_params)
        if cache_key in self._derived_background_cache:
            return self._derived_background_cache[cache_key]

        cache_dir = self.config.intermediate_dir
        if cache_dir is None:
            # Use output sibling when the user disabled keep_tmp. It will be removed
            # by the final cleanup path if keep_tmp is false.
            cache_dir = self.config.output_path.parent / "tmp"
        cache_dir.mkdir(parents=True, exist_ok=True)

        fasta_path = cache_dir / f"ppca_derived_{emb_set.name}_{cache_key}.fasta"
        h5_path = cache_dir / f"ppca_derived_{emb_set.name}_{cache_key}.h5"
        _write_fasta(records, fasta_path)

        from protspace.data.loaders.fasta import embed_fasta

        try:
            derived_set = embed_fasta(
                fasta_path,
                embedder,
                embed_config=None,
                embedding_cache=h5_path,
            )
        except Exception as exc:  # noqa: BLE001 - provide domain-specific context.
            raise ValueError(
                f"Failed to embed derived nuisance-only background with embedder {embedder!r}. "
                "For HDF5 inputs, make sure the embedding set name or "
                "--ppca-derived-embedder is a supported ProtSpace embedder, and "
                "that it matches the target embedding dimensionality."
            ) from exc

        X_B = np.asarray(derived_set.data, dtype=np.float64)
        if X_B.shape[1] != emb_set.data.shape[1]:
            raise ValueError(
                f"Derived background embedder {embedder!r} produced {X_B.shape[1]} features, "
                f"but target embedding '{emb_set.name}' has {emb_set.data.shape[1]}. "
                "Use the same model for target and derived background."
            )
        derived_headers = list(derived_set.headers)
        details = {
            **details,
            "embedder": embedder,
            "fasta_path": str(fasta_path),
            "embedding_cache": str(h5_path),
            "cache_key": cache_key,
            "n_background": int(X_B.shape[0]),
            "n_target": int(emb_set.data.shape[0]),
        }
        result = (X_B, derived_headers, details)
        self._derived_background_cache[cache_key] = result
        return result


    def _build_paired_delta_background(
        self,
        emb_set: EmbeddingSet,
        annotations_df: pd.DataFrame,
        params: dict[str, Any],
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        """Build a signed paired-delta background for ρPCA.

        For each selected protein with both embeddings available, compute

            delta_i = E(full_i) - E(mature_i)

        and create two background rows

            +scale * delta_i,  -scale * delta_i.

        The signed construction is important: ordinary covariance centers the
        background. If all signal peptides shift embeddings in nearly the same
        direction, using raw deltas alone can center away that common shift.
        Signed deltas keep the common full-vs-mature shift in the covariance
        through delta_i delta_i^T.
        """
        full_path = str(params.get("paired_full_path", ""))
        mature_path = str(params.get("paired_mature_path", ""))
        if not full_path or not mature_path:
            raise ValueError(
                "ρPCA paired_delta mode requires --ppca-paired-full <full.h5> "
                "and --ppca-paired-mature <mature.h5>."
            )

        scale = float(params.get("paired_delta_scale", 0.5))
        if scale <= 0.0:
            raise ValueError("--ppca-paired-delta-scale must be > 0 for paired_delta mode.")

        cache_payload = {
            "mode": "paired_delta",
            "full": _file_fingerprint(Path(full_path)),
            "mature": _file_fingerprint(Path(mature_path)),
            "scale": scale,
            "selector_column": str(params.get("background_annotation", "")),
            "selector_values": list(tuple(params.get("background_values", ()) or ())),
            "target_headers_hash": hashlib.sha256(
                json.dumps(list(emb_set.headers), sort_keys=False).encode()
            ).hexdigest()[:16],
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        if cache_key in self._paired_delta_background_cache:
            return self._paired_delta_background_cache[cache_key]

        full_data, full_headers = self._get_explicit_background(full_path)
        mature_data, mature_headers = self._get_explicit_background(mature_path)

        if full_data.shape[1] != emb_set.data.shape[1]:
            raise ValueError(
                f"Paired full embeddings have {full_data.shape[1]} features, but "
                f"target embedding '{emb_set.name}' has {emb_set.data.shape[1]}. "
                "Use the same embedding model for target, full, and mature inputs."
            )
        if mature_data.shape[1] != emb_set.data.shape[1]:
            raise ValueError(
                f"Paired mature embeddings have {mature_data.shape[1]} features, but "
                f"target embedding '{emb_set.name}' has {emb_set.data.shape[1]}. "
                "Use the same embedding model for target, full, and mature inputs."
            )

        full_index = {h: i for i, h in enumerate(full_headers)}
        mature_index = {h: i for i, h in enumerate(mature_headers)}

        selector_details: dict[str, Any] | None = None
        candidate_mask = np.ones(len(emb_set.headers), dtype=bool)
        if str(params.get("background_annotation", "")):
            candidate_mask, selector_details = self._annotation_mask(
                list(emb_set.headers),
                annotations_df,
                str(params.get("background_annotation", "")),
                tuple(params.get("background_values", ()) or ()),
            )

        selected_headers: list[str] = []
        missing_full = 0
        missing_mature = 0
        for i, header in enumerate(emb_set.headers):
            if not candidate_mask[i]:
                continue
            has_full = header in full_index
            has_mature = header in mature_index
            if has_full and has_mature:
                selected_headers.append(header)
            else:
                if not has_full:
                    missing_full += 1
                if not has_mature:
                    missing_mature += 1

        if len(selected_headers) < 2:
            raise ValueError(
                f"ρPCA paired_delta mode found only {len(selected_headers)} usable "
                "full/mature pair(s); need at least 2. Check paired HDF5 identifiers "
                "and the optional --ppca-background-annotation selector."
            )

        deltas = np.vstack([
            full_data[full_index[h]] - mature_data[mature_index[h]]
            for h in selected_headers
        ]).astype(np.float64)

        # Signed rows preserve the common shift direction under covariance
        # centering. We explicitly recenter to remove tiny floating-point drift.
        X_B = np.vstack([scale * deltas, -scale * deltas]).astype(np.float64)
        X_B = X_B - X_B.mean(axis=0, keepdims=True)
        background_headers = (
            [f"{_safe_identifier(h)}__paired_delta_pos" for h in selected_headers]
            + [f"{_safe_identifier(h)}__paired_delta_neg" for h in selected_headers]
        )

        details = {
            "mode": "paired_delta",
            "selector": selector_details,
            "full_path": full_path,
            "mature_path": mature_path,
            "delta_scale": scale,
            "n_pairs": len(selected_headers),
            "n_background": int(X_B.shape[0]),
            "n_target": int(emb_set.data.shape[0]),
            "missing_full": missing_full,
            "missing_mature": missing_mature,
            "cache_key": cache_key,
            "note": (
                "Background rows are signed paired deltas: ±scale·(E(full)-E(mature)). "
                "This models the signal-peptide-induced embedding shift while "
                "controlling for each protein's mature-domain biology."
            ),
        }
        result = (X_B, background_headers, details)
        self._paired_delta_background_cache[cache_key] = result
        return result

    def _prepare_ppca_inputs(
        self,
        emb_set: EmbeddingSet,
        annotations_df: pd.DataFrame,
        effective_params: dict[str, Any],
    ) -> None:
        mode = str(effective_params.get("ppca_mode", "explicit"))
        if mode not in {"explicit", "annotation", "derived", "paired_delta"}:
            raise ValueError(
                f"Unknown ppca_mode={mode!r}. Expected explicit, annotation, derived, or paired_delta."
            )

        if mode == "explicit":
            background_path = str(effective_params.get("background_path", ""))
            background_data, background_headers = self._get_explicit_background(background_path)
            if background_data.shape[1] != emb_set.data.shape[1]:
                raise ValueError(
                    f"ρPCA background '{background_path}' has {background_data.shape[1]} features, "
                    f"but target embedding '{emb_set.name}' has {emb_set.data.shape[1]}. "
                    "Use the same embedding model for target and background."
                )
            details = {
                "mode": "explicit",
                "path": background_path,
                "n_background": int(background_data.shape[0]),
                "n_target": int(emb_set.data.shape[0]),
                "n_background_headers": len(background_headers),
            }
            effective_params["background_data"] = background_data
            effective_params["background_source"] = "explicit"
            effective_params["background_n_samples"] = int(background_data.shape[0])
            effective_params["background_details"] = details
            effective_params["background_fingerprint"] = _file_fingerprint(Path(background_path))
            return

        if mode == "annotation":
            background_data, details = self._build_annotation_background(
                emb_set, annotations_df, effective_params
            )
            effective_params["background_data"] = background_data
            effective_params["background_source"] = "annotation"
            effective_params["background_n_samples"] = int(background_data.shape[0])
            effective_params["background_details"] = details
            effective_params["background_fingerprint"] = {
                "mode": "annotation",
                "embedding": emb_set.name,
                "selector": details["selector"],
                "n_background": int(background_data.shape[0]),
                "annotation_hash": hashlib.sha256(
                    pd.util.hash_pandas_object(
                        self._aligned_annotations(emb_set.headers, annotations_df),
                        index=True,
                    ).values.tobytes()
                ).hexdigest()[:16],
            }
            return

        if mode == "paired_delta":
            background_data, background_headers, details = self._build_paired_delta_background(
                emb_set, annotations_df, effective_params
            )
            effective_params["background_data"] = background_data
            effective_params["background_source"] = "paired_delta"
            effective_params["background_n_samples"] = int(background_data.shape[0])
            effective_params["background_details"] = details
            effective_params["background_fingerprint"] = {
                "mode": "paired_delta",
                "embedding": emb_set.name,
                "cache_key": details.get("cache_key"),
                "n_pairs": details.get("n_pairs"),
                "n_background": int(background_data.shape[0]),
                "n_background_headers": len(background_headers),
                "full_path": details.get("full_path"),
                "mature_path": details.get("mature_path"),
                "delta_scale": details.get("delta_scale"),
            }
            return

        # mode == derived
        background_data, background_headers, details = self._build_derived_background(
            emb_set, annotations_df, effective_params
        )
        effective_params["background_data"] = background_data
        effective_params["background_source"] = "derived"
        effective_params["background_n_samples"] = int(background_data.shape[0])
        effective_params["background_details"] = details
        effective_params["background_fingerprint"] = {
            "mode": "derived",
            "embedding": emb_set.name,
            "cache_key": details.get("cache_key"),
            "n_background": int(background_data.shape[0]),
            "n_background_headers": len(background_headers),
            "embedding_cache": details.get("embedding_cache"),
        }

    def _run_reductions(
        self, embedding_sets: list[EmbeddingSet], annotations_df: pd.DataFrame
    ) -> list[dict[str, Any]]:
        all_reductions = []
        cached_projections: list[str] = []
        computed_count = 0
        method_counts = Counter((spec.method, spec.dims) for spec in self.config.methods)
        global_params = asdict(self.config.reducer_params)

        for emb_set in embedding_sets:
            if emb_set.precomputed:
                self._similarity_matrix_cache = emb_set.data
                cached = self._load_cached_projection(emb_set.name, MDS_NAME, 2, global_params)
                if cached:
                    all_reductions.append(cached)
                    cached_projections.append(f"MDS 2 ({emb_set.name})")
                    continue
                effective_params = {**global_params, "precomputed": True}
                reduction = _run_with_overridden_config(
                    self.base, effective_params, MDS_NAME, 2, emb_set.data
                )
                reduction["name"] = format_projection_name(emb_set.name, MDS_NAME, 2)
                all_reductions.append(reduction)
                self._save_projection_cache(emb_set.name, MDS_NAME, 2, reduction, global_params)
                computed_count += 1
                continue

            for spec in self.config.methods:
                method, dims = spec.method, spec.dims
                if method not in self.base.reducers:
                    continue

                effective_params = {**global_params, **spec.overrides_dict}

                if method == PPCA_NAME:
                    self._prepare_ppca_inputs(emb_set, annotations_df, effective_params)

                param_suffix = disambiguation_suffix(spec, method_counts)
                cached = self._load_cached_projection(
                    emb_set.name, method, dims, effective_params, param_suffix
                )
                if cached:
                    all_reductions.append(cached)
                    cached_projections.append(f"{method.upper()} {dims} ({emb_set.name})")
                    continue

                logger.info(f"Applying {method.upper()} {dims} to '{emb_set.name}'")
                reduction = _run_with_overridden_config(
                    self.base, effective_params, method, dims, emb_set.data
                )
                reduction["name"] = format_projection_name(
                    emb_set.name, method, dims, param_suffix
                )
                all_reductions.append(reduction)
                self._save_projection_cache(emb_set.name, method, dims, reduction, effective_params)
                computed_count += 1

        if cached_projections:
            logger.warning(
                "Using %d cached projection%s",
                len(cached_projections),
                "s" if len(cached_projections) != 1 else "",
            )

        return all_reductions