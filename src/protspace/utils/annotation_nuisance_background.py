"""Build annotation-defined nuisance-effect backgrounds for ProtSpace ρPCA.

Library module. Given target embeddings, an aligned annotation table, and one or
more user-declared nuisance specs, it estimates the embedding-space component
predictable from each nuisance and assembles a signed ρPCA background. The single
public entry point is :func:`build_annotation_nuisance_background`, which returns a
:class:`NuisanceBackgroundResult` (call ``.write(out_dir)`` to emit ``background.h5``
plus diagnostics). The ``protspace prepare --nuisance`` pipeline drives it; there is
no standalone CLI in this module.

Core model for each nuisance k:

    X_c ≈ Φ_k W_k
    G_k = Φ_k W_k
    B_k = [ +α_k G_k ; -α_k G_k ]

where X is n × d target embeddings, Φ_k is a typed annotation design matrix
(``type = auto | continuous | binary | categorical | multilabel``), G_k is the
predicted embedding-shift matrix, and B_k is the signed ρPCA background block.

Any annotation column can be declared a nuisance; the module does not judge whether
that is biologically sensible, but it records diagnostics and warnings for
instability, sparsity, missingness, high cardinality, and overfitting risk.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

try:
    from scipy import sparse
except Exception:  # pragma: no cover - scipy should exist in ProtSpace env
    sparse = None  # type: ignore[assignment]


LOGGER = logging.getLogger("annotation_nuisance_background")
SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
MISSING_STRINGS = {"", "nan", "none", "null", "na", "n/a", "<n/a>"}
FALSE_STRINGS = {"false", "0", "no", "n", "absent", "negative", "off"}
TRUE_STRINGS = {"true", "1", "yes", "y", "present", "positive", "on"}

# Candidate grid for auto-tuning ridge_alpha (annotation→embedding model) via GCV.
RIDGE_ALPHA_GRID = np.logspace(-2, 4, 13)  # 0.01 … 10000, spans the old fixed 10.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NuisanceSpec:
    """Parsed user declaration for one nuisance annotation."""

    raw: str
    name: str
    column: str
    nuisance_type: str = "auto"
    scale: float = 1.0
    ridge_alpha: float | str = "auto"  # "auto" → CV-GCV tuned; a float fixes it
    cross_fit: int | None = None
    random_state: int = 42

    # Continuous options ("auto" → CV-selected)
    transform: str = "auto"
    basis: str = "spline"
    n_knots: int | str = "auto"
    degree: int = 3

    # Categorical / multilabel options
    min_count: int = 5
    max_features: int = 5000
    rare_policy: str = "rare"  # rare or drop
    rare_value: str = "__rare__"
    missing: str = "auto"  # auto, drop, category, zero, median
    missing_value: str = "__missing__"
    sep: str = ";"
    strip_scores: bool = True
    strip_evidence: bool = True
    token_mode: str = "raw"  # raw, accession, name

    # Conditional / residual options
    condition_on: str = ""
    condition_type: str = "auto"
    condition_missing: str = "category"
    condition_min_count: int = 5
    condition_max_features: int = 5000

    # Misc
    fit_intercept: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class DesignMatrix:
    """Annotation encoding for the subset of rows used by one effect model."""

    name: str
    column: str
    annotation_type: str
    row_mask: np.ndarray
    matrix: Any  # np.ndarray or scipy sparse matrix
    feature_names: list[str]
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.matrix.shape[1])


@dataclass
class EffectBlock:
    """Embedding-space nuisance-effect block before signing."""

    name: str
    column: str
    effect_type: str
    ids: list[str]
    effects: np.ndarray
    scale: float
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.effects = np.asarray(self.effects, dtype=np.float64)
        if self.effects.ndim != 2:
            raise ValueError(f"EffectBlock {self.name!r}: effects must be 2D, got {self.effects.shape}")
        if len(self.ids) != self.effects.shape[0]:
            raise ValueError(
                f"EffectBlock {self.name!r}: ids length {len(self.ids)} != rows {self.effects.shape[0]}"
            )
        if self.effects.shape[0] < 1:
            raise ValueError(f"EffectBlock {self.name!r}: no rows")
        if not np.isfinite(self.effects).all():
            raise ValueError(f"EffectBlock {self.name!r}: effects contain NaN or inf")




@dataclass
class NuisanceBackgroundResult:
    """In-memory result for an annotation-defined ρPCA background."""

    background: np.ndarray
    background_ids: list[str]
    manifest: dict[str, Any]
    diagnostics_md: str
    effect_summary: pd.DataFrame
    block_details: list[dict[str, Any]]
    annotations_aligned: pd.DataFrame

    def write(self, out_dir: str | Path, *, background_name: str = "background.h5") -> Path:
        """Write background.h5 plus diagnostics/manifest files to a directory."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        bg_path = out_path / background_name
        write_h5_matrix(bg_path, self.background_ids, self.background, model_name="background")

        manifest = dict(self.manifest)
        manifest["background_h5"] = str(bg_path)
        manifest["recommended_protspace_command"] = (
            "protspace prepare -i TARGET.h5:MODEL -m pca2,umap2,rhopca2 "
            f"--rhopca-background {bg_path} --no-standard-scale "
            f"--regularization-mu {manifest.get('regularization_mu', 1e-6)} -o OUT"
        )
        diagnostics_md = render_diagnostics(manifest)

        write_json(out_path / "background.manifest.json", manifest)
        write_text(out_path / "diagnostics.md", diagnostics_md)
        self.effect_summary.to_csv(out_path / "effect_summary.csv", index=False)
        self.annotations_aligned.to_csv(out_path / "annotations_aligned.csv", index=False)
        return bg_path


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def safe_id(x: Any) -> str:
    s = SAFE_RE.sub("_", str(x))
    return s.strip("._-") or "id"


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def normalize_string(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def is_missing_value(value: Any) -> bool:
    return normalize_string(value).lower() in MISSING_STRINGS


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUE_STRINGS:
        return True
    if text in FALSE_STRINGS:
        return False
    return default


def decode_sep(value: str | None, default: str = ";") -> str:
    if value is None or value == "":
        return default
    text = str(value).strip().strip("'\"")
    named = {
        "semicolon": ";",
        "semi": ";",
        "sc": ";",
        "pipe": "|",
        "bar": "|",
        "comma": ",",
        "tab": "\t",
        "space": " ",
    }
    return named.get(text.lower(), text)


def split_unquoted(text: str, sep: str = ";") -> list[str]:
    """Split text on sep while ignoring separators inside quotes."""
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escape = False
    for ch in text:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            buf.append(ch)
            escape = True
            continue
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            buf.append(ch)
            continue
        if ch == sep:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf).strip())
    return parts


def strip_quotes(value: str) -> str:
    s = value.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        return s[1:-1]
    return s


def coerce_number_or_bool(value: str) -> Any:
    v = strip_quotes(value)
    low = v.lower()
    if low in TRUE_STRINGS:
        return True
    if low in FALSE_STRINGS:
        return False
    if low in {"none", "null"}:
        return None
    try:
        if re.match(r"^[+-]?\d+$", v):
            return int(v)
        if re.match(r"^[+-]?(\d+\.\d*|\d*\.\d+|\d+)([eE][+-]?\d+)?$", v):
            return float(v)
    except Exception:
        pass
    return v


# ---------------------------------------------------------------------------
# HDF5 background output
# ---------------------------------------------------------------------------


def write_h5_matrix(path: str | Path, ids: Sequence[str], X: np.ndarray, *, model_name: str | None = None) -> None:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {X.shape}")
    if len(ids) != X.shape[0]:
        raise ValueError(f"ids length {len(ids)} does not match matrix rows {X.shape[0]}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    with h5py.File(path, "w") as h5:
        if model_name:
            h5.attrs["model_name"] = str(model_name)
        h5.attrs["background_kind"] = "annotation_nuisance_effect"
        for identifier, vec in zip(ids, X, strict=False):
            sid = safe_id(identifier)
            if sid in seen:
                seen[sid] += 1
                sid = f"{sid}__dup{seen[sid]}"
            else:
                seen[sid] = 0
            h5.create_dataset(sid, data=np.asarray(vec, dtype=np.float32))


# ---------------------------------------------------------------------------
# Annotation alignment
# ---------------------------------------------------------------------------


def align_annotations(ids: Sequence[str], annotations: pd.DataFrame) -> pd.DataFrame:
    if "identifier" not in annotations.columns:
        raise ValueError("Annotation table must contain an 'identifier' column")
    aligned = annotations.drop_duplicates("identifier", keep="last").set_index("identifier").reindex(list(ids))
    aligned.index.name = "identifier"
    return aligned.reset_index()


# ---------------------------------------------------------------------------
# Nuisance spec parsing
# ---------------------------------------------------------------------------


def _resolve_alpha(value: Any) -> float | str:
    """Resolve a ridge_alpha value: None/'auto' → 'auto' (CV-tuned), else a float."""
    if value is None:
        return "auto"
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return float(value)


def _resolve_n_knots(value: Any) -> int | str:
    if value is None:
        return "auto"
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return int(value)


def parse_nuisance_spec(raw: str, *, global_cross_fit: int, global_ridge_alpha: float | str | None, random_state: int) -> NuisanceSpec:
    raw = raw.strip()
    if not raw:
        raise ValueError("Empty --nuisance spec")
    if ":" in raw:
        column_part, options_part = raw.split(":", 1)
    else:
        column_part, options_part = raw, ""

    opts: dict[str, Any] = {}
    if options_part:
        for part in split_unquoted(options_part, sep=";"):
            if not part:
                continue
            if "=" not in part:
                raise ValueError(
                    f"Invalid nuisance option {part!r} in {raw!r}; expected key=value"
                )
            key, val = part.split("=", 1)
            opts[key.strip().replace("-", "_")] = coerce_number_or_bool(val.strip())

    # Allow --nuisance "column=length;type=continuous" too.
    column = str(opts.get("column") or column_part.strip())
    if column.startswith("column="):
        _, val = column.split("=", 1)
        column = strip_quotes(val)
    column = strip_quotes(column)
    name = str(opts.get("name") or column)

    spec = NuisanceSpec(
        raw=raw,
        name=safe_id(name),
        column=column,
        nuisance_type=str(opts.get("type", opts.get("nuisance_type", "auto"))).lower(),
        scale=float(opts.get("scale", 1.0)),
        ridge_alpha=_resolve_alpha(opts.get("ridge_alpha", global_ridge_alpha)),
        cross_fit=int(opts["cross_fit"]) if "cross_fit" in opts and opts["cross_fit"] is not None else global_cross_fit,
        random_state=int(opts.get("random_state", random_state)),
        transform=str(opts.get("transform", "auto")).lower(),
        basis=str(opts.get("basis", opts.get("model", "spline"))).lower(),
        n_knots=_resolve_n_knots(opts.get("n_knots", "auto")),
        degree=int(opts.get("degree", 3)),
        min_count=int(opts.get("min_count", 5)),
        max_features=int(opts.get("max_features", 5000)),
        rare_policy=str(opts.get("rare_policy", "rare")).lower(),
        rare_value=str(opts.get("rare_value", "__rare__")),
        missing=str(opts.get("missing", "auto")).lower(),
        missing_value=str(opts.get("missing_value", "__missing__")),
        sep=decode_sep(str(opts.get("sep")) if "sep" in opts else None, default=";"),
        strip_scores=parse_bool(opts.get("strip_scores"), default=True),
        strip_evidence=parse_bool(opts.get("strip_evidence"), default=True),
        token_mode=str(opts.get("token_mode", "raw")).lower(),
        condition_on=str(opts.get("condition_on", opts.get("condition", "")) or ""),
        condition_type=str(opts.get("condition_type", "auto")).lower(),
        condition_missing=str(opts.get("condition_missing", "category")).lower(),
        condition_min_count=int(opts.get("condition_min_count", opts.get("min_condition_count", 5))),
        condition_max_features=int(opts.get("condition_max_features", opts.get("max_condition_features", 5000))),
        fit_intercept=parse_bool(opts.get("fit_intercept"), default=True),
        options=opts,
    )
    if spec.nuisance_type not in {"auto", "continuous", "binary", "categorical", "multilabel"}:
        raise ValueError(
            f"Nuisance {spec.name!r}: unknown type {spec.nuisance_type!r}. "
            "Use auto, continuous, binary, categorical, or multilabel."
        )
    if spec.rare_policy not in {"rare", "drop"}:
        raise ValueError(f"Nuisance {spec.name!r}: rare_policy must be rare or drop")
    return spec


# ---------------------------------------------------------------------------
# Annotation type inference and token parsing
# ---------------------------------------------------------------------------


def infer_annotation_type(series: pd.Series) -> tuple[str, dict[str, Any], list[str]]:
    s = series.copy()
    nonmissing = s[~s.map(is_missing_value)]
    warnings: list[str] = []
    details: dict[str, Any] = {
        "n_nonmissing_for_inference": int(len(nonmissing)),
        "n_total_for_inference": int(len(s)),
    }
    if len(nonmissing) == 0:
        warnings.append("all values are missing; defaulting type to categorical")
        return "categorical", details, warnings

    # Multi-label if a substantial fraction contains semicolon-separated tokens.
    text = nonmissing.astype(str)
    semicolon_frac = float(text.str.contains(";", regex=False).mean())
    details["semicolon_fraction"] = semicolon_frac
    if semicolon_frac >= 0.05:
        return "multilabel", details, warnings

    # Binary truthy/falsey or only two unique labels.
    low_values = {str(v).strip().lower() for v in nonmissing.unique()}
    details["n_unique_nonmissing"] = int(len(low_values))
    if low_values and low_values <= (TRUE_STRINGS | FALSE_STRINGS | MISSING_STRINGS):
        return "binary", details, warnings
    if len(low_values) == 2:
        # Could be categorical, but binary is the safer compact encoding.
        return "binary", details, warnings

    numeric = pd.to_numeric(nonmissing, errors="coerce")
    numeric_frac = float(numeric.notna().mean()) if len(nonmissing) else 0.0
    details["numeric_fraction"] = numeric_frac
    if numeric_frac >= 0.90 and numeric.nunique(dropna=True) > 10:
        return "continuous", details, warnings
    if numeric_frac >= 0.90 and numeric.nunique(dropna=True) <= 10:
        # numeric but few levels is better treated as categorical/binary.
        return "categorical", details, warnings

    return "categorical", details, warnings


def parse_token_string(
    value: Any,
    *,
    sep: str,
    strip_scores: bool,
    strip_evidence: bool,
    token_mode: str,
) -> list[str]:
    text = normalize_string(value)
    if not text:
        return []
    if sep:
        raw_tokens = [t.strip() for t in text.split(sep)]
    else:
        raw_tokens = [text]
    out: list[str] = []
    for tok in raw_tokens:
        if not tok:
            continue
        if strip_scores or strip_evidence:
            # ProtSpace annotations often append evidence/scores as '|EXP' or '|42.1'.
            tok = tok.split("|", 1)[0].strip()
        if token_mode == "accession":
            # Keep first accession-like token before whitespace or '('.
            tok = tok.split("(", 1)[0].strip().split()[0] if tok.split() else tok
        elif token_mode == "name":
            # Prefer text in parentheses when available: PF00001 (Name) -> Name.
            m = re.search(r"\((.*?)\)", tok)
            if m:
                tok = m.group(1).strip()
        elif token_mode != "raw":
            raise ValueError(f"Unknown token_mode={token_mode!r}; use raw, accession, or name")
        if tok:
            out.append(tok)
    # deterministic de-duplication preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for tok in out:
        if tok not in seen:
            seen.add(tok)
            uniq.append(tok)
    return uniq


# ---------------------------------------------------------------------------
# Design matrix builders
# ---------------------------------------------------------------------------


def ensure_sparse_available() -> None:
    if sparse is None:
        raise RuntimeError("scipy is required for sparse categorical/multilabel designs")


def _default_missing_policy(annotation_type: str, explicit: str) -> str:
    if explicit and explicit != "auto":
        return explicit
    if annotation_type == "continuous":
        return "drop"
    if annotation_type == "binary":
        return "category"
    if annotation_type == "categorical":
        return "category"
    if annotation_type == "multilabel":
        return "empty"
    return "category"


def _standardize_dense(H: np.ndarray) -> np.ndarray:
    if H.size == 0:
        return H
    return StandardScaler().fit_transform(H)


def build_continuous_design(series: pd.Series, spec: NuisanceSpec, *, name: str, base_mask: np.ndarray | None = None) -> DesignMatrix:
    raw = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    n = len(raw)
    mask = np.ones(n, dtype=bool) if base_mask is None else base_mask.copy()
    missing_policy = _default_missing_policy("continuous", spec.missing)
    finite = np.isfinite(raw)
    n_missing = int((~finite & mask).sum())

    values = raw.copy()
    warnings: list[str] = []
    if missing_policy == "drop":
        mask &= finite
    elif missing_policy == "median":
        med = float(np.nanmedian(values[mask & finite])) if np.any(mask & finite) else 0.0
        values[mask & ~finite] = med
    elif missing_policy == "zero":
        values[mask & ~finite] = 0.0
    else:
        raise ValueError(f"Continuous column {spec.column!r}: missing policy {missing_policy!r} not supported")

    # Resolve the "auto" sentinel to a concrete default here (used when this design is
    # built directly, i.e. not via the CV selection in select_continuous_design):
    # log1p when the column is non-negative (the negative fallback below handles the
    # rest), and 6 spline knots.
    effective_transform = "log1p" if spec.transform == "auto" else spec.transform
    n_knots_req = 6 if spec.n_knots == "auto" else int(spec.n_knots)

    # log1p is undefined for negatives. Rather than silently dropping those rows,
    # fall back to identity for the whole column and warn (the spline/poly basis still
    # models nonlinearity). Positive-only data (length, counts, …) keeps using log1p.
    if effective_transform == "log1p":
        n_negative = int(np.sum(values[mask] < 0))
        if n_negative:
            warnings.append(
                f"{n_negative} negative value(s) present; log1p is undefined for "
                "negatives → falling back to transform=identity (no rows dropped)"
            )
            effective_transform = "none"

    if effective_transform == "log1p":
        z_values = np.full(n, np.nan, dtype=float)
        z_values[mask] = np.log1p(values[mask])
    elif effective_transform in {"none", "identity"}:
        z_values = values.astype(float)
    else:
        raise ValueError(f"Continuous column {spec.column!r}: unknown transform {spec.transform!r}")

    z = z_values[mask].reshape(-1, 1)
    if z.shape[0] < 2:
        raise ValueError(f"Continuous column {spec.column!r}: fewer than two usable rows")
    unique_n = int(pd.Series(z[:, 0]).nunique())
    if unique_n < 2:
        raise ValueError(f"Continuous column {spec.column!r}: fewer than two unique usable values")

    basis = spec.basis.lower()
    degree = int(spec.degree)
    details: dict[str, Any] = {
        "missing_policy": missing_policy,
        "n_missing": n_missing,
        "transform": effective_transform,
        "transform_requested": spec.transform,
        "basis": basis,
        "degree": degree,
        "n_unique_values": unique_n,
    }
    if basis in {"spline", "spline_ridge"}:
        n_knots = max(2, min(n_knots_req, unique_n))
        transformer = SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False)
        H = transformer.fit_transform(z)
        details["n_knots"] = n_knots
        details["n_knots_requested"] = spec.n_knots
    elif basis in {"poly", "polynomial", "poly_ridge"}:
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        H = transformer.fit_transform(z)
    elif basis in {"linear", "linear_ridge"}:
        H = z
    else:
        raise ValueError(f"Continuous column {spec.column!r}: unknown basis {basis!r}")
    H = _standardize_dense(np.asarray(H, dtype=np.float64))
    feature_names = [f"{spec.column}__{basis}_{j}" for j in range(H.shape[1])]
    if n_missing / max(1, int(base_mask.sum()) if base_mask is not None else n) > 0.5:
        warnings.append("more than 50% of rows are missing for this continuous annotation")
    return DesignMatrix(
        name=name,
        column=spec.column,
        annotation_type="continuous",
        row_mask=mask,
        matrix=H,
        feature_names=feature_names,
        details={**details, "n_design_features": int(H.shape[1])},
        warnings=warnings,
    )


def build_binary_design(series: pd.Series, spec: NuisanceSpec, *, name: str, base_mask: np.ndarray | None = None) -> DesignMatrix:
    n = len(series)
    base = np.ones(n, dtype=bool) if base_mask is None else base_mask.copy()
    missing_policy = _default_missing_policy("binary", spec.missing)
    vals: list[str] = []
    missing_count = 0
    unknown_count = 0
    for v in series:
        text = normalize_string(v).lower()
        if text in MISSING_STRINGS:
            missing_count += 1
            vals.append(spec.missing_value if missing_policy == "category" else "0")
        elif text in TRUE_STRINGS:
            vals.append("1")
        elif text in FALSE_STRINGS:
            vals.append("0")
        else:
            # Try numeric conversion, otherwise treat non-empty unknown as category.
            try:
                vals.append("1" if float(text) != 0.0 else "0")
            except ValueError:
                unknown_count += 1
                vals.append(text if missing_policy == "category" else "0")

    s = pd.Series(vals, index=series.index)
    if missing_policy == "drop":
        base &= np.array([v not in {spec.missing_value, ""} for v in vals], dtype=bool)
    s_used = s[base]
    categories = sorted(s_used.unique().tolist())
    if len(categories) <= 2 and spec.missing_value not in categories:
        H = (s_used.astype(str).map(lambda x: 1.0 if x == "1" else 0.0).to_numpy(dtype=float).reshape(-1, 1))
        feature_names = [f"{spec.column}__binary"]
        matrix: Any = H
    else:
        ensure_sparse_available()
        cat_to_col = {cat: j for j, cat in enumerate(categories)}
        row_ind = np.arange(len(s_used), dtype=int)
        col_ind = np.array([cat_to_col[x] for x in s_used], dtype=int)
        data = np.ones(len(s_used), dtype=np.float64)
        matrix = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(s_used), len(categories)))
        feature_names = [f"{spec.column}={cat}" for cat in categories]
    warnings: list[str] = []
    if unknown_count:
        warnings.append(f"{unknown_count} non-standard binary values were coerced")
    return DesignMatrix(
        name=name,
        column=spec.column,
        annotation_type="binary",
        row_mask=base,
        matrix=matrix,
        feature_names=feature_names,
        details={
            "missing_policy": missing_policy,
            "n_missing": int(missing_count),
            "n_unknown_nonstandard": int(unknown_count),
            "categories": categories,
            "n_design_features": int(matrix.shape[1]),
        },
        warnings=warnings,
    )


def build_categorical_design(series: pd.Series, spec: NuisanceSpec, *, name: str, base_mask: np.ndarray | None = None) -> DesignMatrix:
    ensure_sparse_available()
    n = len(series)
    mask = np.ones(n, dtype=bool) if base_mask is None else base_mask.copy()
    missing_policy = _default_missing_policy("categorical", spec.missing)

    raw_vals: list[str] = []
    missing_count = 0
    for v in series:
        text = normalize_string(v)
        if text.lower() in MISSING_STRINGS:
            missing_count += 1
            if missing_policy == "drop":
                raw_vals.append("")
            else:
                raw_vals.append(spec.missing_value)
        else:
            raw_vals.append(text)
    if missing_policy == "drop":
        mask &= np.array([bool(v) for v in raw_vals], dtype=bool)

    s = pd.Series(raw_vals, index=series.index)
    s_used = s[mask].astype(str)
    counts = s_used.value_counts(dropna=False)
    kept = counts[counts >= int(spec.min_count)].index.tolist()
    warnings: list[str] = []
    rare_n = int((~s_used.isin(kept)).sum())
    if rare_n:
        if spec.rare_policy == "rare":
            s_used = s_used.where(s_used.isin(kept), spec.rare_value)
        else:
            keep_mask_used = s_used.isin(kept).to_numpy(dtype=bool)
            full_indices = np.where(mask)[0]
            new_mask = np.zeros_like(mask)
            new_mask[full_indices[keep_mask_used]] = True
            mask = new_mask
            s_used = s[mask].astype(str)
        warnings.append(f"{rare_n} rows belong to categories below min_count={spec.min_count}")

    categories = sorted(s_used.unique().tolist())
    if not categories:
        raise ValueError(f"Categorical column {spec.column!r}: no usable categories")
    cat_to_col = {cat: j for j, cat in enumerate(categories)}
    row_ind = np.arange(len(s_used), dtype=int)
    col_ind = np.array([cat_to_col[x] for x in s_used], dtype=int)
    data = np.ones(len(s_used), dtype=np.float64)
    H = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(len(s_used), len(categories)))
    if len(categories) > max(50, len(s_used) // 5):
        warnings.append("high-cardinality categorical annotation; cross-fitting is strongly recommended")
    if missing_count / max(1, int(base_mask.sum() if base_mask is not None else n)) > 0.5:
        warnings.append("more than 50% of rows are missing for this categorical annotation")
    return DesignMatrix(
        name=name,
        column=spec.column,
        annotation_type="categorical",
        row_mask=mask,
        matrix=H,
        feature_names=[f"{spec.column}={cat}" for cat in categories],
        details={
            "missing_policy": missing_policy,
            "n_missing": int(missing_count),
            "min_count": int(spec.min_count),
            "rare_policy": spec.rare_policy,
            "n_rare_rows": int(rare_n),
            "n_categories": int(len(categories)),
            "n_design_features": int(H.shape[1]),
            "top_categories": counts.head(20).astype(int).to_dict(),
        },
        warnings=warnings,
    )


def build_multilabel_design(series: pd.Series, spec: NuisanceSpec, *, name: str, base_mask: np.ndarray | None = None) -> DesignMatrix:
    ensure_sparse_available()
    n = len(series)
    mask = np.ones(n, dtype=bool) if base_mask is None else base_mask.copy()
    missing_policy = _default_missing_policy("multilabel", spec.missing)

    token_lists: list[list[str]] = []
    missing_count = 0
    for v in series:
        toks = parse_token_string(
            v,
            sep=spec.sep,
            strip_scores=spec.strip_scores,
            strip_evidence=spec.strip_evidence,
            token_mode=spec.token_mode,
        )
        if not toks and is_missing_value(v):
            missing_count += 1
            if missing_policy == "category":
                toks = [spec.missing_value]
        token_lists.append(toks)

    if missing_policy == "drop":
        mask &= np.array([len(toks) > 0 for toks in token_lists], dtype=bool)

    # Count tokens only among rows in the active mask.
    counts: dict[str, int] = {}
    for active, toks in zip(mask, token_lists, strict=False):
        if not active:
            continue
        for tok in toks:
            counts[tok] = counts.get(tok, 0) + 1
    if not counts:
        raise ValueError(f"Multilabel column {spec.column!r}: no usable tokens")

    sorted_tokens = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kept_tokens = [tok for tok, c in sorted_tokens if c >= int(spec.min_count)]
    n_below_min = len(sorted_tokens) - len(kept_tokens)
    if int(spec.max_features) > 0 and len(kept_tokens) > int(spec.max_features):
        kept_tokens = kept_tokens[: int(spec.max_features)]
    token_to_col = {tok: j for j, tok in enumerate(kept_tokens)}
    if not token_to_col:
        raise ValueError(
            f"Multilabel column {spec.column!r}: no tokens pass min_count={spec.min_count}; lower min_count"
        )

    full_indices = np.where(mask)[0]
    rows: list[int] = []
    cols: list[int] = []
    no_kept_token_rows = 0
    for out_row, original_idx in enumerate(full_indices):
        toks = token_lists[original_idx]
        kept_for_row = [token_to_col[t] for t in toks if t in token_to_col]
        if not kept_for_row:
            no_kept_token_rows += 1
        for c in sorted(set(kept_for_row)):
            rows.append(out_row)
            cols.append(c)
    data = np.ones(len(rows), dtype=np.float64)
    H = sparse.csr_matrix((data, (rows, cols)), shape=(len(full_indices), len(token_to_col)))
    warnings: list[str] = []
    if n_below_min:
        warnings.append(f"{n_below_min} tokens were removed by min_count={spec.min_count}")
    if no_kept_token_rows > 0:
        warnings.append(f"{no_kept_token_rows} active rows have no kept tokens after filtering")
    if len(token_to_col) > max(100, len(full_indices) // 5):
        warnings.append("high-dimensional multilabel annotation; cross-fitting and regularization are important")
    return DesignMatrix(
        name=name,
        column=spec.column,
        annotation_type="multilabel",
        row_mask=mask,
        matrix=H,
        feature_names=[f"{spec.column}={tok}" for tok in kept_tokens],
        details={
            "missing_policy": missing_policy,
            "n_missing": int(missing_count),
            "separator": spec.sep,
            "strip_scores": bool(spec.strip_scores),
            "strip_evidence": bool(spec.strip_evidence),
            "token_mode": spec.token_mode,
            "min_count": int(spec.min_count),
            "max_features": int(spec.max_features),
            "n_tokens_total": int(len(counts)),
            "n_tokens_below_min_count": int(n_below_min),
            "n_design_features": int(H.shape[1]),
            "top_tokens": dict(sorted_tokens[:20]),
        },
        warnings=warnings,
    )


def build_design(
    annotations: pd.DataFrame,
    spec: NuisanceSpec,
    *,
    name: str | None = None,
    column: str | None = None,
    annotation_type: str | None = None,
    base_mask: np.ndarray | None = None,
    missing_override: str | None = None,
    min_count_override: int | None = None,
    max_features_override: int | None = None,
) -> DesignMatrix:
    col = column or spec.column
    if col not in annotations.columns:
        raise ValueError(
            f"Annotation column {col!r} not found. Available columns: {', '.join(map(str, annotations.columns))}"
        )
    tmp_spec = NuisanceSpec(**asdict(spec))
    tmp_spec.column = col
    if missing_override is not None:
        tmp_spec.missing = missing_override
    if min_count_override is not None:
        tmp_spec.min_count = int(min_count_override)
    if max_features_override is not None:
        tmp_spec.max_features = int(max_features_override)

    requested_type = (annotation_type or spec.nuisance_type or "auto").lower()
    infer_details: dict[str, Any] = {}
    infer_warnings: list[str] = []
    if requested_type == "auto":
        requested_type, infer_details, infer_warnings = infer_annotation_type(annotations[col])

    design_name = name or spec.name
    if requested_type == "continuous":
        design = build_continuous_design(annotations[col], tmp_spec, name=design_name, base_mask=base_mask)
    elif requested_type == "binary":
        design = build_binary_design(annotations[col], tmp_spec, name=design_name, base_mask=base_mask)
    elif requested_type == "categorical":
        design = build_categorical_design(annotations[col], tmp_spec, name=design_name, base_mask=base_mask)
    elif requested_type == "multilabel":
        design = build_multilabel_design(annotations[col], tmp_spec, name=design_name, base_mask=base_mask)
    else:
        raise ValueError(f"Unknown annotation type {requested_type!r}")
    if infer_details:
        design.details["auto_inference"] = infer_details
    design.warnings.extend(infer_warnings)
    return design


# ---------------------------------------------------------------------------
# Ridge effect model
# ---------------------------------------------------------------------------


def is_sparse_matrix(X: Any) -> bool:
    return sparse is not None and sparse.issparse(X)


def fit_ridge(Phi: Any, Y: np.ndarray, *, alpha: float, fit_intercept: bool) -> Ridge:
    """Fit ridge `Y ≈ Φ W`. `Phi` = encoded-annotation design matrix (n × p),
    `Y` = centered embeddings (n × d), `alpha` = L2 strength."""
    solver = "lsqr" if is_sparse_matrix(Phi) else "auto"
    model = Ridge(alpha=float(alpha), fit_intercept=fit_intercept, solver=solver)
    try:
        model.fit(Phi, Y)
        return model
    except Exception as first_err:  # noqa: BLE001
        # Dense fallback often helps if solver='auto' picks an unsuitable route.
        if not is_sparse_matrix(Phi):
            try:
                model = Ridge(alpha=float(alpha), fit_intercept=fit_intercept, solver="svd")
                model.fit(Phi, Y)
                return model
            except Exception:  # noqa: BLE001
                pass
        try:
            model = Ridge(alpha=float(alpha), fit_intercept=fit_intercept, solver="lsqr")
            model.fit(Phi, Y)
            return model
        except Exception as second_err:  # noqa: BLE001
            raise RuntimeError(f"Ridge fit failed: {second_err}; first error: {first_err}") from second_err


def _fit_ridgecv(
    Phi: Any,
    Y: np.ndarray,
    *,
    fit_intercept: bool,
    fold_ids: Sequence[str] | None,
    alphas: np.ndarray,
) -> tuple[RidgeCV, str]:
    """Fit RidgeCV, preferring order-invariant closed-form GCV; fall back to
    deterministic canonical-order KFold if GCV rejects the (sparse) design."""
    grid = np.asarray(alphas, dtype=float)
    Yf = np.asarray(Y, dtype=np.float64)
    try:
        model = RidgeCV(alphas=grid, fit_intercept=fit_intercept, alpha_per_target=False)
        model.fit(Phi, Yf)
        return model, "auto_gcv"
    except Exception:  # noqa: BLE001 — GCV path can reject some sparse layouts
        from sklearn.model_selection import PredefinedSplit

        n = Yf.shape[0]
        n_splits = min(5, n)
        order = (np.argsort(np.asarray(fold_ids, dtype=object), kind="stable")
                 if fold_ids is not None else np.arange(n))
        # Deterministic, order-invariant folds: round-robin over the canonical order.
        test_folds = np.empty(n, dtype=int)
        test_folds[order] = np.arange(n) % n_splits
        model = RidgeCV(alphas=grid, fit_intercept=fit_intercept,
                        cv=PredefinedSplit(test_folds))
        model.fit(Phi, Yf)
        return model, "auto_kfold"


def select_ridge_alpha(
    Phi: Any,
    Y: np.ndarray,
    *,
    fit_intercept: bool,
    fold_ids: Sequence[str] | None = None,
    alphas: np.ndarray = RIDGE_ALPHA_GRID,
) -> tuple[float, dict[str, Any]]:
    """Pick ridge_alpha by cross-validated predictive score (annotation → embedding).

    Uses efficient closed-form GCV (leave-one-out), which is order-invariant. If GCV
    fails (rare, some sparse designs), falls back to KFold CV over the *canonical
    identifier order* so the choice stays independent of input row order.
    """
    model, source = _fit_ridgecv(
        Phi, Y, fit_intercept=fit_intercept, fold_ids=fold_ids, alphas=alphas,
    )
    return float(model.alpha_), {"ridge_alpha_source": source}


def select_continuous_design(
    series: pd.Series,
    target: np.ndarray,
    spec: NuisanceSpec,
    *,
    name: str,
    base_mask: np.ndarray | None = None,
) -> tuple[DesignMatrix, dict[str, Any]]:
    """Pick the continuous `transform` / `n_knots` that best predict the embedding.

    Only the axes left as `"auto"` are searched; explicit values are respected. Each
    candidate design is scored by the same CV predictive score used for ridge_alpha
    (RidgeCV best CV score, higher = better). Returns the winning design plus a
    selection record for diagnostics.
    """
    raw = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(raw)
    all_nonneg = bool(np.all(raw[finite] >= 0)) if finite.any() else False

    if spec.transform != "auto":
        transforms = [spec.transform]
    else:
        transforms = ["log1p", "identity"] if all_nonneg else ["identity"]

    is_spline = spec.basis.lower() in {"spline", "spline_ridge"}
    if not is_spline or spec.n_knots != "auto":
        knot_opts = [6 if spec.n_knots == "auto" else int(spec.n_knots)]
    else:
        knot_opts = [4, 6, 8, 12]

    X = np.asarray(target, dtype=np.float64)
    best: tuple[float, DesignMatrix, str, int] | None = None
    candidates: list[dict[str, Any]] = []
    for tf in transforms:
        for nk in knot_opts:
            cand_spec = NuisanceSpec(**{**asdict(spec), "transform": tf, "n_knots": nk})
            try:
                design = build_continuous_design(series, cand_spec, name=name, base_mask=base_mask)
            except ValueError:
                continue
            Yc = X[design.row_mask]
            Yc = Yc - Yc.mean(axis=0, keepdims=True)
            model, _ = _fit_ridgecv(
                design.matrix, Yc, fit_intercept=spec.fit_intercept,
                fold_ids=None, alphas=RIDGE_ALPHA_GRID,
            )
            score = float(model.best_score_)
            candidates.append({"transform": design.details.get("transform", tf),
                               "n_knots": design.details.get("n_knots"),
                               "cv_score": score, "ridge_alpha": float(model.alpha_)})
            if best is None or score > best[0]:
                best = (score, design, tf, nk)

    if best is None:  # nothing built (degenerate column) — let build_design raise cleanly
        return build_continuous_design(series, spec, name=name, base_mask=base_mask), {}
    score, design, tf, nk = best
    selection = {
        "selected_transform": design.details.get("transform", tf),
        "selected_n_knots": design.details.get("n_knots"),
        "cv_score": score,
        "candidates": candidates,
    }
    design.details["hparam_selection"] = selection
    return design, selection


def ridge_predict_effect(
    Phi: Any,
    Y: np.ndarray,
    *,
    alpha: float | str,
    cross_fit: int | None,
    random_state: int,
    fit_intercept: bool,
    fold_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Predict the effect `E ≈ Φ W` with ridge, optionally out-of-fold.

    `Phi` = encoded-annotation design matrix (n × p), `Y` = centered embeddings
    (n × d); the returned prediction is the effect `E` used to build the background.

    ``alpha`` may be a float (fixed) or the string ``"auto"`` — in which case it is
    chosen once by cross-validated GCV (``select_ridge_alpha``) before the (optional)
    out-of-fold effect estimation runs at that alpha.

    When ``fold_ids`` is given, cross-fit folds are assigned by a canonical ordering
    of the identifiers rather than by row position, so the out-of-fold predictions
    (and hence the background) are invariant to the input row order. Assumes the
    identifiers are unique (true for H5 keys); ties fall back to a stable order.
    """
    Y = np.asarray(Y, dtype=np.float64)
    n = Y.shape[0]
    warnings: list[str] = []
    if n < 2:
        raise ValueError("Need at least two rows for ridge effect estimation")

    # Resolve an "auto" alpha once (order-invariant), then estimate the effect at it.
    if isinstance(alpha, str):
        alpha_value, alpha_details = select_ridge_alpha(
            Phi, Y, fit_intercept=fit_intercept, fold_ids=fold_ids,
        )
        alpha_details["ridge_alpha_grid"] = [float(a) for a in RIDGE_ALPHA_GRID]
    else:
        alpha_value = float(alpha)
        alpha_details = {"ridge_alpha_source": "explicit"}

    use_cv = cross_fit is not None and int(cross_fit) > 1 and n >= int(cross_fit)
    if use_cv:
        n_splits = min(int(cross_fit), n)
        pred = np.zeros_like(Y, dtype=np.float64)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))
        # Assign folds by canonical identifier order (order-invariant) when ids are
        # available; otherwise fall back to positional folds.
        if fold_ids is not None:
            order = np.argsort(np.asarray(fold_ids, dtype=object), kind="stable")
        else:
            order = np.arange(n)
        fold_sizes: list[int] = []
        for train_pos, test_pos in kf.split(np.arange(n)):
            train_idx, test_idx = order[train_pos], order[test_pos]
            fold_sizes.append(int(len(test_idx)))
            model = fit_ridge(Phi[train_idx], Y[train_idx], alpha=alpha_value, fit_intercept=fit_intercept)
            pred[test_idx] = model.predict(Phi[test_idx])
        fit_mode = "cross_fit"
    else:
        if cross_fit and int(cross_fit) > 1 and n < int(cross_fit):
            warnings.append(f"cross_fit={cross_fit} requested but only n={n}; fitting on all rows")
        model = fit_ridge(Phi, Y, alpha=alpha_value, fit_intercept=fit_intercept)
        pred = model.predict(Phi)
        n_splits = 0
        fold_sizes = []
        fit_mode = "fit_all"

    pred = np.asarray(pred, dtype=np.float64)
    details = {
        "fit_mode": fit_mode,
        "cross_fit": int(n_splits),
        "ridge_alpha": float(alpha_value),
        "fit_intercept": bool(fit_intercept),
        "n_fit_rows": int(n),
        "n_design_features": int(Phi.shape[1]),
        "design_is_sparse": bool(is_sparse_matrix(Phi)),
        "fold_sizes": fold_sizes,
        **alpha_details,
    }
    return pred, details, warnings


def center_effects(E: np.ndarray) -> np.ndarray:
    E = np.asarray(E, dtype=np.float64)
    return E - E.mean(axis=0, keepdims=True)


def trace_second_moment(E: np.ndarray) -> float:
    E = np.asarray(E, dtype=np.float64)
    if E.size == 0 or E.shape[0] == 0:
        return 0.0
    return float(np.sum(E * E) / E.shape[0])


def estimate_effect_block(ids: list[str], X: np.ndarray, annotations: pd.DataFrame, spec: NuisanceSpec) -> EffectBlock:
    """Estimate one nuisance effect block: encode the annotation into a design `Φ`,
    ridge-fit it against the centered embeddings `X` (n × d), and return the predicted
    effect `E` (the block later signed into the background). `ids` label the `X` rows."""
    design = build_design(annotations, spec)
    # For continuous nuisances, CV-select transform / n_knots left as "auto".
    if design.annotation_type == "continuous" and (spec.transform == "auto" or spec.n_knots == "auto"):
        design, _sel = select_continuous_design(
            annotations[spec.column], X, spec, name=spec.name,
        )
    row_mask = design.row_mask.copy()
    used_indices = np.where(row_mask)[0]
    if len(used_indices) < 2:
        raise ValueError(f"Nuisance {spec.name!r}: fewer than two usable rows")

    Phi = design.matrix
    X_used = X[used_indices]
    X_centered = X_used - X_used.mean(axis=0, keepdims=True)
    model_target = X_centered
    details: dict[str, Any] = {
        "raw_spec": spec.raw,
        "column": spec.column,
        "name": spec.name,
        "requested_type": spec.nuisance_type,
        "resolved_type": design.annotation_type,
        "n_total_rows": int(X.shape[0]),
        "n_used_rows": int(len(used_indices)),
        "n_features": int(X.shape[1]),
        "scale": float(spec.scale),
        "encoder_details": design.details,
    }
    warnings = list(design.warnings)

    # Optional conditioning: first estimate the embedding component predictable
    # from the condition annotation, then fit the nuisance on the residual.
    condition_details: dict[str, Any] | None = None
    if spec.condition_on:
        cond_spec = NuisanceSpec(**asdict(spec))
        cond_spec.column = spec.condition_on
        cond_spec.nuisance_type = spec.condition_type
        cond_spec.missing = spec.condition_missing
        cond_spec.min_count = spec.condition_min_count
        cond_spec.max_features = spec.condition_max_features
        cond_design = build_design(
            annotations,
            cond_spec,
            name=f"{spec.name}__condition_on__{safe_id(spec.condition_on)}",
            column=spec.condition_on,
            annotation_type=spec.condition_type,
            base_mask=row_mask,
            missing_override=spec.condition_missing,
            min_count_override=spec.condition_min_count,
            max_features_override=spec.condition_max_features,
        )
        cond_mask = cond_design.row_mask
        # If conditioning dropped additional rows, subset nuisance design to the
        # intersection. Design rows are ordered according to row_mask.
        if not np.array_equal(cond_mask, row_mask):
            old_used = np.where(row_mask)[0]
            keep_old_rows = np.isin(old_used, np.where(cond_mask)[0])
            Phi = Phi[keep_old_rows]
            row_mask = cond_mask
            used_indices = np.where(row_mask)[0]
            X_used = X[used_indices]
            X_centered = X_used - X_used.mean(axis=0, keepdims=True)
        cond_pred, cond_fit_details, cond_fit_warnings = ridge_predict_effect(
            cond_design.matrix,
            X_centered,
            alpha=spec.ridge_alpha,
            cross_fit=spec.cross_fit,
            random_state=spec.random_state + 1009,
            fit_intercept=spec.fit_intercept,
            fold_ids=[ids[i] for i in used_indices],
        )
        cond_pred = center_effects(cond_pred)
        model_target = X_centered - cond_pred
        condition_details = {
            "condition_on": spec.condition_on,
            "condition_resolved_type": cond_design.annotation_type,
            "condition_encoder_details": cond_design.details,
            "condition_fit_details": cond_fit_details,
            "condition_trace_second_moment": trace_second_moment(cond_pred),
            "note": "Nuisance effect was fit to embedding residuals after removing the condition_on-predictable component.",
        }
        warnings.extend(cond_design.warnings)
        warnings.extend([f"condition_on: {w}" for w in cond_fit_warnings])

    G_pred, fit_details, fit_warnings = ridge_predict_effect(
        Phi,
        model_target,
        alpha=spec.ridge_alpha,
        cross_fit=spec.cross_fit,
        random_state=spec.random_state,
        fit_intercept=spec.fit_intercept,
        fold_ids=[ids[i] for i in used_indices],
    )
    G = center_effects(G_pred)
    warnings.extend(fit_warnings)

    target_trace = trace_second_moment(X_centered)
    effect_trace = trace_second_moment(G)
    residual = model_target - G
    denom = float(np.sum(model_target * model_target))
    fro_r2 = 1.0 - float(np.sum(residual * residual)) / denom if denom > 1e-12 else float("nan")
    trace_ratio = effect_trace / target_trace if target_trace > 1e-12 else float("nan")

    if fit_details["fit_mode"] == "fit_all" and design.annotation_type in {"categorical", "multilabel"} and design.n_features > max(20, design.n_rows // 10):
        warnings.append("high-dimensional categorical/multilabel design was fit without cross-fitting; overfitting risk")
    if effect_trace <= 1e-12:
        warnings.append("near-zero predicted nuisance effect; background may be uninformative")
    if design.n_features >= design.n_rows:
        warnings.append("design has at least as many features as rows; ridge/cross-fitting is important")

    effect_ids = [f"{safe_id(ids[i])}__{safe_id(spec.name)}" for i in used_indices]
    details.update(
        {
            "fit_details": fit_details,
            "condition_details": condition_details,
            "target_trace_second_moment": target_trace,
            "effect_trace_second_moment": effect_trace,
            "effect_trace_ratio_to_target": trace_ratio,
            "frobenius_r2_against_model_target": fro_r2,
        }
    )
    return EffectBlock(
        name=spec.name,
        column=spec.column,
        effect_type=f"annotation_{design.annotation_type}_ridge_effect",
        ids=effect_ids,
        effects=G,
        scale=spec.scale,
        details=details,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Background assembly
# ---------------------------------------------------------------------------


def normalize_effects(E: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    mode = (mode or "none").lower()
    E0 = center_effects(E)
    before_trace = trace_second_moment(E0)
    if mode in {"none", "row_count", "legacy_row_count"}:
        E1 = E0
        factor = 1.0
    elif mode == "trace":
        if before_trace <= 1e-12:
            raise ValueError("Cannot trace-normalize a near-zero effect block")
        factor = 1.0 / math.sqrt(before_trace)
        E1 = E0 * factor
    else:
        raise ValueError("--block-normalization must be none, trace, or row_count")
    return E1, {
        "block_normalization": mode,
        "trace_second_moment_before_normalization": before_trace,
        "normalization_factor": factor,
        "trace_second_moment_after_normalization": trace_second_moment(E1),
    }


def sign_background(E: np.ndarray, scale: float) -> np.ndarray:
    B = np.vstack([float(scale) * E, -float(scale) * E])
    return center_effects(B)


def block_to_background(block: EffectBlock, *, normalization: str) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    E_norm, norm_details = normalize_effects(block.effects, normalization)
    B = sign_background(E_norm, block.scale)
    ids = [f"{safe_id(i)}__pos" for i in block.ids] + [f"{safe_id(i)}__neg" for i in block.ids]
    details = {
        "name": block.name,
        "column": block.column,
        "type": block.effect_type,
        "scale": float(block.scale),
        "n_effect_rows": int(block.effects.shape[0]),
        "n_background_rows_before_balance": int(B.shape[0]),
        "n_features": int(B.shape[1]),
        **norm_details,
        "estimator_details": block.details,
        "warnings": block.warnings,
    }
    return ids, B, details


def apply_row_count_balance(blocks: list[tuple[list[str], np.ndarray, dict[str, Any]]]) -> list[tuple[list[str], np.ndarray, dict[str, Any]]]:
    if len(blocks) <= 1:
        return blocks
    max_rows = max(B.shape[0] for _, B, _ in blocks)
    out: list[tuple[list[str], np.ndarray, dict[str, Any]]] = []
    for ids, B, details in blocks:
        factor = math.sqrt(max_rows / B.shape[0])
        out.append((ids, B * factor, {**details, "row_count_balance_factor": factor}))
    return out


def assemble_background(
    blocks: list[EffectBlock],
    *,
    block_normalization: str,
) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
    signed_blocks = [block_to_background(block, normalization=block_normalization) for block in blocks]
    if block_normalization.lower() in {"row_count", "legacy_row_count"}:
        signed_blocks = apply_row_count_balance(signed_blocks)

    all_ids: list[str] = []
    matrices: list[np.ndarray] = []
    details: list[dict[str, Any]] = []
    feature_dim: int | None = None
    for ids, B, det in signed_blocks:
        if feature_dim is None:
            feature_dim = B.shape[1]
        elif B.shape[1] != feature_dim:
            raise ValueError(f"Feature mismatch across blocks: expected {feature_dim}, got {B.shape[1]}")
        all_ids.extend(ids)
        matrices.append(B)
        details.append({**det, "n_background_rows": int(B.shape[0])})
    B_final = np.vstack(matrices).astype(np.float32)
    B_final = center_effects(B_final).astype(np.float32)
    if B_final.shape[0] < 2:
        raise ValueError("Background needs at least two rows")
    if not np.isfinite(B_final).all():
        raise ValueError("Background contains NaN or inf")
    return all_ids, B_final, details




def effect_summary_dataframe(block_details: list[dict[str, Any]]) -> pd.DataFrame:
    """Return the same compact effect summary written by the standalone CLI."""
    rows: list[dict[str, Any]] = []
    for b in block_details:
        est = b.get("estimator_details", {})
        fit = est.get("fit_details", {}) or {}
        enc = est.get("encoder_details", {}) or {}
        rows.append(
            {
                "name": b.get("name"),
                "column": b.get("column"),
                "type": b.get("type"),
                "scale": b.get("scale"),
                "n_effect_rows": b.get("n_effect_rows"),
                "n_background_rows": b.get("n_background_rows"),
                "n_features": b.get("n_features"),
                "resolved_type": est.get("resolved_type"),
                "n_used_rows": est.get("n_used_rows"),
                "n_design_features": fit.get("n_design_features", enc.get("n_design_features")),
                "fit_mode": fit.get("fit_mode"),
                "cross_fit": fit.get("cross_fit"),
                "ridge_alpha": fit.get("ridge_alpha"),
                "ridge_alpha_source": fit.get("ridge_alpha_source"),
                "effect_trace_ratio_to_target": est.get("effect_trace_ratio_to_target"),
                "frobenius_r2_against_model_target": est.get("frobenius_r2_against_model_target"),
                "warnings": "; ".join(map(str, b.get("warnings", []))),
            }
        )
    return pd.DataFrame(rows)


def build_annotation_nuisance_background(
    *,
    ids: Sequence[str],
    X: np.ndarray,
    annotations: pd.DataFrame,
    nuisance_specs: Sequence[str | NuisanceSpec],
    ridge_alpha: float | str | None = "auto",
    cross_fit: int = 1,
    random_state: int = 42,
    block_normalization: str = "none",
    target_name: str = "target",
    regularization_mu: float = 1e-6,
) -> NuisanceBackgroundResult:
    """Build an in-memory signed annotation-nuisance background for ρPCA.

    This is the integrated ProtSpace entry point. It keeps the mathematical
    behavior of the standalone experiment script while avoiding HDF5/CLI I/O
    during projection.
    """
    ids = [str(i) for i in ids]
    if not ids:
        raise ValueError("No protein identifiers supplied for nuisance background.")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Target embeddings must be 2D, got {X.shape}.")
    if X.shape[0] != len(ids):
        raise ValueError(f"ids length {len(ids)} != embedding rows {X.shape[0]}.")
    if X.shape[0] < 2:
        raise ValueError("Need at least two target embeddings for ρPCA.")
    if not np.isfinite(X).all():
        raise ValueError("Target embeddings contain NaN or infinite values.")
    if annotations is None or annotations.empty:
        raise ValueError("--nuisance requires an annotation table with an identifier column.")
    if not nuisance_specs:
        raise ValueError("At least one --nuisance specification is required.")

    aligned = align_annotations(ids, annotations)
    parsed_specs: list[NuisanceSpec] = []
    for spec in nuisance_specs:
        if isinstance(spec, NuisanceSpec):
            parsed_specs.append(spec)
        else:
            parsed_specs.append(
                parse_nuisance_spec(
                    str(spec),
                    global_cross_fit=int(cross_fit),
                    global_ridge_alpha=_resolve_alpha(ridge_alpha),
                    random_state=int(random_state),
                )
            )

    blocks: list[EffectBlock] = []
    warnings: list[str] = []
    for spec in parsed_specs:
        LOGGER.info("Building annotation nuisance block: %s from column %s", spec.name, spec.column)
        block = estimate_effect_block(ids, X, aligned, spec)
        blocks.append(block)
        warnings.extend([f"{spec.name}: {w}" for w in block.warnings])

    bg_ids, B, block_details = assemble_background(blocks, block_normalization=block_normalization)
    summary = effect_summary_dataframe(block_details)
    manifest: dict[str, Any] = {
        "version": 1,
        "mode": "annotation_nuisance",
        "target_name": str(target_name),
        "regularization_mu": float(regularization_mu),
        "background_h5": "",
        "recommended_protspace_command": (
            "protspace prepare -i TARGET.h5:MODEL -m pca2,umap2,rhopca2 "
            "--rhopca-background background.h5 --no-standard-scale "
            f"--regularization-mu {regularization_mu} -o OUT"
        ),
        "target": {
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "ids_preview": ids[:10],
        },
        "background": {
            "n_background": int(B.shape[0]),
            "n_features": int(B.shape[1]),
            "block_normalization": str(block_normalization),
        },
        "nuisance_specs": [asdict(s) for s in parsed_specs],
        "blocks": block_details,
        "warnings": warnings,
        "note": (
            "Background rows are signed predicted annotation→embedding shifts. "
            "Target embeddings are not modified; ρPCA uses this matrix only as Σ_B."
        ),
    }
    diagnostics = render_diagnostics(manifest)
    return NuisanceBackgroundResult(
        background=B,
        background_ids=bg_ids,
        manifest=manifest,
        diagnostics_md=diagnostics,
        effect_summary=summary,
        block_details=block_details,
        annotations_aligned=aligned,
    )


# ---------------------------------------------------------------------------
# Outputs and diagnostics
# ---------------------------------------------------------------------------


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=json_default)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def render_diagnostics(manifest: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Annotation nuisance ρPCA background diagnostics\n\n")
    lines.append(f"Background HDF5: `{manifest['background_h5']}`\n\n")
    lines.append(f"Target rows: **{manifest['target']['n_rows']}**, features: **{manifest['target']['n_features']}**\n\n")
    lines.append(f"Background rows: **{manifest['background']['n_background']}**, features: **{manifest['background']['n_features']}**\n\n")
    lines.append(f"Block normalization: `{manifest['background']['block_normalization']}`\n\n")
    lines.append("## Recommended ProtSpace command\n\n")
    lines.append("```bash\n")
    lines.append(manifest["recommended_protspace_command"] + "\n")
    lines.append("```\n\n")

    all_warnings = manifest.get("warnings", [])
    if all_warnings:
        lines.append("## Global warnings\n\n")
        for w in all_warnings:
            lines.append(f"- {w}\n")
        lines.append("\n")

    lines.append("## Nuisance blocks\n\n")
    for block in manifest["blocks"]:
        lines.append(f"### {block['name']} (`{block['type']}`)\n\n")
        lines.append(f"- annotation column: `{block.get('column', '')}`\n")
        lines.append(f"- scale: `{block['scale']}`\n")
        lines.append(f"- effect rows: `{block['n_effect_rows']}`\n")
        lines.append(f"- signed background rows: `{block['n_background_rows']}`\n")
        lines.append(f"- features: `{block['n_features']}`\n")
        if "row_count_balance_factor" in block:
            lines.append(f"- row-count balance factor: `{block['row_count_balance_factor']:.6g}`\n")
        est = block.get("estimator_details", {})
        for key in [
            "resolved_type",
            "n_used_rows",
            "effect_trace_ratio_to_target",
            "frobenius_r2_against_model_target",
        ]:
            if key in est:
                val = est[key]
                if isinstance(val, float):
                    lines.append(f"- {key}: `{val:.6g}`\n")
                else:
                    lines.append(f"- {key}: `{val}`\n")
        fit = est.get("fit_details", {}) or {}
        if fit:
            lines.append(f"- fit mode: `{fit.get('fit_mode')}`\n")
            lines.append(f"- cross-fit folds: `{fit.get('cross_fit')}`\n")
            lines.append(f"- design features: `{fit.get('n_design_features')}`\n")
            lines.append(f"- ridge alpha: `{fit.get('ridge_alpha')}`\n")
        cond = est.get("condition_details")
        if cond:
            lines.append(f"- conditioned on: `{cond.get('condition_on')}`\n")
        warnings = block.get("warnings", [])
        if warnings:
            lines.append("- warnings:\n")
            for w in warnings:
                lines.append(f"  - {w}\n")
        lines.append("\n")
    return "".join(lines)
