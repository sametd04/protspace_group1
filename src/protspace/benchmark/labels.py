"""Categorical label loading for label-based benchmark metrics.

Some metrics (silhouette, concordex) need a categorical label per protein.
This module reads labels from the ``protein_annotations`` table inside a
``.parquetbundle`` and aligns them with a given embedding row order.

Default label source: the UniProt ``keyword`` annotation column. Each
protein's keywords are a ``;``-separated list (alphabetically sorted),
which mixes biology (``Cardiotoxin``, ``Neurotoxin``) with structural
metadata (``Disulfide bond``, ``Signal``). We strip the ``KW-NNNN`` prefix,
filter out non-functional keywords, and use the first remaining keyword
as the label.

For other annotation columns (e.g., ``protein_families`` for clean
single-string labels), use ``load_labels_from_bundle(...,
column="protein_families", filter_keywords=False)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protspace.benchmark.io.paths import BenchmarkPaths

import io
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from protspace.data.io.bundle import read_bundle

logger = logging.getLogger(__name__)

# Generic, structural, or methodological UniProt keywords that don't
# describe a protein's function. Filtered out so the first remaining
# keyword reflects functional class.
NON_FUNCTIONAL_KEYWORDS: set[str] = {
    "3D-structure",
    "Alternative initiation",
    "Alternative splicing",
    "Amidation",
    "Cell membrane",
    "Cleavage on pair of basic residues",
    "Direct protein sequencing",
    "Disulfide bond",
    "Glycoprotein",
    "Hydroxylation",
    "Lipoprotein",
    "Membrane",
    "Phosphoprotein",
    "Pyrrolidone carboxylic acid",
    "Reference proteome",
    "Secreted",
    "Signal",
    "Toxin",  # too generic when the dataset is "all toxins"
    "Transmembrane",
    "Transmembrane helix",
}


def _strip_kw_code(kw: str) -> str:
    """``"KW-0123 (Cardiotoxin)"`` -> ``"Cardiotoxin"``."""
    kw = kw.strip()
    if "(" in kw and kw.endswith(")"):
        return kw.split("(", 1)[1][:-1].strip()
    return kw


def first_functional_keyword(raw: str | None) -> str | None:
    """Return the first non-structural keyword from a ``;``-separated list."""
    if not raw:
        return None
    for token in raw.split(";"):
        name = _strip_kw_code(token)
        if name and name not in NON_FUNCTIONAL_KEYWORDS:
            return name
    return None


def load_labels_from_bundle(
    bundle_path: Path | str,
    headers: list[str],
    column: str = "keyword",
    filter_keywords: bool = True,
    min_class_size: int = 3,
) -> np.ndarray:
    """Load categorical labels from a parquetbundle, aligned to ``headers``.

    Parameters
    ----------
    bundle_path
        Path to ``.parquetbundle`` containing a ``protein_annotations`` table.
    headers
        Protein IDs in the order they appear in the embedding matrix
        (``EmbeddingSet.headers``). The returned array is in this order.
    column
        Annotation column to use as the label source. Default ``"keyword"``.
    filter_keywords
        If ``True`` and ``column == "keyword"``, drop structural / metadata
        keywords and use the first remaining keyword. If ``False``, use the
        raw column value as-is.
    min_class_size
        Drop classes with fewer than this many members (set their label to
        ``None``) to reduce noise from singleton classes.

    Returns
    -------
    Array of shape ``(len(headers),)`` with one label per protein. Missing
    or filtered labels are ``None``.
    """
    bundle_path = Path(bundle_path)
    parquets, _ = read_bundle(bundle_path)
    ann = pq.read_table(io.BytesIO(parquets[0])).to_pandas()

    return _labels_from_annotations_frame(
        ann,
        headers=headers,
        column=column,
        filter_keywords=filter_keywords,
        min_class_size=min_class_size,
        source=bundle_path,
    )


def load_labels_from_parquet(
    parquet_path: Path | str,
    headers: list[str],
    column: str = "keyword",
    filter_keywords: bool = True,
    min_class_size: int = 3,
) -> np.ndarray:
    """Load categorical labels from a parquet annotation table."""
    parquet_path = Path(parquet_path)
    ann = pq.read_table(parquet_path).to_pandas()
    return _labels_from_annotations_frame(
        ann,
        headers=headers,
        column=column,
        filter_keywords=filter_keywords,
        min_class_size=min_class_size,
        source=parquet_path,
    )


def _labels_from_annotations_frame(
    ann,
    headers: list[str],
    column: str,
    filter_keywords: bool,
    min_class_size: int,
    source: Path,
) -> np.ndarray:
    """Extract labels from a protein annotation table."""
    if column not in ann.columns:
        raise ValueError(
            f"Column '{column}' not found in {source}. "
            f"Available: {sorted(ann.columns)}"
        )

    id_column = "protein_id" if "protein_id" in ann.columns else "identifier"
    if id_column not in ann.columns:
        raise ValueError(
            f"No protein identifier column found in {source}. "
            "Expected one of ['protein_id', 'identifier']."
        )

    if filter_keywords and column == "keyword":
        ann["_label"] = ann[column].apply(first_functional_keyword)
    else:
        ann["_label"] = ann[column].where(ann[column].astype(bool), None)

    if min_class_size > 1:
        counts = ann["_label"].value_counts(dropna=True)
        small = set(counts[counts < min_class_size].index)
        if small:
            ann.loc[ann["_label"].isin(small), "_label"] = None

    id_to_label = dict(zip(ann[id_column], ann["_label"], strict=False))
    return np.array([id_to_label.get(h) for h in headers], dtype=object)


def load_silhouette_labels(
    paths: BenchmarkPaths, headers: list[str]
) -> np.ndarray | None:
    """Load labels for silhouette metric from bundle.

    Convenience wrapper around load_labels_from_bundle for benchmark pipeline.
    Looks up labels in this order:
    1. Dataset bundle / benchmark bundle (`paths.bundle_path`)
    2. Dataset tmp annotations parquet (`data/<dataset>/tmp/all_annotations.parquet`)
    Returns None if neither source provides usable labels.

    Parameters
    ----------
    paths
        Benchmark paths configuration.
    headers
        Protein IDs in embedding order.

    Returns
    -------
    Label array or None if bundle unavailable.
    """
    annotation_errors: list[str] = []

    if paths.bundle_path.exists():
        try:
            return load_labels_from_bundle(paths.bundle_path, headers)
        except ValueError as exc:
            annotation_errors.append(str(exc))

    tmp_annotations = paths.dataset_dir / "tmp" / "all_annotations.parquet"
    if tmp_annotations.exists():
        try:
            return load_labels_from_parquet(tmp_annotations, headers)
        except ValueError as exc:
            annotation_errors.append(str(exc))

    if annotation_errors:
        logger.warning(
            "Skipping silhouette labels for %s: %s",
            paths.data,
            " | ".join(annotation_errors),
        )
    return None


def label_summary(labels: np.ndarray) -> dict[str, int | dict[str, int]]:
    """Quick summary of a label array. Useful for logging in run scripts."""
    valid = [lbl for lbl in labels if lbl is not None]
    classes: dict[str, int] = {}
    for lbl in valid:
        classes[lbl] = classes.get(lbl, 0) + 1
    return {
        "n_total": len(labels),
        "n_labelled": len(valid),
        "n_classes": len(classes),
        "classes": dict(sorted(classes.items(), key=lambda kv: -kv[1])),
    }
