#!/usr/bin/env python3
"""
Clean final-slide workflow for 3FTx and ToxProt ρPCA results.

3FTx main-slide story:
  1. PCA on mixed full-or-mature embeddings colored by SP-in-embedding.
  2. PCA on mature-only embeddings colored by major_group as the positive control.
  3. ρPCA using the manual paired-delta background Δ = E(full)-E(mature), colored by major_group.
  4. ρPCA using the unified annotation-derived background from sp_in_embedding, colored by major_group.

ToxProt main-slide story:
  1. PCA colored by length.
  2. PCA colored by taxonomy order.
  3. ρPCA with length+order nuisance colored by length.
  4. ρPCA with length+order nuisance colored by taxonomy order.

The script intentionally creates all generated files from minimal inputs:
  - 3FTx: data/3FTx/full/3FTx_data.xlsx
  - ToxProt: UniProt reviewed toxin query fetched online

It shells out to the installed `protspace` CLI.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import textwrap
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score


STEP_SENTENCES = [
    "read data/3FTx/full/3FTx_data.xlsx",
    "create full_or_mature FASTA",
    "create mature_only FASTA",
    "create annotations CSV",
    "embed both",
    "run PCA + ρPCA",
    "create the four 3FTx slide plots + metrics",
    "fetch reviewed toxin FASTA from UniProt",
    "rewrite headers to accession-only IDs",
    "embed successfully retrieved sequences",
    "fetch/build taxonomy order annotation",
    "run PCA + ρPCA with length + order nuisance",
    "create the four slide plots + metrics",
]

# The plots do not print metric boxes. Metrics are exported as CSV and optional
# standalone metric-card images so they can be shown/hidden in the slide deck.
METRIC_COLUMNS = ["label_silhouette", "label_eta2", "kmeans_nmi"]
METRIC_LABELS = {
    "label_silhouette": "Silhouette",
    "label_eta2": "Label variance η²",
    "kmeans_nmi": "NMI",
}
MISSING_ORDER = "__missing_order__"
AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYXBZUOJ]", re.IGNORECASE)
SP_COLORS = {"no": "#0072B2", "yes": "#009E73"}  # blue = SP absent, green = SP present


@dataclass(frozen=True)
class Paths:
    root: Path
    data_3ftx_xlsx: Path
    run_3ftx: Path
    inputs_3ftx: Path
    emb_3ftx: Path
    prep_3ftx: Path
    figs_3ftx: Path
    backup_3ftx: Path
    scores_3ftx: Path
    backup_tox: Path
    data_tox: Path
    run_tox: Path
    emb_tox: Path
    prep_tox: Path
    figs_tox: Path
    scores_tox: Path


def get_paths(root: Path) -> Paths:
    run_3ftx = root / "results" / "3ftx_final_rhopca"
    run_tox = root / "results" / "toxprot_final_rhopca"
    return Paths(
        root=root,
        data_3ftx_xlsx=root / "data" / "3FTx" / "full" / "3FTx_data.xlsx",
        run_3ftx=run_3ftx,
        inputs_3ftx=run_3ftx / "inputs",
        emb_3ftx=run_3ftx / "embeddings",
        prep_3ftx=run_3ftx / "protspace",
        figs_3ftx=run_3ftx / "slide_plots",
        backup_3ftx=run_3ftx / "backup_plots",
        scores_3ftx=run_3ftx / "scores",
        data_tox=root / "data" / "toxprot",
        run_tox=run_tox,
        emb_tox=run_tox / "embeddings",
        prep_tox=run_tox / "protspace",
        figs_tox=run_tox / "slide_plots",
        scores_tox=run_tox / "scores",
        backup_tox=run_tox / "backup_plots",
    )


def log_step(sentence: str) -> None:
    print(f"\n=== {sentence} ===", flush=True)


def shlex_quote(s: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=,+@%-]+", s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def run_cmd(cmd: Sequence[str], *, cwd: Path, dry_run: bool = False) -> None:
    printable = " ".join(shlex_quote(str(x)) for x in cmd)
    print(f"$ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def clean_sequence(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    seq = str(value).strip().replace(" ", "").replace("\n", "").replace("\r", "")
    if not seq or seq.lower() in {"nan", "none", "na", "-"}:
        return ""
    return AA_RE.sub("", seq.upper())


def write_fasta(records: Sequence[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for identifier, sequence in records:
            f.write(f">{identifier}\n")
            f.write("\n".join(textwrap.wrap(sequence, 80)))
            f.write("\n")


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str], *, required: bool = True) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        key = cand.lower()
        if key in lower:
            return lower[key]
    if required:
        raise KeyError(f"None of {candidates!r} found. Available columns: {list(df.columns)!r}")
    return None


def infer_identifier(row: pd.Series, idx: int) -> str:
    for col in ["uniprot_id", "uniprot", "accession", "Entry", "entry", "identifier"]:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip() and str(val).strip().lower() != "nan":
                token = str(val).strip()
                m = re.match(r"^(?:sp|tr)\|([^|]+)\|", token)
                if m:
                    return m.group(1)
                return re.split(r"\s+", token)[0]
    if "id_new" in row.index and pd.notna(row["id_new"]):
        return f"FTX_{str(row['id_new']).strip()}"
    return f"FTX_{idx + 1:05d}"


# read data/3FTx/full/3FTx_data.xlsx
def read_3ftx_excel(paths: Paths) -> pd.DataFrame:
    if not paths.data_3ftx_xlsx.exists():
        raise FileNotFoundError(f"Missing 3FTx Excel file: {paths.data_3ftx_xlsx}")
    return pd.read_excel(paths.data_3ftx_xlsx)


# create full_or_mature FASTA
# create mature_only FASTA
# create annotations CSV
def prepare_3ftx_inputs(paths: Paths, df: pd.DataFrame) -> dict[str, Path]:
    full_col = first_existing_col(df, ["full_seq", "full_sequence", "Full sequence", "full sequence"], required=False)
    mature_col = first_existing_col(df, ["mature_seq", "mature_sequence", "Mature sequence", "mature sequence"])
    major_col = first_existing_col(df, ["major_group", "major group", "Major group", "Major_Group"], required=False)
    cysteine_col = first_existing_col(df, ["cysteine_group", "cysteine group", "Cysteine group", "Cysteine_Group"], required=False)

    records_full: list[tuple[str, str]] = []
    records_mature: list[tuple[str, str]] = []
    ann_rows_full: list[dict[str, object]] = []
    ann_rows_mature: list[dict[str, object]] = []

    seen: set[str] = set()
    for idx, row in df.iterrows():
        identifier = infer_identifier(row, idx)
        if identifier in seen:
            base = identifier
            k = 2
            while f"{base}_{k}" in seen:
                k += 1
            identifier = f"{base}_{k}"
        seen.add(identifier)

        full_seq = clean_sequence(row[full_col]) if full_col is not None else ""
        mature_seq = clean_sequence(row[mature_col])
        if not full_seq and not mature_seq:
            continue
        use_full = bool(full_seq)
        seq_full_or_mature = full_seq if use_full else mature_seq
        seq_mature = mature_seq if mature_seq else seq_full_or_mature

        records_full.append((identifier, seq_full_or_mature))
        records_mature.append((identifier, seq_mature))

        major = str(row[major_col]).strip() if major_col and pd.notna(row[major_col]) else "unknown"
        cysteine = str(row[cysteine_col]).strip() if cysteine_col and pd.notna(row[cysteine_col]) else "unknown"
        ann_rows_full.append(
            {
                "identifier": identifier,
                "sp_in_embedding": "yes" if use_full else "no",
                "sequence_source": "full_seq" if use_full else "mature_seq_fallback",
                "sequence_length_used": len(seq_full_or_mature),
                "major_group": major,
                "cysteine_group": cysteine,
            }
        )
        ann_rows_mature.append(
            {
                "identifier": identifier,
                "sp_in_embedding": "no",
                "sequence_source": "mature_seq",
                "sequence_length_used": len(seq_mature),
                "major_group": major,
                "cysteine_group": cysteine,
            }
        )

    ensure_dirs(paths.inputs_3ftx)
    fasta_full = paths.inputs_3ftx / "3ftx_full_or_mature.fasta"
    fasta_mature = paths.inputs_3ftx / "3ftx_mature_only.fasta"
    ann_full = paths.inputs_3ftx / "3ftx_annotations_full_or_mature.csv"
    ann_mature = paths.inputs_3ftx / "3ftx_annotations_mature_only.csv"
    summary = paths.inputs_3ftx / "3ftx_preparation_summary.txt"

    write_fasta(records_full, fasta_full)
    write_fasta(records_mature, fasta_mature)
    pd.DataFrame(ann_rows_full).to_csv(ann_full, index=False)
    pd.DataFrame(ann_rows_mature).to_csv(ann_mature, index=False)

    for annotation_path in [ann_full, ann_mature]:
        header = annotation_path.read_text().splitlines()[0].split(",")
        if header.count("identifier") != 1:
            raise RuntimeError(f"Expected exactly one identifier column in {annotation_path}; header={header}")

    full_count = sum(1 for r in ann_rows_full if r["sp_in_embedding"] == "yes")
    summary.write_text(
        "3FTx final optimal-story input preparation\n"
        f"Excel: {paths.data_3ftx_xlsx}\n"
        f"Rows read: {len(df)}\n"
        f"Sequences written: {len(records_full)}\n"
        f"full_or_mature uses full_seq: {full_count}\n"
        f"full_or_mature uses mature fallback: {len(records_full) - full_count}\n"
        f"mature_only sequences: {len(records_mature)}\n"
        f"full_col: {full_col}\n"
        f"mature_col: {mature_col}\n"
        f"major_group_col: {major_col}\n"
        f"cysteine_group_col: {cysteine_col}\n"
        "main_3ftx_slide: one SP-in-embedding problem plot plus three major_group outcome plots\n"
    )
    print(summary.read_text())
    return {
        "fasta_full": fasta_full,
        "fasta_mature": fasta_mature,
        "ann_full": ann_full,
        "ann_mature": ann_mature,
        "summary": summary,
    }


def protspace_embed(fasta: Path, out_dir: Path, *, cwd: Path, dry_run: bool = False) -> Path:
    run_cmd(["protspace", "embed", "-i", str(fasta), "-e", "prot_t5", "-o", str(out_dir)], cwd=cwd, dry_run=dry_run)
    return out_dir / "prot_t5.h5"


def safe_h5_id(identifier: str) -> str:
    sid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(identifier))
    return sid or "id"


def load_h5_rows(path: Path, ids: Sequence[str]) -> np.ndarray:
    rows = []
    with h5py.File(path, "r") as h5:
        for identifier in ids:
            key = str(identifier)
            if key not in h5:
                alt = safe_h5_id(key)
                if alt in h5:
                    key = alt
                else:
                    raise KeyError(f"Identifier {identifier!r} not found in {path}")
            arr = np.asarray(h5[key], dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError(f"Embedding {identifier!r} in {path} is not 1D: {arr.shape}")
            rows.append(arr)
    return np.vstack(rows)


def write_h5_matrix(path: Path, ids: Sequence[str], X: np.ndarray, *, attrs: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {X.shape}")
    if len(ids) != X.shape[0]:
        raise ValueError(f"ids length {len(ids)} != matrix rows {X.shape[0]}")
    seen: dict[str, int] = {}
    with h5py.File(path, "w") as h5:
        if attrs:
            for k, v in attrs.items():
                h5.attrs[k] = v
        for identifier, vec in zip(ids, X):
            sid = safe_h5_id(identifier)
            if sid in seen:
                seen[sid] += 1
                sid = f"{sid}__dup{seen[sid]}"
            else:
                seen[sid] = 0
            h5.create_dataset(sid, data=np.asarray(vec, dtype=np.float32))


def build_3ftx_manual_delta_background(paths: Paths, files: dict[str, Path]) -> Path:
    h5_full = paths.emb_3ftx / "full_or_mature" / "prot_t5.h5"
    h5_mature = paths.emb_3ftx / "mature_only" / "prot_t5.h5"
    ann = pd.read_csv(files["ann_full"])
    paired = ann[ann["sp_in_embedding"].astype(str).str.lower().eq("yes")].copy()
    ids = paired["identifier"].astype(str).tolist()
    if not ids:
        raise ValueError("No full_seq rows found for paired-delta background")
    X_full = load_h5_rows(h5_full, ids)
    X_mature = load_h5_rows(h5_mature, ids)
    delta = X_full - X_mature
    B = np.vstack([delta, -delta])
    bg_ids = [f"plus_delta_{x}" for x in ids] + [f"minus_delta_{x}" for x in ids]
    out = paths.inputs_3ftx / "3ftx_manual_paired_delta_background.h5"
    write_h5_matrix(
        out,
        bg_ids,
        B,
        attrs={
            "background_kind": "manual_paired_delta_full_minus_mature",
            "n_pairs": int(len(ids)),
            "n_background_rows": int(B.shape[0]),
            "n_features": int(B.shape[1]),
        },
    )
    summary = paths.inputs_3ftx / "3ftx_manual_delta_background_summary.txt"
    trace_ratio_hint = float(np.sum(delta * delta) / max(np.sum(X_full * X_full), 1e-12))
    summary.write_text(
        "3FTx manual paired-delta background\n"
        f"pairs_used: {len(ids)}\n"
        f"background_rows: {B.shape[0]}\n"
        f"features: {B.shape[1]}\n"
        "definition: B = [+Δ; -Δ], Δ = E(full_i) - E(mature_i) for proteins with full_seq\n"
        f"delta_energy_over_full_embedding_energy_hint: {trace_ratio_hint:.6g}\n"
        f"background_h5: {out}\n"
    )
    print(summary.read_text())
    return out


# embed both
# run PCA + ρPCA
def protspace_prepare_3ftx(paths: Paths, files: dict[str, Path], *, dry_run: bool = False) -> dict[str, Path]:
    h5_full = paths.emb_3ftx / "full_or_mature" / "prot_t5.h5"
    h5_mature = paths.emb_3ftx / "mature_only" / "prot_t5.h5"
    unified_run = paths.prep_3ftx / "full_or_mature_pca_rhopca_unified"
    mature_run = paths.prep_3ftx / "mature_only_pca_positive_control"
    manual_run = paths.prep_3ftx / "full_or_mature_rhopca_manual_delta"

    run_cmd(
        [
            "protspace", "prepare",
            "-i", f"{h5_full}:prot_t5",
            "-f", str(files["fasta_full"]),
            "-a", "default",
            "-a", str(files["ann_full"]),
            "-m", "pca2,umap2,ppca2",
            "--nuisance", "sp_in_embedding:type=binary;missing=zero;scale=0.5;cross_fit=5",
            "-o", str(unified_run),
            "--keep-tmp",
            "--refetch", "all",
            "-v",
        ],
        cwd=paths.root,
        dry_run=dry_run,
    )
    run_cmd(
        [
            "protspace", "prepare",
            "-i", f"{h5_mature}:prot_t5",
            "-f", str(files["fasta_mature"]),
            "-a", "default",
            "-a", str(files["ann_mature"]),
            "-m", "pca2,umap2",
            "-o", str(mature_run),
            "--keep-tmp",
            "--refetch", "all",
            "-v",
        ],
        cwd=paths.root,
        dry_run=dry_run,
    )

    if dry_run:
        manual_bg = paths.inputs_3ftx / "3ftx_manual_paired_delta_background.h5"
    else:
        manual_bg = build_3ftx_manual_delta_background(paths, files)
    run_cmd(
        [
            "protspace", "prepare",
            "-i", f"{h5_full}:prot_t5",
            "-f", str(files["fasta_full"]),
            "-a", "default",
            "-a", str(files["ann_full"]),
            "-m", "ppca2",
            "--ppca-background", str(manual_bg),
            "-o", str(manual_run),
            "--keep-tmp",
            "--refetch", "all",
            "-v",
        ],
        cwd=paths.root,
        dry_run=dry_run,
    )
    return {"unified_run": unified_run, "mature_run": mature_run, "manual_run": manual_run, "manual_background": manual_bg}


def load_npz_coords(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    for key in ["data", "projection", "coordinates", "coords", "embedding", "X", "arr_0"]:
        if key in obj:
            arr = np.asarray(obj[key])
            if arr.ndim == 2 and arr.shape[1] >= 2 and np.issubdtype(arr.dtype, np.number):
                return arr[:, :2].astype(float)
    for key in obj.files:
        arr = np.asarray(obj[key])
        if arr.ndim == 2 and arr.shape[1] >= 2 and np.issubdtype(arr.dtype, np.number):
            return arr[:, :2].astype(float)
    raise ValueError(f"Could not find a numeric 2D coordinate array in {path}; keys={obj.files}")


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern!r} in {root}")
    if len(matches) > 1:
        print(f"[warn] Multiple matches for {pattern!r}; using {matches[0]}")
    return matches[0]


def clean_label_series(s: pd.Series) -> pd.Series:
    return s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan, "NaN": np.nan})


def drop_rare_for_metrics(labels: pd.Series, min_count: int = 2) -> pd.Series:
    labels = clean_label_series(labels.astype("object"))
    counts = labels.value_counts(dropna=True)
    keep = set(counts[counts >= min_count].index)
    return labels.where(labels.isin(keep), np.nan)


def label_eta2(coords: np.ndarray, labels: np.ndarray) -> float:
    xy = np.asarray(coords, dtype=float)
    if len(xy) < 2:
        return float("nan")
    grand = xy.mean(axis=0)
    total = float(np.sum((xy - grand) ** 2))
    if total <= 0:
        return float("nan")
    between = 0.0
    for lab in pd.unique(labels):
        group = xy[labels == lab]
        if len(group) == 0:
            continue
        center = group.mean(axis=0)
        between += len(group) * float(np.sum((center - grand) ** 2))
    return between / total


def compute_label_metrics(coords: np.ndarray, labels: pd.Series, *, random_state: int = 42) -> dict[str, object]:
    labels = drop_rare_for_metrics(labels)
    valid = labels.notna().to_numpy()
    xy = coords[valid]
    lab = labels[valid].astype(str).to_numpy()
    n_classes = len(pd.unique(lab))
    out: dict[str, object] = {"n_points": len(xy), "n_label_classes": n_classes}
    if n_classes < 2 or len(xy) <= n_classes:
        out.update({c: np.nan for c in METRIC_COLUMNS})
        return out
    try:
        out["label_silhouette"] = float(silhouette_score(xy, lab))
    except Exception:
        out["label_silhouette"] = np.nan
    try:
        out["label_eta2"] = float(label_eta2(xy, lab))
    except Exception:
        out["label_eta2"] = np.nan
    try:
        pred = KMeans(n_clusters=n_classes, random_state=random_state, n_init=20).fit_predict(xy)
        out["kmeans_nmi"] = float(normalized_mutual_info_score(lab, pred))
    except Exception:
        out["kmeans_nmi"] = np.nan
    return out


def zoom_limits(coords: np.ndarray, q: float = 0.01) -> tuple[tuple[float, float], tuple[float, float]]:
    xy = np.asarray(coords, dtype=float)
    x0, x1 = np.nanquantile(xy[:, 0], [q, 1 - q])
    y0, y1 = np.nanquantile(xy[:, 1], [q, 1 - q])
    dx = max(float(x1 - x0), 1e-9)
    dy = max(float(y1 - y0), 1e-9)
    return (float(x0 - 0.04 * dx), float(x1 + 0.04 * dx)), (float(y0 - 0.04 * dy), float(y1 + 0.04 * dy))


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.16, linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_categorical_clean(
    coords: np.ndarray,
    labels: pd.Series,
    out_path: Path,
    *,
    color_map: dict[str, str],
    axis_prefix: str,
    zoom_q: float = 0.01,
    marker_size: float = 9,
    alpha: float = 0.82,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = clean_label_series(labels.astype("object")).fillna("__missing__").astype(str)
    fig, ax = plt.subplots(figsize=(5.2, 4.45))
    for lab in sorted(pd.unique(labels), key=lambda x: str(x)):
        mask = labels.eq(lab).to_numpy()
        color = color_map.get(str(lab), "#8c8c8c")
        ax.scatter(coords[mask, 0], coords[mask, 1], s=marker_size, alpha=alpha, linewidths=0, c=color)
    ax.set_xlabel(f"{axis_prefix} component 1", fontsize=9)
    ax.set_ylabel(f"{axis_prefix} component 2", fontsize=9)
    xlim, ylim = zoom_limits(coords, q=zoom_q)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    style_axes(ax)
    fig.savefig(out_path, dpi=350, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_continuous_clean(
    coords: np.ndarray,
    values: pd.Series,
    out_path: Path,
    *,
    axis_prefix: str,
    vmin: float,
    vmax: float,
    zoom_q: float = 0.01,
    marker_size: float = 7,
    alpha: float = 0.82,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna().to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.45))
    ax.scatter(coords[valid, 0], coords[valid, 1], c=numeric[valid], s=marker_size, alpha=alpha, linewidths=0, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel(f"{axis_prefix} component 1", fontsize=9)
    ax.set_ylabel(f"{axis_prefix} component 2", fontsize=9)
    xlim, ylim = zoom_limits(coords[valid], q=zoom_q)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    style_axes(ax)
    fig.savefig(out_path, dpi=350, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_categorical_legend(path: Path, labels_and_colors: Sequence[tuple[str, str]], *, title: str, ncol: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handles = [Line2D([0], [0], marker="o", linestyle="", color=color, label=label, markersize=7) for label, color in labels_and_colors]
    width = 2.2 if ncol == 1 else 6.0
    height = max(0.55, 0.35 * math.ceil(len(handles) / ncol) + 0.38)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    ax.legend(handles=handles, title=title, loc="center", frameon=False, ncol=ncol, fontsize=8, title_fontsize=9)
    fig.savefig(path, dpi=350, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_continuous_legend(path: Path, *, label: str, vmin: float, vmax: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.0, 0.42))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), cax=ax, orientation="horizontal")
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=350, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_metric_card(path: Path, title: str, df: pd.DataFrame, *, max_rows: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    show = df.head(max_rows).copy()
    cols = [c for c in ["plot_label", "label_column", "label_silhouette", "label_eta2", "kmeans_nmi"] if c in show.columns]
    show = show[cols]
    for c in ["label_silhouette", "label_eta2", "kmeans_nmi"]:
        if c in show.columns:
            show[c] = show[c].map(lambda x: "N/A" if pd.isna(x) else f"{float(x):.3f}")
    fig, ax = plt.subplots(figsize=(7.0, 0.62 + 0.32 * max(len(show), 1)))
    ax.axis("off")
    ax.text(0.0, 1.02, title, fontsize=10, fontweight="bold", va="bottom")
    table = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.15)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def load_annotations_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "identifier" not in df.columns:
        raise KeyError(f"Missing identifier column in {path}")
    return df


# create the four 3FTx slide plots + metrics
def make_3ftx_plots(paths: Paths, files: dict[str, Path]) -> pd.DataFrame:
    unified_run = paths.prep_3ftx / "full_or_mature_pca_rhopca_unified"
    manual_run = paths.prep_3ftx / "full_or_mature_rhopca_manual_delta"
    mature_run = paths.prep_3ftx / "mature_only_pca_positive_control"

    full_pca = load_npz_coords(find_one(unified_run / "tmp", "proj_*_pca2_*.npz"))
    full_umap = load_npz_coords(find_one(unified_run / "tmp", "proj_*_umap2_*.npz"))
    unified_ppca = load_npz_coords(find_one(unified_run / "tmp", "proj_*_ppca2_*.npz"))
    manual_ppca = load_npz_coords(find_one(manual_run / "tmp", "proj_*_ppca2_*.npz"))
    mature_pca = load_npz_coords(find_one(mature_run / "tmp", "proj_*_pca2_*.npz"))
    mature_umap = load_npz_coords(find_one(mature_run / "tmp", "proj_*_umap2_*.npz"))

    ann_full = load_annotations_csv(files["ann_full"])
    ann_mature = load_annotations_csv(files["ann_mature"])

    sp_colors = {"no": SP_COLORS["no"], "yes": SP_COLORS["yes"]}
    write_categorical_legend(
        paths.figs_3ftx / "legend_3ftx_sp_in_embedding.png",
        [("SP absent", SP_COLORS["no"]), ("SP present", SP_COLORS["yes"])],
        title="SP in embedding",
        ncol=2,
    )

    rows: list[dict[str, object]] = []

    # Main slide plot 1: show the nuisance in the mixed full-or-mature input.
    metrics = compute_label_metrics(full_pca, ann_full["sp_in_embedding"])
    plot_categorical_clean(
        full_pca,
        ann_full["sp_in_embedding"],
        paths.figs_3ftx / "01_3ftx_PCA_full_or_mature__sp.png",
        color_map=sp_colors,
        axis_prefix="PCA",
    )
    row = {"plot": "01_3ftx_PCA_full_or_mature__sp.png", "plot_label": "PCA full-or-mature", "experiment": "3ftx", "method": "pca2", "label_column": "sp_in_embedding"}
    row.update(metrics)
    rows.append(row)
    print(f"Wrote {paths.figs_3ftx / '01_3ftx_PCA_full_or_mature__sp.png'}")

    # Main slide plots 2-4: show biological major-group organization.
    if "major_group" not in ann_full.columns or "major_group" not in ann_mature.columns:
        raise KeyError("major_group is required for the optimal 3FTx main-slide story")

    all_major = pd.concat([ann_full["major_group"], ann_mature["major_group"]], ignore_index=True).astype(str)
    cats = all_major.value_counts().index.tolist()
    cmap = plt.get_cmap("tab10", max(len(cats), 1))
    major_colors = {cat: cmap(i) for i, cat in enumerate(cats)}
    write_categorical_legend(
        paths.figs_3ftx / "legend_3ftx_major_group.png",
        [(cat, major_colors[cat]) for cat in cats],
        title="major group",
        ncol=2,
    )

    main_major_specs = [
        ("02_3ftx_PCA_mature_only_positive_control__major_group.png", "PCA mature-only positive control", mature_pca, ann_mature, "pca2", "PCA"),
        ("03_3ftx_rhoPCA_manual_delta__major_group.png", "ρPCA manual Δ", manual_ppca, ann_full, "ppca2_manual_delta", "ρPCA"),
        ("04_3ftx_rhoPCA_unified_annotation__major_group.png", "ρPCA unified annotation", unified_ppca, ann_full, "ppca2_unified", "ρPCA"),
    ]
    for filename, plot_label, coords, ann, method, axis_prefix in main_major_specs:
        metrics = compute_label_metrics(coords, ann["major_group"])
        plot_categorical_clean(coords, ann["major_group"], paths.figs_3ftx / filename, color_map=major_colors, axis_prefix=axis_prefix)
        row = {"plot": filename, "plot_label": plot_label, "experiment": "3ftx", "method": method, "label_column": "major_group"}
        row.update(metrics)
        rows.append(row)
        print(f"Wrote {paths.figs_3ftx / filename}")

    # Backup SP plots, including UMAP baselines, for discussion / backup slides only.
    backup_sp_specs = [
        ("backup_00a_3ftx_UMAP_full_or_mature__sp.png", "UMAP full-or-mature", full_umap, ann_full, "umap2", "UMAP"),
        ("backup_00b_3ftx_UMAP_mature_only__sp.png", "UMAP mature-only", mature_umap, ann_mature, "umap2", "UMAP"),
        ("backup_01_3ftx_PCA_mature_only_positive_control__sp.png", "PCA mature-only", mature_pca, ann_mature, "pca2", "PCA"),
        ("backup_02_3ftx_rhoPCA_manual_delta__sp.png", "ρPCA manual Δ", manual_ppca, ann_full, "ppca2_manual_delta", "ρPCA"),
        ("backup_03_3ftx_rhoPCA_unified_annotation__sp.png", "ρPCA unified", unified_ppca, ann_full, "ppca2_unified", "ρPCA"),
    ]
    for filename, plot_label, coords, ann, method, axis_prefix in backup_sp_specs:
        metrics = compute_label_metrics(coords, ann["sp_in_embedding"])
        plot_categorical_clean(coords, ann["sp_in_embedding"], paths.backup_3ftx / filename, color_map=sp_colors, axis_prefix=axis_prefix)
        row = {"plot": filename, "plot_label": plot_label, "experiment": "3ftx_backup", "method": method, "label_column": "sp_in_embedding"}
        row.update(metrics)
        rows.append(row)
        print(f"Wrote {paths.backup_3ftx / filename}")

    # Backup major-group plots, including PCA full-or-mature and UMAP baselines.
    write_categorical_legend(paths.backup_3ftx / "legend_3ftx_major_group.png", [(cat, major_colors[cat]) for cat in cats], title="major group", ncol=2)
    backup_major_specs = [
        ("backup_04_3ftx_PCA_full_or_mature__major_group.png", "PCA full-or-mature", full_pca, ann_full, "pca2", "PCA"),
        ("backup_05_3ftx_UMAP_full_or_mature__major_group.png", "UMAP full-or-mature", full_umap, ann_full, "umap2", "UMAP"),
        ("backup_06_3ftx_UMAP_mature_only__major_group.png", "UMAP mature-only", mature_umap, ann_mature, "umap2", "UMAP"),
    ]
    for filename, plot_label, coords, ann, method, axis_prefix in backup_major_specs:
        metrics = compute_label_metrics(coords, ann["major_group"])
        plot_categorical_clean(coords, ann["major_group"], paths.backup_3ftx / filename, color_map=major_colors, axis_prefix=axis_prefix)
        row = {"plot": filename, "plot_label": plot_label, "experiment": "3ftx_backup", "method": method, "label_column": "major_group"}
        row.update(metrics)
        rows.append(row)
        print(f"Wrote {paths.backup_3ftx / filename}")

    summary = pd.DataFrame(rows)
    summary_path = paths.figs_3ftx / "3ftx_optimal_story_metric_summary.csv"
    ensure_dirs(paths.figs_3ftx, paths.scores_3ftx)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(paths.scores_3ftx / "3ftx_optimal_story_metrics.csv", index=False)
    write_metric_card(paths.figs_3ftx / "metrics_card_3ftx_main.png", "3FTx main-slide metrics", summary[summary["experiment"].eq("3ftx")])
    write_metric_card(paths.backup_3ftx / "metrics_card_3ftx_backup.png", "3FTx backup metrics", summary[summary["experiment"].eq("3ftx_backup")], max_rows=12)
    print(f"Wrote {summary_path}")
    return summary


def urlretrieve_text(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "protspace-rhopca-final-slide-results/1.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out.write_bytes(r.read())


def build_uniprot_stream_url(*, query: str, fmt: str, fields: str | None = None) -> str:
    params = {"query": query, "format": fmt, "compressed": "false"}
    if fields:
        params["fields"] = fields
    return "https://rest.uniprot.org/uniprotkb/stream?" + urllib.parse.urlencode(params)


# fetch reviewed toxin FASTA from UniProt
def fetch_toxprot_fasta(paths: Paths) -> dict[str, Path]:
    fasta_full = paths.data_tox / "toxprot_reviewed_toxin_full.fasta"
    meta_tsv = paths.data_tox / "toxprot_reviewed_toxin_metadata.tsv"
    manifest = paths.data_tox / "toxprot_download_manifest.txt"
    query = "(reviewed:true) AND (keyword:KW-0800)"
    fasta_url = build_uniprot_stream_url(query=query, fmt="fasta")
    meta_url = build_uniprot_stream_url(query=query, fmt="tsv", fields="accession,id,protein_name,organism_name,length")
    print(f"Downloading FASTA: {fasta_url}")
    urlretrieve_text(fasta_url, fasta_full)
    print(f"Downloading metadata: {meta_url}")
    urlretrieve_text(meta_url, meta_tsv)
    n = count_fasta_records(fasta_full)
    manifest.write_text(
        "ToxProt final slide UniProt download\n"
        f"Query: {query}\n"
        f"FASTA: {fasta_full}\n"
        f"Metadata: {meta_tsv}\n"
        f"Downloaded records: {n}\n"
        f"Downloaded at unix_time: {time.time()}\n"
    )
    print(manifest.read_text())
    return {"fasta_full": fasta_full, "meta_tsv": meta_tsv, "manifest": manifest}


def count_fasta_records(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open() if line.startswith(">"))


# rewrite headers to accession-only IDs
def rewrite_uniprot_fasta_headers(fasta_full: Path, clean_fasta: Path) -> Path:
    acc_re = re.compile(r"^(sp|tr)\|([^|]+)\|")
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq: list[str] = []

    def flush() -> None:
        nonlocal header, seq
        if header is None:
            return
        token = header.split()[0]
        m = acc_re.match(token)
        acc = m.group(2) if m else token
        sequence = clean_sequence("".join(seq))
        if acc and sequence:
            records.append((acc, sequence))
        header = None
        seq = []

    with fasta_full.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
            else:
                seq.append(line)
        flush()

    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for acc, sequence in records:
        if acc in seen:
            continue
        seen.add(acc)
        unique.append((acc, sequence))
    write_fasta(unique, clean_fasta)
    print(f"Read records: {len(records)}")
    print(f"Unique accession records written: {len(unique)}")
    print(f"Wrote {clean_fasta}")
    return clean_fasta


# embed successfully retrieved sequences
def protspace_annotation_check_tox(paths: Paths, h5: Path, clean_fasta: Path, *, dry_run: bool = False) -> Path:
    check_run = paths.prep_tox / "toxprot_annotation_check_clean_ids"
    run_cmd(
        [
            "protspace", "prepare",
            "-i", f"{h5}:prot_t5",
            "-f", str(clean_fasta),
            "-a", "default",
            "-m", "pca2",
            "-o", str(check_run),
            "--keep-tmp",
            "--refetch", "all",
            "-v",
        ],
        cwd=paths.root,
        dry_run=dry_run,
    )
    return check_run


def fetch_json(url: str, timeout: int = 30) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "protspace-rhopca-taxonomy-order/1.1", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "protspace-rhopca-taxonomy-order/1.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def order_from_uniprot_taxonomy(taxid: str) -> str | None:
    data = fetch_json(f"https://rest.uniprot.org/taxonomy/{taxid}.json")

    def walk(obj: object) -> str | None:
        if isinstance(obj, dict):
            rank = str(obj.get("rank", "")).lower()
            name = obj.get("scientificName") or obj.get("scientific_name") or obj.get("name")
            if rank == "order" and name:
                return str(name)
            for v in obj.values():
                found = walk(v)
                if found:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = walk(item)
                if found:
                    return found
        return None

    return walk(data)


def order_from_ncbi_taxonomy(taxid: str) -> str | None:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi" f"?db=taxonomy&id={taxid}&retmode=xml"
    root = ET.fromstring(fetch_text(url))
    for taxon in root.findall(".//Taxon"):
        if taxon.findtext("Rank") == "order" and taxon.findtext("ScientificName"):
            return taxon.findtext("ScientificName")
    for taxon in root.findall(".//LineageEx/Taxon"):
        if taxon.findtext("Rank") == "order" and taxon.findtext("ScientificName"):
            return taxon.findtext("ScientificName")
    return None


def resolve_order(taxid: str, retries: int = 3) -> tuple[str, str]:
    taxid = str(taxid).strip()
    if not taxid or taxid.lower() in {"nan", "none", ""}:
        return MISSING_ORDER, "missing_taxid"
    for attempt in range(retries):
        try:
            order = order_from_uniprot_taxonomy(taxid)
            if order:
                return order, "uniprot_taxonomy"
        except Exception:
            pass
        try:
            order = order_from_ncbi_taxonomy(taxid)
            if order:
                return order, "ncbi_taxonomy"
        except Exception:
            pass
        time.sleep(0.5 + attempt)
    return MISSING_ORDER, "unresolved"


# fetch/build taxonomy order annotation
def build_tox_taxonomy_order_annotations(paths: Paths, check_run: Path, *, sleep_s: float = 0.12) -> Path:
    ann_path = check_run / "tmp" / "all_annotations.parquet"
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing annotation check parquet: {ann_path}")
    df = pd.read_parquet(ann_path)
    if "identifier" not in df.columns or "organism_id" not in df.columns:
        raise KeyError(f"Need identifier and organism_id columns in {ann_path}; got {df.columns.tolist()}")

    cache_path = paths.data_tox / "taxonomy_order_cache.json"
    out = paths.data_tox / "toxprot_taxonomy_order_annotations.csv"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    else:
        cache = {}

    taxids = df["organism_id"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()
    print(f"Input proteins: {len(df)}")
    print(f"Unique organism_id values: {len(taxids)}")
    print(f"Cached taxonomy records: {len(cache)}")

    for i, taxid in enumerate(taxids, start=1):
        if taxid in cache:
            continue
        order, source = resolve_order(taxid)
        cache[taxid] = {"order": order, "source": source}
        if i % 25 == 0:
            cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
            print(f"Resolved {i}/{len(taxids)} taxonomy IDs...")
        time.sleep(sleep_s)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))

    out_df = df[["identifier", "organism_id"]].copy()
    out_df["organism_id"] = out_df["organism_id"].astype(str)
    out_df["order"] = out_df["organism_id"].map(lambda x: cache.get(str(x), {}).get("order", MISSING_ORDER))
    out_df["taxonomy_order"] = out_df["order"]
    out_df["tax_order"] = out_df["order"]
    out_df["taxonomy_order_source"] = out_df["organism_id"].map(lambda x: cache.get(str(x), {}).get("source", "missing"))
    out_df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print("Top orders:")
    print(out_df["order"].value_counts(dropna=False).head(30).to_string())
    return out


# run PCA + ρPCA with length + order nuisance
def protspace_prepare_tox(paths: Paths, h5: Path, clean_fasta: Path, order_csv: Path, *, dry_run: bool = False) -> Path:
    final_run = paths.prep_tox / "toxprot_pca_rhopca_length_order_clean_ids"
    run_cmd(
        [
            "protspace", "prepare",
            "-i", f"{h5}:prot_t5",
            "-f", str(clean_fasta),
            "-a", "default",
            "-a", str(order_csv),
            "-m", "pca2,umap2,ppca2",
            "--nuisance", "length:type=continuous;transform=log1p;basis=spline;n_knots=6;scale=0.5;cross_fit=5",
            "--nuisance", "order:type=categorical;missing=category;min_count=10;scale=0.5;cross_fit=5",
            "-o", str(final_run),
            "--keep-tmp",
            "--refetch", "all",
            "-v",
        ],
        cwd=paths.root,
        dry_run=dry_run,
    )
    return final_run


def merge_order_into_tox_plot_annotations(final_run: Path, order_csv: Path) -> pd.DataFrame:
    ann_path = final_run / "tmp" / "all_annotations.parquet"
    df = pd.read_parquet(ann_path)
    order = pd.read_csv(order_csv)
    needed = ["identifier", "order", "taxonomy_order", "tax_order", "taxonomy_order_source"]
    missing = [c for c in needed if c not in order.columns]
    if missing:
        raise KeyError(f"Order CSV missing columns: {missing}")
    drop = [c for c in needed if c != "identifier" and c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    merged = df.merge(order[needed], on="identifier", how="left", validate="one_to_one")
    merged.to_parquet(ann_path, index=False)
    print(f"Merged order into {ann_path}")
    print("Top orders after merge:")
    print(merged["order"].value_counts(dropna=False).head(20).to_string())
    return merged


def make_length_bin(values: pd.Series, q: int = 5) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    labels = [f"Q{i + 1}" for i in range(q)]
    try:
        return pd.qcut(numeric, q=q, labels=labels, duplicates="drop").astype("object")
    except Exception:
        return pd.cut(numeric, bins=q, labels=labels).astype("object")


def collapse_categories(s: pd.Series, min_count: int = 25, top_k: int = 11) -> pd.Series:
    s = clean_label_series(s.astype("object")).fillna("__missing__").astype(str)
    counts = s.value_counts()
    keep = set(counts[counts >= min_count].head(top_k).index)
    return s.where(s.isin(keep), "__other__")


# create the four slide plots + metrics
def make_tox_plots(paths: Paths, final_run: Path, order_csv: Path) -> pd.DataFrame:
    ann = merge_order_into_tox_plot_annotations(final_run, order_csv)
    pca = load_npz_coords(find_one(final_run / "tmp", "proj_*_pca2_*.npz"))
    umap = load_npz_coords(find_one(final_run / "tmp", "proj_*_umap2_*.npz"))
    ppca = load_npz_coords(find_one(final_run / "tmp", "proj_*_ppca2_*.npz"))
    if len(pca) != len(ann) or len(umap) != len(ann) or len(ppca) != len(ann):
        raise ValueError(f"Coordinate rows and annotation rows differ: pca={len(pca)}, umap={len(umap)}, ppca={len(ppca)}, ann={len(ann)}")

    length_col = first_existing_col(ann, ["length", "sequence_length"])
    order_col = first_existing_col(ann, ["order", "taxonomy_order", "tax_order"])
    length_bin = make_length_bin(ann[length_col], q=5)
    order_plot = collapse_categories(ann[order_col], min_count=25, top_k=11)
    numeric_length = pd.to_numeric(ann[length_col], errors="coerce")
    vmin, vmax = np.nanquantile(numeric_length, [0.01, 0.99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(numeric_length)), float(np.nanmax(numeric_length))

    order_cats = order_plot.value_counts().index.tolist()
    cmap = plt.get_cmap("tab20", max(len(order_cats), 1))
    order_colors = {cat: cmap(i) for i, cat in enumerate(order_cats)}
    write_continuous_legend(paths.figs_tox / "legend_toxprot_length.png", label="sequence length", vmin=float(vmin), vmax=float(vmax))
    write_categorical_legend(paths.figs_tox / "legend_toxprot_order.png", [(cat, order_colors[cat]) for cat in order_cats], title="taxonomy order", ncol=3)

    specs = [
        ("01_toxprot_PCA__length.png", "PCA length", pca, "pca2", "length_bin", length_bin, "PCA", "length"),
        ("02_toxprot_PCA__order.png", "PCA order", pca, "pca2", "order_plot", order_plot, "PCA", "order"),
        ("03_toxprot_rhoPCA_length_order__length.png", "ρPCA length", ppca, "ppca2", "length_bin", length_bin, "ρPCA", "length"),
        ("04_toxprot_rhoPCA_length_order__order.png", "ρPCA order", ppca, "ppca2", "order_plot", order_plot, "ρPCA", "order"),
    ]

    rows: list[dict[str, object]] = []
    for filename, plot_label, coords, method, label_column, labels, axis_prefix, display_label in specs:
        metrics = compute_label_metrics(coords, labels)
        if display_label == "length":
            plot_continuous_clean(coords, numeric_length, paths.figs_tox / filename, axis_prefix=axis_prefix, vmin=float(vmin), vmax=float(vmax), marker_size=5.5, alpha=0.76)
        else:
            plot_categorical_clean(coords, labels, paths.figs_tox / filename, color_map=order_colors, axis_prefix=axis_prefix, marker_size=5.5, alpha=0.76)
        row = {"plot": filename, "plot_label": plot_label, "experiment": "toxprot", "method": method, "label_column": label_column, "display_label": display_label}
        row.update(metrics)
        rows.append(row)
        print(f"Wrote {paths.figs_tox / filename}")

    # Backup UMAP baseline plots for discussion / backup slides only.
    backup_specs = [
        ("backup_01_toxprot_UMAP__length.png", "UMAP length", umap, "umap2", "length_bin", length_bin, "UMAP", "length"),
        ("backup_02_toxprot_UMAP__order.png", "UMAP order", umap, "umap2", "order_plot", order_plot, "UMAP", "order"),
    ]
    for filename, plot_label, coords, method, label_column, labels, axis_prefix, display_label in backup_specs:
        metrics = compute_label_metrics(coords, labels)
        if display_label == "length":
            plot_continuous_clean(coords, numeric_length, paths.backup_tox / filename, axis_prefix=axis_prefix, vmin=float(vmin), vmax=float(vmax), marker_size=5.5, alpha=0.76)
        else:
            plot_categorical_clean(coords, labels, paths.backup_tox / filename, color_map=order_colors, axis_prefix=axis_prefix, marker_size=5.5, alpha=0.76)
        row = {"plot": filename, "plot_label": plot_label, "experiment": "toxprot_backup", "method": method, "label_column": label_column, "display_label": display_label}
        row.update(metrics)
        rows.append(row)
        print(f"Wrote {paths.backup_tox / filename}")

    summary = pd.DataFrame(rows)
    summary_path = paths.figs_tox / "toxprot_optimal_story_metric_summary.csv"
    ensure_dirs(paths.figs_tox, paths.scores_tox)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(paths.scores_tox / "toxprot_optimal_story_metrics.csv", index=False)
    write_metric_card(paths.figs_tox / "metrics_card_toxprot.png", "ToxProt nuisance metrics: lower is better", summary)
    print(f"Wrote {summary_path}")
    return summary


def write_metrics_readme(paths: Paths) -> None:
    out = paths.root / "results" / "final_slide_metric_notes.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Final slide metric notes\n\n"
        "The final slide plots are clean scatter plots without embedded metric boxes. "
        "Metrics are written to CSV and standalone metric-card PNGs so they can be shown only if useful.\n\n"
        "Metrics used:\n"
        "- Silhouette: direct annotation-label separation in the 2D plot. For nuisance labels, lower is better.\n"
        "- Label variance η²: fraction of 2D variance explained by the annotation label. For nuisance labels, lower is better.\n"
        "- NMI: agreement between KMeans clusters in the 2D plot and the annotation labels. For nuisance labels, lower is better.\n\n"
        "Dropped from the main slide workflow:\n"
        "- Davies-Bouldin, because it overlaps with Silhouette and can explode for highly overlapping groups.\n"
        "- Calinski-Harabasz, because it overlaps with other compactness/separation measures and is unbounded.\n"
        "- ARI, because it overlaps with NMI and is less readable for imbalanced labels.\n\n"
        "3FTx main slide recommendation:\n"
        "Show one SP-in-embedding problem plot (PCA full-or-mature), then compare mature-only PCA, manual paired-delta ρPCA, and unified annotation ρPCA colored by major group. "
        "The manual paired-delta background is the 3FTx-specific best-case control; the unified annotation background is the generalizable ProtSpace method.\n\n"
        "ToxProt main slide recommendation:\n"
        "Show PCA and ρPCA colored by length and taxonomy order. The ρPCA background is built from length + order.\n"
    )
    print(f"Wrote {out}")


def run_3ftx(paths: Paths, *, dry_run: bool = False) -> pd.DataFrame | None:
    # read data/3FTx/full/3FTx_data.xlsx
    log_step("read data/3FTx/full/3FTx_data.xlsx")
    df = read_3ftx_excel(paths)
    # create full_or_mature FASTA
    log_step("create full_or_mature FASTA")
    # create mature_only FASTA
    log_step("create mature_only FASTA")
    # create annotations CSV
    log_step("create annotations CSV")
    files = prepare_3ftx_inputs(paths, df)

    # embed both
    log_step("embed both")
    if not dry_run:
        ensure_dirs(paths.emb_3ftx)
    protspace_embed(files["fasta_full"], paths.emb_3ftx / "full_or_mature", cwd=paths.root, dry_run=dry_run)
    protspace_embed(files["fasta_mature"], paths.emb_3ftx / "mature_only", cwd=paths.root, dry_run=dry_run)

    # run PCA + ρPCA
    log_step("run PCA + ρPCA")
    protspace_prepare_3ftx(paths, files, dry_run=dry_run)

    # create the four 3FTx slide plots + metrics
    log_step("create the four 3FTx slide plots + metrics")
    if dry_run:
        return None
    return make_3ftx_plots(paths, files)


def run_toxprot(paths: Paths, *, dry_run: bool = False) -> pd.DataFrame | None:
    # fetch reviewed toxin FASTA from UniProt
    log_step("fetch reviewed toxin FASTA from UniProt")
    tox_files = fetch_toxprot_fasta(paths)

    # rewrite headers to accession-only IDs
    log_step("rewrite headers to accession-only IDs")
    clean_fasta = paths.data_tox / "toxprot_reviewed_toxin_accession_ids.fasta"
    rewrite_uniprot_fasta_headers(tox_files["fasta_full"], clean_fasta)

    # embed successfully retrieved sequences
    log_step("embed successfully retrieved sequences")
    h5 = protspace_embed(clean_fasta, paths.emb_tox / "toxprot_reviewed_toxin_accession_ids", cwd=paths.root, dry_run=dry_run)
    check_run = protspace_annotation_check_tox(paths, h5, clean_fasta, dry_run=dry_run)

    # fetch/build taxonomy order annotation
    log_step("fetch/build taxonomy order annotation")
    if dry_run:
        order_csv = paths.data_tox / "toxprot_taxonomy_order_annotations.csv"
    else:
        order_csv = build_tox_taxonomy_order_annotations(paths, check_run)

    # run PCA + ρPCA with length + order nuisance
    log_step("run PCA + ρPCA with length + order nuisance")
    final_run = protspace_prepare_tox(paths, h5, clean_fasta, order_csv, dry_run=dry_run)

    # create the four slide plots + metrics
    log_step("create the four slide plots + metrics")
    if dry_run:
        return None
    return make_tox_plots(paths, final_run, order_csv)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate clean final-slide 3FTx and ToxProt PCA/ρPCA results.")
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    ap.add_argument("--only", choices=["all", "3ftx", "toxprot"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running ProtSpace")
    args = ap.parse_args()

    root = args.root.resolve()
    paths = get_paths(root)
    ensure_dirs(
        paths.inputs_3ftx,
        paths.emb_3ftx,
        paths.prep_3ftx,
        paths.figs_3ftx,
        paths.backup_3ftx,
        paths.scores_3ftx,
        paths.data_tox,
        paths.emb_tox,
        paths.prep_tox,
        paths.figs_tox,
        paths.scores_tox,
        paths.backup_tox,
    )
    write_metrics_readme(paths)

    if args.only in {"all", "3ftx"}:
        run_3ftx(paths, dry_run=args.dry_run)
    if args.only in {"all", "toxprot"}:
        run_toxprot(paths, dry_run=args.dry_run)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())