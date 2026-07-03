#!/usr/bin/env python3
"""Compute 5 clustering-style scores for ProtSpace DR coordinates.

Supports the current ProtSpace run layout where cached projections are stored as
``tmp/proj_<embedding>_<method><dims>_<hash>.npz``.  For these NPZ files, the row
order is assumed to match the annotation CSV row order. This is true for the
3FTx final workflow because the FASTA and annotation CSV are generated together.

For each detected 2D coordinate pair and each requested biological label column,
the script clusters the 2D coordinates with KMeans(k = number of label classes)
and reports:
  1. silhouette_score              higher is better
  2. calinski_harabasz_score        higher is better
  3. davies_bouldin_score           lower is better
  4. adjusted_rand_score            higher is better, external vs label
  5. normalized_mutual_info_score   higher is better, external vs label
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table file: {path}")


def find_projection_npz(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".npz" and p.name.startswith("proj_"):
            out.append(p)
        elif p.is_dir():
            out.extend(q for q in p.rglob("proj_*.npz") if q.is_file())
    return sorted(set(out))


def find_tables(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    allowed = {".csv", ".tsv", ".txt", ".parquet", ".pq"}
    for p in paths:
        if p.is_file() and p.suffix.lower() in allowed:
            out.append(p)
        elif p.is_dir():
            for q in p.rglob("*"):
                if q.is_file() and q.suffix.lower() in allowed:
                    # Avoid diagnostics tables; they are not projection tables.
                    if q.name in {"effect_summary.csv", "annotations_aligned.csv"}:
                        continue
                    out.append(q)
    return sorted(set(out))


def infer_method_from_npz_name(path: Path) -> str:
    # Examples:
    #   proj_prot_t5_pca2_381e9a6ba35d.npz  -> pca2
    #   proj_prot_t5_ppca2_d4fd0709685c.npz -> ppca2
    m = re.search(r"_([a-z]+\d+)_[0-9a-f]{6,}\.npz$", path.name)
    if m:
        return m.group(1)
    return path.stem


def normalize_labels(s: pd.Series) -> pd.Series:
    out = s.astype("object").where(s.notna(), np.nan)
    return out.map(
        lambda x: np.nan
        if pd.isna(x) or str(x).strip() == "" or str(x).strip().lower() == "nan"
        else str(x).strip()
    )


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.to_numeric(s, errors="coerce").notna().mean() > 0.95


def numeric_array(df: pd.DataFrame, x: str, y: str) -> np.ndarray:
    return df[[x, y]].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def discover_coordinate_pairs(df: pd.DataFrame, table_name: str) -> list[tuple[str, str, str]]:
    cols = list(df.columns)
    lower = {str(c).lower(): c for c in cols}
    pairs: list[tuple[str, str, str]] = []

    exact_pairs = [
        ("x", "y", table_name),
        ("x_coord", "y_coord", table_name),
        ("dim1", "dim2", table_name),
        ("dim_1", "dim_2", table_name),
        ("component_1", "component_2", table_name),
        ("pc1", "pc2", table_name),
    ]
    for a, b, name in exact_pairs:
        if a in lower and b in lower:
            pairs.append((name, lower[a], lower[b]))

    by_prefix: dict[str, dict[str, str]] = {}
    patterns = [
        (re.compile(r"^(.+?)[_\-. ]x$", re.I), "x"),
        (re.compile(r"^(.+?)[_\-. ]y$", re.I), "y"),
        (re.compile(r"^(.+?)[_\-. ]1$", re.I), "1"),
        (re.compile(r"^(.+?)[_\-. ]2$", re.I), "2"),
        (re.compile(r"^(.+?)[_\-. ]0$", re.I), "0"),
    ]
    for c in cols:
        for pat, key in patterns:
            m = pat.match(str(c))
            if m:
                pref = m.group(1).strip().lower()
                by_prefix.setdefault(pref, {})[key] = c
    for pref, d in by_prefix.items():
        if "x" in d and "y" in d:
            pairs.append((pref, d["x"], d["y"]))
        if "1" in d and "2" in d:
            pairs.append((pref, d["1"], d["2"]))
        elif "0" in d and "1" in d:
            pairs.append((pref, d["0"], d["1"]))

    numeric_cols = [c for c in cols if is_numeric_series(df[c])]
    if len(numeric_cols) == 2:
        pairs.append((table_name, numeric_cols[0], numeric_cols[1]))

    seen = set()
    clean: list[tuple[str, str, str]] = []
    for name, x, y in pairs:
        key = (name, x, y)
        if key in seen:
            continue
        seen.add(key)
        if x in df.columns and y in df.columns and is_numeric_series(df[x]) and is_numeric_series(df[y]):
            clean.append((name, x, y))
    return clean


def parse_coord_arg(raw: str) -> tuple[str, str, str]:
    if ":" not in raw or "," not in raw:
        raise argparse.ArgumentTypeError("--coords must have format method:x_col,y_col")
    name, rest = raw.split(":", 1)
    x, y = rest.split(",", 1)
    return name.strip(), x.strip(), y.strip()


def compute_scores(coords: np.ndarray, true_labels: np.ndarray, random_state: int) -> dict[str, float | int]:
    n = coords.shape[0]
    k = len(pd.Series(true_labels).astype(str).unique())
    if k < 2 or n <= k:
        raise ValueError(f"Need at least 2 classes and n > k; got n={n}, k={k}")
    pred = KMeans(n_clusters=k, random_state=random_state, n_init=50).fit_predict(coords)
    pred_k = len(np.unique(pred))
    if pred_k < 2 or pred_k >= n:
        raise ValueError(f"Invalid predicted cluster count for metrics: {pred_k}")
    return {
        "n_points": n,
        "n_label_classes": k,
        "n_pred_clusters": pred_k,
        "silhouette": float(silhouette_score(coords, pred)),
        "calinski_harabasz": float(calinski_harabasz_score(coords, pred)),
        "davies_bouldin": float(davies_bouldin_score(coords, pred)),
        "adjusted_rand": float(adjusted_rand_score(true_labels, pred)),
        "normalized_mutual_info": float(normalized_mutual_info_score(true_labels, pred)),
    }


def make_npz_frame(path: Path, ann: pd.DataFrame, id_col: str, label_columns: list[str]) -> tuple[pd.DataFrame, tuple[str, str, str]]:
    z = np.load(path, allow_pickle=False)
    if "data" not in z:
        raise ValueError(f"{path} has no 'data' array")
    coords = np.asarray(z["data"])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"{path} data must have shape (n, >=2), got {coords.shape}")
    if len(ann) != coords.shape[0]:
        raise ValueError(
            f"{path} has {coords.shape[0]} rows but annotation table has {len(ann)} rows. "
            "For NPZ scoring, row order must match. Use a projection table with identifiers if available."
        )
    method = infer_method_from_npz_name(path)
    x_col = f"{method}_1"
    y_col = f"{method}_2"
    keep = [id_col] + [c for c in label_columns if c in ann.columns]
    df = ann[keep].copy()
    df[x_col] = coords[:, 0]
    df[y_col] = coords[:, 1]
    return df, (method, x_col, y_col)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", nargs="+", type=Path, required=True, help="Run output dir(s), projection NPZ(s), or table(s)")
    ap.add_argument("--annotations", type=Path, required=True, help="Annotation CSV/TSV/parquet containing identifiers and labels")
    ap.add_argument("--id-col", default="identifier", help="Identifier column used for merging annotations")
    ap.add_argument("--label-column", action="append", required=True, help="Biological label column; repeatable")
    ap.add_argument("--coords", action="append", type=parse_coord_arg, help="Manual coordinate pair, format method:x_col,y_col; repeatable")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    ann = read_table(args.annotations)
    if args.id_col not in ann.columns:
        raise SystemExit(f"Annotation file lacks id column {args.id_col!r}")

    missing_labels = [c for c in args.label_column if c not in ann.columns]
    if missing_labels:
        raise SystemExit(f"Annotation file lacks requested label column(s): {missing_labels}")

    frames: list[tuple[Path, pd.DataFrame, list[tuple[str, str, str]]]] = []

    # Prefer cached NPZ projections if present, because current ProtSpace keeps them with --keep-tmp.
    for path in find_projection_npz(args.input):
        try:
            df, pair = make_npz_frame(path, ann, args.id_col, args.label_column)
            frames.append((path, df, [pair]))
        except Exception as e:
            print(f"[skip] {path}: {e}")

    # Also support flat projection tables if the user passes them explicitly or uses --no-bundled output.
    for path in find_tables(args.input):
        try:
            df = read_table(path)
        except Exception as e:
            print(f"[skip] {path}: cannot read ({e})")
            continue
        if args.id_col in df.columns:
            keep_cols = [args.id_col] + [c for c in args.label_column if c in ann.columns and c not in df.columns]
            if len(keep_cols) > 1:
                df = df.merge(ann[keep_cols], on=args.id_col, how="left")
        coord_pairs = args.coords if args.coords else discover_coordinate_pairs(df, path.stem)
        if coord_pairs:
            frames.append((path, df, coord_pairs))

    rows = []
    for source, df, coord_pairs in frames:
        for method, x_col, y_col in coord_pairs:
            if x_col not in df.columns or y_col not in df.columns:
                print(f"[skip] {source}: missing coords {x_col},{y_col}")
                continue
            coords_all = numeric_array(df, x_col, y_col)
            for label_col in args.label_column:
                if label_col not in df.columns:
                    print(f"[skip] {source}: missing label {label_col!r}")
                    continue
                labels = normalize_labels(df[label_col])
                mask = labels.notna().to_numpy() & np.isfinite(coords_all).all(axis=1)
                coords = coords_all[mask]
                y_true = labels[mask].to_numpy(str)
                try:
                    scores = compute_scores(coords, y_true, args.random_state)
                    rows.append({
                        "source": str(source),
                        "method": method,
                        "x_col": x_col,
                        "y_col": y_col,
                        "label_column": label_col,
                        **scores,
                    })
                except Exception as e:
                    rows.append({
                        "source": str(source),
                        "method": method,
                        "x_col": x_col,
                        "y_col": y_col,
                        "label_column": label_col,
                        "error": str(e),
                    })

    if not rows:
        raise SystemExit(
            "No scorable coordinate/label combinations found. "
            "For the current ProtSpace layout, run with --keep-tmp so tmp/proj_*.npz files exist."
        )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    md = args.out.with_suffix(".md")
    with md.open("w") as f:
        f.write("# DR clustering scores\n\n")
        f.write("Five scores are reported per coordinate pair and label column. Higher is better for silhouette, Calinski-Harabasz, adjusted Rand, and NMI. Lower is better for Davies-Bouldin.\n\n")
        f.write(out.to_markdown(index=False))
        f.write("\n")
    print(f"Wrote {args.out}")
    print(f"Wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())