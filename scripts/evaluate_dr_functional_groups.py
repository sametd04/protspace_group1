"""Evaluate whether DR suppresses length while preserving functional labels.

Input:
  - ProtSpace output directory with projections_data.parquet
  - eval_labels.tsv created from UniProt annotations

Outputs:
  - eval_summary.tsv in the projection output directory

Metrics:
  - length_r2: how well x/y explain sequence length (lower means length is less dominant)
  - label_macro_f1: how well x/y predict a functional label (higher means clearer grouping)
  - label_knn_purity: local same-label neighbor purity (higher means clearer local grouping)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _filter_label(df: pd.DataFrame, label_col: str, min_class_size: int) -> pd.DataFrame:
    out = df.copy()
    out[label_col] = out[label_col].fillna("").astype(str)
    out = out[out[label_col].str.len() > 0]
    counts = out[label_col].value_counts()
    keep = counts[counts >= min_class_size].index
    return out[out[label_col].isin(keep)]


def _macro_f1(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    labels, counts = np.unique(y, return_counts=True)
    if labels.size < 2 or counts.min() < 2:
        return np.nan
    n_splits = min(5, int(counts.min()))
    y_enc = LabelEncoder().fit_transform(y)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, X, y_enc, cv=cv)
    return float(f1_score(y_enc, pred, average="macro"))


def _knn_purity(X: np.ndarray, y: np.ndarray, k: int) -> float:
    if len(y) <= k:
        return np.nan
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    neigh = idx[:, 1:]
    purity = [(y[neigh_i] == y[i]).mean() for i, neigh_i in enumerate(neigh)]
    return float(np.mean(purity))


def evaluate_projection(
    proj_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    label_col: str,
    min_class_size: int,
    knn_k: int,
    seed: int,
) -> dict[str, float | str | int]:
    merged = proj_df.merge(labels_df, on="identifier", how="inner")
    merged = merged.dropna(subset=["x", "y", "seq_length"])
    X_all = merged[["x", "y"]].to_numpy(dtype=float)
    X_all = StandardScaler().fit_transform(X_all)
    length = merged["seq_length"].to_numpy(dtype=float)

    reg = LinearRegression().fit(X_all, length)
    length_r2 = float(r2_score(length, reg.predict(X_all)))

    label_df = _filter_label(merged, label_col, min_class_size)
    X_label = label_df[["x", "y"]].to_numpy(dtype=float)
    X_label = StandardScaler().fit_transform(X_label)
    y = label_df[label_col].astype(str).to_numpy()

    return {
        "projection_name": str(proj_df["projection_name"].iloc[0]),
        "n_total": int(len(merged)),
        "length_r2": length_r2,
        "label": label_col,
        "n_labelled": int(len(label_df)),
        "n_label_classes": int(pd.Series(y).nunique()) if len(y) else 0,
        "label_macro_f1": _macro_f1(X_label, y, seed) if len(y) else np.nan,
        "label_knn_purity": _knn_purity(X_label, y, knn_k) if len(y) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projections-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("data/swissprot_rr/eval_labels.tsv"))
    parser.add_argument("--label-col", default="keyword_first")
    parser.add_argument("--min-class-size", type=int, default=30)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    projections = pd.read_parquet(args.projections_dir / "projections_data.parquet")
    labels = pd.read_csv(args.labels, sep="\t")

    rows = []
    for projection_name, proj_df in projections.groupby("projection_name", sort=False):
        rows.append(
            evaluate_projection(
                proj_df=proj_df,
                labels_df=labels,
                label_col=args.label_col,
                min_class_size=args.min_class_size,
                knn_k=args.knn_k,
                seed=args.seed,
            )
        )

    summary = pd.DataFrame(rows).sort_values(
        ["length_r2", "label_macro_f1"], ascending=[True, False]
    )
    out_path = args.projections_dir / f"eval_summary_{args.label_col}.tsv"
    summary.to_csv(out_path, sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
