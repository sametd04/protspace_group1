#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRIC_COLUMNS = [
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "adjusted_rand",
    "normalized_mutual_info",
]

METRIC_LABELS = {
    "silhouette": "Silhouette",
    "calinski_harabasz": "Calinski-Harabasz",
    "davies_bouldin": "Davies-Bouldin",
    "adjusted_rand": "ARI",
    "normalized_mutual_info": "NMI",
}


def find_one(pattern: str, root: Path) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {root}")
    if len(matches) > 1:
        print(f"[warn] multiple matches for {pattern!r}; using {matches[0]}")
    return matches[0]


def load_npz_coords(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)

    preferred = [
        "data",
        "projection",
        "coordinates",
        "coords",
        "embedding",
        "X",
        "arr_0",
    ]

    for key in preferred:
        if key in obj:
            arr = np.asarray(obj[key])
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr[:, :2].astype(float)

    for key in obj.files:
        arr = np.asarray(obj[key])
        if arr.ndim == 2 and arr.shape[1] >= 2 and np.issubdtype(arr.dtype, np.number):
            return arr[:, :2].astype(float)

    raise ValueError(f"Could not find a 2D coordinate array in {path}. Keys: {obj.files}")


def clean_label_series(s: pd.Series) -> pd.Series:
    out = s.copy()
    out = out.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan})
    return out


def read_scores(scores_dir: Path) -> pd.DataFrame:
    combined = scores_dir / "combined_3ftx_scores.csv"
    if combined.exists():
        return pd.read_csv(combined)

    parts = []
    files = [
        ("full_or_mature_biology", scores_dir / "full_or_mature_pca_rhopca_scores.csv"),
        ("mature_only_biology", scores_dir / "mature_only_pca_scores.csv"),
        ("full_or_mature_sp_nuisance", scores_dir / "full_or_mature_sp_in_embedding_scores.csv"),
    ]
    for experiment, path in files:
        if path.exists():
            df = pd.read_csv(path)
            df["experiment"] = experiment
            parts.append(df)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


def lookup_metrics(
    scores: pd.DataFrame,
    experiment: str | None,
    method: str,
    label_column: str,
) -> dict[str, str]:
    if experiment is None or scores.empty:
        return {c: "N/A" for c in METRIC_COLUMNS}

    df = scores.copy()
    if "experiment" in df.columns:
        df = df[df["experiment"] == experiment]
    df = df[(df["method"] == method) & (df["label_column"] == label_column)]

    if df.empty:
        return {c: "N/A" for c in METRIC_COLUMNS}

    row = df.iloc[0]
    out = {}
    for c in METRIC_COLUMNS:
        val = row.get(c, np.nan)
        if pd.isna(val):
            out[c] = "N/A"
        else:
            out[c] = f"{float(val):.3f}"
    return out


def sort_categories(values: pd.Series) -> list:
    cats = [x for x in pd.unique(values.dropna())]

    def key(x):
        text = str(x)
        return (text.lower(), text)

    return sorted(cats, key=key)


def draw_plot(
    coords: np.ndarray,
    annotations: pd.DataFrame,
    label_column: str,
    title: str,
    metrics: dict[str, str],
    out_path: Path,
):
    if len(coords) != len(annotations):
        raise ValueError(
            f"Coordinate rows ({len(coords)}) do not match annotation rows ({len(annotations)}) "
            f"for {out_path.name}"
        )

    if label_column not in annotations.columns:
        raise KeyError(f"Missing annotation column {label_column!r}")

    labels = clean_label_series(annotations[label_column])
    valid = labels.notna().to_numpy()
    xy = coords[valid]
    lab = labels[valid]

    fig = plt.figure(figsize=(10.8, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.15], wspace=0.12)

    ax = fig.add_subplot(gs[0, 0])
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_metrics.axis("off")

    cats = sort_categories(lab)
    cmap = plt.get_cmap("tab20", max(len(cats), 1))

    for i, cat in enumerate(cats):
        mask = (lab == cat).to_numpy()
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=12,
            alpha=0.82,
            linewidths=0,
            label=str(cat),
            color=cmap(i),
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.grid(True, alpha=0.18)

    if len(cats) <= 12:
        ax.legend(
            title=label_column,
            loc="best",
            fontsize=7.5,
            title_fontsize=8,
            frameon=True,
            markerscale=1.4,
        )
    else:
        ax.text(
            0.02,
            0.98,
            f"{len(cats)} categories",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75),
        )

    ax_metrics.text(
        0.0,
        0.98,
        "Cluster metrics",
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    ax_metrics.text(
        0.0,
        0.90,
        f"label: {label_column}",
        fontsize=9.2,
        va="top",
    )

    y = 0.78
    for c in METRIC_COLUMNS:
        ax_metrics.text(0.0, y, METRIC_LABELS[c], fontsize=9.8, va="center")
        ax_metrics.text(1.0, y, metrics[c], fontsize=9.8, va="center", ha="right")
        y -= 0.105

    if len(cats) < 2:
        ax_metrics.text(
            0.0,
            0.15,
            "N/A: only one label class",
            fontsize=9,
            color="darkred",
            va="top",
            wrap=True,
        )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def method_from_npz(path: Path) -> str:
    m = re.search(r"_((?:pca|ppca)\d)_", path.name)
    if m:
        return m.group(1)
    if "ppca" in path.name:
        return "ppca2"
    return "pca2"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run", required=True, type=Path)
    parser.add_argument("--mature-run", required=True, type=Path)
    parser.add_argument("--ann-full", required=True, type=Path)
    parser.add_argument("--ann-mature", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    ann_full = pd.read_csv(args.ann_full)
    ann_mature = pd.read_csv(args.ann_mature)
    scores = read_scores(args.scores)

    full_tmp = args.full_run / "tmp"
    mature_tmp = args.mature_run / "tmp"

    full_pca_npz = find_one("proj_*_pca2_*.npz", full_tmp)
    full_ppca_npz = find_one("proj_*_ppca2_*.npz", full_tmp)
    mature_pca_npz = find_one("proj_*_pca2_*.npz", mature_tmp)

    coords_full_pca = load_npz_coords(full_pca_npz)
    coords_full_ppca = load_npz_coords(full_ppca_npz)
    coords_mature_pca = load_npz_coords(mature_pca_npz)

    plot_specs = [
        {
            "filename": "01_full_or_mature_PCA__sp_in_embedding.png",
            "coords": coords_full_pca,
            "ann": ann_full,
            "label": "sp_in_embedding",
            "title": "PCA · full-or-mature sequences · colored by SP in embedding",
            "experiment": "full_or_mature_sp_nuisance",
            "method": "pca2",
        },
        {
            "filename": "02_full_or_mature_PCA__major_group.png",
            "coords": coords_full_pca,
            "ann": ann_full,
            "label": "major_group",
            "title": "PCA · full-or-mature sequences · colored by major group",
            "experiment": "full_or_mature_biology",
            "method": "pca2",
        },
        {
            "filename": "03_mature_only_PCA__sp_in_embedding.png",
            "coords": coords_mature_pca,
            "ann": ann_mature,
            "label": "sp_in_embedding",
            "title": "PCA · mature-only sequences · colored by SP in embedding",
            "experiment": None,
            "method": "pca2",
        },
        {
            "filename": "04_mature_only_PCA__major_group.png",
            "coords": coords_mature_pca,
            "ann": ann_mature,
            "label": "major_group",
            "title": "PCA · mature-only sequences · colored by major group",
            "experiment": "mature_only_biology",
            "method": "pca2",
        },
        {
            "filename": "05_full_or_mature_rhoPCA__sp_in_embedding.png",
            "coords": coords_full_ppca,
            "ann": ann_full,
            "label": "sp_in_embedding",
            "title": "ρPCA · full-or-mature sequences · colored by SP in embedding",
            "experiment": "full_or_mature_sp_nuisance",
            "method": "ppca2",
        },
        {
            "filename": "06_full_or_mature_rhoPCA__major_group.png",
            "coords": coords_full_ppca,
            "ann": ann_full,
            "label": "major_group",
            "title": "ρPCA · full-or-mature sequences · colored by major group",
            "experiment": "full_or_mature_biology",
            "method": "ppca2",
        },
    ]

    summary_rows = []
    for spec in plot_specs:
        label_values = clean_label_series(spec["ann"][spec["label"]])
        n_classes = label_values.dropna().nunique()

        if n_classes < 2:
            metrics = {c: "N/A" for c in METRIC_COLUMNS}
        else:
            metrics = lookup_metrics(
                scores=scores,
                experiment=spec["experiment"],
                method=spec["method"],
                label_column=spec["label"],
            )

        out_path = args.out / spec["filename"]
        draw_plot(
            coords=spec["coords"],
            annotations=spec["ann"],
            label_column=spec["label"],
            title=spec["title"],
            metrics=metrics,
            out_path=out_path,
        )

        row = {
            "plot": spec["filename"],
            "method": spec["method"],
            "label_column": spec["label"],
            "n_label_classes": n_classes,
        }
        row.update(metrics)
        summary_rows.append(row)
        print(f"Wrote {out_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = args.out / "week_update_plot_metric_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
