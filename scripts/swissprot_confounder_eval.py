"""End-to-end SwissProt confounder evaluation for RQ3 sub-question 4.

Research question: Can contrastive DR (ρPCA) remove sequence length as a dominant
axis on the redundancy-reduced SwissProt set, revealing functional groupings instead?

Setup:
  - Target: full prot_t5.h5 (9,757 human proteins, length range 16–2104)
  - Background: length_extremes_20pct.h5 (1,955 proteins from top/bottom 10% by length)
  - ρPCA suppresses directions where length-extreme proteins vary most
  - Comparison: PCA (linear baseline), UMAP (nonlinear upper bound)

Metrics (bootstrapped, 100 rounds, 80% subsample):
  - length_r2: R² of linear regression predicting seq_length from 2D coords (lower = better)
  - function_macro_f1: 5-fold CV logistic regression predicting EC class (higher = better)
  - function_knn_purity: fraction of k=15 nearest neighbors sharing same EC label (higher = better)

Output:
  - results/swissprot_confounder_eval.csv (metrics table with CIs)
  - results/swissprot_confounder_eval.png (bar chart with error bars)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "swissprot_rr"
RESULTS = ROOT / "src" / "protspace" / "benchmark" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
KNN_K = 15
MIN_CLASS_SIZE = 50
LABEL_COL = "ec_top"
N_BOOTSTRAP = 100
SUBSAMPLE_FRAC = 0.8


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    ids, arrays = [], []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            ids.append(key)
            arrays.append(np.array(f[key], dtype=np.float64))
    return ids, np.array(arrays)


def run_ppca(X_target: np.ndarray, X_bg: np.ndarray) -> np.ndarray:
    from protspace.utils.constants import DimensionReductionConfig
    from protspace.utils.reducers import PPCAReducer

    config = DimensionReductionConfig(
        n_components=2,
        random_state=SEED,
        regularization_mu=1e-3,
    )
    object.__setattr__(config, "background_data", X_bg)
    reducer = PPCAReducer(config)
    return reducer.fit_transform(X_target)


def compute_length_r2(X_2d: np.ndarray, lengths: np.ndarray) -> float:
    X_scaled = StandardScaler().fit_transform(X_2d)
    reg = LinearRegression().fit(X_scaled, lengths)
    return float(r2_score(lengths, reg.predict(X_scaled)))


def compute_macro_f1(X_2d: np.ndarray, y: np.ndarray, seed: int) -> float:
    labels, counts = np.unique(y, return_counts=True)
    if labels.size < 2 or counts.min() < 2:
        return np.nan
    X_scaled = StandardScaler().fit_transform(X_2d)
    n_splits = min(5, int(counts.min()))
    y_enc = LabelEncoder().fit_transform(y)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, X_scaled, y_enc, cv=cv)
    return float(f1_score(y_enc, pred, average="macro"))


def compute_knn_purity(X_2d: np.ndarray, y: np.ndarray, k: int) -> float:
    if len(y) <= k:
        return np.nan
    X_scaled = StandardScaler().fit_transform(X_2d)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_scaled)
    _, idx = nn.kneighbors(X_scaled)
    neigh = idx[:, 1:]
    purity = [(y[neigh_i] == y[i]).mean() for i, neigh_i in enumerate(neigh)]
    return float(np.mean(purity))


def bootstrap_metrics(
    projections: dict[str, np.ndarray],
    labels_df: pd.DataFrame,
    func_mask: np.ndarray,
    func_labels: np.ndarray,
    n_bootstrap: int,
    subsample_frac: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n_total = len(labels_df)
    n_func = int(func_mask.sum())
    func_indices = np.where(func_mask)[0]

    all_results = []

    for b in range(n_bootstrap):
        total_idx = rng.choice(n_total, size=int(n_total * subsample_frac), replace=False)
        total_idx.sort()
        func_sub_mask = np.isin(np.arange(n_total), total_idx) & func_mask
        func_sub_idx = np.where(func_sub_mask)[0]

        lengths_sub = labels_df["seq_length"].to_numpy(dtype=float)[total_idx]
        func_labels_sub = labels_df["_func_label"].to_numpy()[func_sub_idx]

        for name, proj in projections.items():
            proj_total = proj[total_idx]
            proj_func = proj[func_sub_idx]

            r2 = compute_length_r2(proj_total, lengths_sub)
            f1 = compute_macro_f1(proj_func, func_labels_sub, seed=SEED + b)
            purity = compute_knn_purity(proj_func, func_labels_sub, KNN_K)

            all_results.append({
                "bootstrap": b,
                "method": name,
                "length_r2": r2,
                "function_macro_f1": f1,
                "function_knn_purity": purity,
            })

        if (b + 1) % 20 == 0:
            print(f"    Bootstrap {b + 1}/{n_bootstrap} done")

    return pd.DataFrame(all_results)


def plot_results(boot_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    methods = summary_df["method"].tolist()
    colors = {"PCA": "#4878CF", "UMAP": "#6ACC65", "ρPCA": "#D65F5F"}
    bar_colors = [colors.get(m, "#999999") for m in methods]

    metrics = [
        ("length_r2", "Length R²", "Length as Confounder\n(lower = better)"),
        ("function_macro_f1", "Macro F1 (EC class)", "EC Class Prediction (5-fold CV)\n(higher = better)"),
        ("function_knn_purity", "kNN Purity (k=15)", "Local EC Coherence\n(higher = better)"),
    ]

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        means = summary_df[f"{metric}_mean"].tolist()
        ci_lo = summary_df[f"{metric}_ci_lo"].tolist()
        ci_hi = summary_df[f"{metric}_ci_hi"].tolist()
        errors_lo = [m - lo for m, lo in zip(means, ci_lo)]
        errors_hi = [hi - m for m, hi in zip(means, ci_hi)]

        bars = ax.bar(
            methods, means, color=bar_colors,
            edgecolor="black", linewidth=0.5,
            yerr=[errors_lo, errors_hi], capsize=5, error_kw={"linewidth": 1.5},
        )
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errors_hi) * 0.3,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
            )
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(ci_hi) * 1.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    n_proteins = summary_df["n_proteins"].iloc[0]
    n_func = summary_df["n_func_labelled"].iloc[0]
    fig.suptitle(
        f"SwissProt RR · Confounder Evaluation ({N_BOOTSTRAP} bootstrap rounds, "
        f"{int(SUBSAMPLE_FRAC*100)}% subsample)\n"
        f"n={n_proteins} proteins · EC label ({n_func} labelled, "
        f"{summary_df['n_func_classes'].iloc[0]} classes ≥{MIN_CLASS_SIZE})",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot: {out_path}")


def main() -> None:
    print("=" * 60)
    print("SwissProt Confounder Evaluation (RQ3.4) — Bootstrapped")
    print("=" * 60)

    # Load embeddings
    print("\n[1/5] Loading embeddings...")
    target_ids, X_target = load_embeddings(DATA / "prot_t5.h5")
    _, X_bg = load_embeddings(DATA / "background_length_extremes_20pct.h5")
    print(f"  Target: {X_target.shape[0]} proteins × {X_target.shape[1]} dims")
    print(f"  Background: {X_bg.shape[0]} proteins × {X_bg.shape[1]} dims")

    # Load labels
    print("\n[2/5] Loading and filtering labels...")
    labels = pd.read_csv(DATA / "eval_labels.tsv", sep="\t")
    id_to_idx = {pid: i for i, pid in enumerate(target_ids)}
    labels = labels[labels["identifier"].isin(id_to_idx)].copy()
    labels["_idx"] = labels["identifier"].map(id_to_idx)
    labels = labels.sort_values("_idx").reset_index(drop=True)

    # Filter EC labels
    labels["_func_label"] = labels[LABEL_COL].fillna("").astype(str)
    labels.loc[labels["_func_label"] == "no_ec", "_func_label"] = ""
    func_mask = labels["_func_label"].str.len() > 0
    func_counts = labels.loc[func_mask, "_func_label"].value_counts()
    valid_classes = func_counts[func_counts >= MIN_CLASS_SIZE].index
    func_mask = func_mask & labels["_func_label"].isin(valid_classes)
    labels.loc[~func_mask, "_func_label"] = ""

    print(f"  Matched: {len(labels)} proteins")
    print(f"  Functional label: {LABEL_COL}")
    print(f"  Valid classes (≥{MIN_CLASS_SIZE}): {len(valid_classes)} → {list(valid_classes)}")
    print(f"  Proteins with valid EC: {func_mask.sum()}")

    # Run projections
    print("\n[3/5] Running projections...")
    from umap import UMAP

    print("  PCA...")
    pca_proj = PCA(n_components=2, random_state=SEED).fit_transform(X_target)

    print("  UMAP...")
    umap_proj = UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED
    ).fit_transform(X_target)

    print("  ρPCA (background=length_extremes_20pct)...")
    ppca_proj = run_ppca(X_target, X_bg)

    # Align projections to protein order in labels
    idx_arr = labels["_idx"].to_numpy()
    projections = {
        "PCA": pca_proj[idx_arr],
        "UMAP": umap_proj[idx_arr],
        "ρPCA": ppca_proj[idx_arr],
    }

    # Bootstrap
    print(f"\n[4/5] Bootstrapping ({N_BOOTSTRAP} rounds, {int(SUBSAMPLE_FRAC*100)}% subsample)...")
    boot_df = bootstrap_metrics(
        projections, labels, func_mask.to_numpy(),
        labels["_func_label"].to_numpy(), N_BOOTSTRAP, SUBSAMPLE_FRAC,
    )

    # Summarize
    summary_rows = []
    for method in ["PCA", "UMAP", "ρPCA"]:
        mdf = boot_df[boot_df["method"] == method]
        row = {"method": method, "n_proteins": len(labels),
               "n_func_labelled": int(func_mask.sum()), "n_func_classes": int(len(valid_classes))}
        for metric in ["length_r2", "function_macro_f1", "function_knn_purity"]:
            vals = mdf[metric].dropna()
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_ci_lo"] = np.percentile(vals, 2.5)
            row[f"{metric}_ci_hi"] = np.percentile(vals, 97.5)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    # Save & plot
    print("\n[5/5] Saving results...")
    csv_path = RESULTS / "swissprot_confounder_eval.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    boot_csv = RESULTS / "swissprot_confounder_eval_bootstrap.csv"
    boot_df.to_csv(boot_csv, index=False)
    print(f"  Bootstrap raw: {boot_csv}")

    plot_results(boot_df, summary, RESULTS / "swissprot_confounder_eval.png")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS (mean ± 95% CI)")
    print("=" * 60)
    for _, row in summary.iterrows():
        print(f"\n  {row['method']}:")
        print(f"    Length R²:       {row['length_r2_mean']:.4f} [{row['length_r2_ci_lo']:.4f}, {row['length_r2_ci_hi']:.4f}]")
        print(f"    Function F1:     {row['function_macro_f1_mean']:.4f} [{row['function_macro_f1_ci_lo']:.4f}, {row['function_macro_f1_ci_hi']:.4f}]")
        print(f"    kNN Purity:      {row['function_knn_purity_mean']:.4f} [{row['function_knn_purity_ci_lo']:.4f}, {row['function_knn_purity_ci_hi']:.4f}]")

    pca_r = summary[summary["method"] == "PCA"].iloc[0]
    ppca_r = summary[summary["method"] == "ρPCA"].iloc[0]
    print(f"\n  ρPCA vs PCA (linear-to-linear comparison):")
    r2_drop = (1 - ppca_r["length_r2_mean"] / pca_r["length_r2_mean"]) * 100
    print(f"    Length R² reduction: {r2_drop:.1f}%")
    f1_delta = ppca_r["function_macro_f1_mean"] - pca_r["function_macro_f1_mean"]
    print(f"    Function F1 delta: {f1_delta:+.4f}")


if __name__ == "__main__":
    main()
