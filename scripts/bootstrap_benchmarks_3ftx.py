"""Bootstrapped trustworthiness (k=1, k=15) and kNN accuracy for 3FTx.

Subsamples 80% of proteins 100 times, computes metrics per subsample,
and produces bar charts with 95% confidence interval error bars.
"""

import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

N_BOOTSTRAP = 100
SUBSAMPLE_FRAC = 0.8
RANDOM_STATE = 42


# --- Data Loading ---


def load_h5_embeddings_with_keys(path: Path) -> tuple[np.ndarray, list[str]]:
    arrays, keys = [], []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            keys.append(key)
            arrays.append(f[key][:])
    return np.array(arrays, dtype=np.float64), keys


def load_labels(csv_path: Path, identifiers: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    id_to_idx = {ident: i for i, ident in enumerate(identifiers)}
    df = df[df["identifier"].isin(id_to_idx)]
    df = df.copy()
    df["_order"] = df["identifier"].map(id_to_idx)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return df


# --- DR Methods ---


def run_pca(data: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(data)


def run_umap(data: np.ndarray) -> np.ndarray:
    from umap import UMAP
    return UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(data)


def run_ppca_external(target: np.ndarray, background: np.ndarray) -> np.ndarray:
    from protspace.utils.constants import DimensionReductionConfig
    from protspace.utils.reducers import PPCAReducer

    config = DimensionReductionConfig(
        n_components=2,
        random_state=42,
        background_strategy="external",
        regularization_mu=1e-3,
    )
    object.__setattr__(config, "background_data", background)
    reducer = PPCAReducer(config)
    return reducer.fit_transform(target)


# --- Metrics ---


def compute_trustworthiness(high_d: np.ndarray, low_d: np.ndarray, k: int) -> float:
    n = high_d.shape[0]
    actual_k = min(k, n - 2)
    return float(sklearn_trustworthiness(high_d, low_d, n_neighbors=actual_k, metric="euclidean"))


def compute_knn_accuracy(projection: np.ndarray, labels: np.ndarray, k: int = 15) -> float:
    actual_k = min(k, len(labels) - 2)
    clf = KNeighborsClassifier(n_neighbors=actual_k)
    loo = LeaveOneOut()
    correct = 0
    total = 0
    for train_idx, test_idx in loo.split(projection):
        clf.fit(projection[train_idx], labels[train_idx])
        pred = clf.predict(projection[test_idx])
        correct += (pred == labels[test_idx]).sum()
        total += len(test_idx)
    return correct / total


# --- Bootstrap ---


def bootstrap_metrics(
    target_data: np.ndarray,
    background_data: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    subsample_frac: float = SUBSAMPLE_FRAC,
):
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples = target_data.shape[0]
    n_sub = int(n_samples * subsample_frac)

    methods = ["PCA", "UMAP", "pPCA-ToxProt"]
    metrics = ["trust_k1", "trust_k15", "knn_acc"]

    results = {m: {metric: [] for metric in metrics} for m in methods}

    for i in range(n_bootstrap):
        if (i + 1) % 10 == 0:
            print(f"  Bootstrap {i+1}/{n_bootstrap}")

        idx = rng.choice(n_samples, size=n_sub, replace=False)
        sub_target = target_data[idx]
        sub_labels = labels[idx]

        projections = {
            "PCA": run_pca(sub_target),
            "UMAP": run_umap(sub_target),
            "pPCA-ToxProt": run_ppca_external(sub_target, background_data),
        }

        for method_name, proj in projections.items():
            t1 = compute_trustworthiness(sub_target, proj, k=1)
            t15 = compute_trustworthiness(sub_target, proj, k=15)
            knn = compute_knn_accuracy(proj, sub_labels, k=15)

            results[method_name]["trust_k1"].append(t1)
            results[method_name]["trust_k15"].append(t15)
            results[method_name]["knn_acc"].append(knn)

    return results


def compute_ci(values, ci=0.95):
    arr = np.array(values)
    mean = arr.mean()
    lower = np.percentile(arr, (1 - ci) / 2 * 100)
    upper = np.percentile(arr, (1 + ci) / 2 * 100)
    return mean, lower, upper


# --- Plotting ---


def plot_bootstrapped_bars(results: dict, output_dir: Path):
    methods = ["PCA", "UMAP", "pPCA-ToxProt"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metric_labels = {
        "trust_k1": "Trustworthiness (k=1)",
        "trust_k15": "Trustworthiness (k=15)",
        "knn_acc": "kNN Accuracy (LOO, k=15)",
    }

    for ax_idx, (metric_key, metric_label) in enumerate(metric_labels.items()):
        ax = axes[ax_idx]
        means, ci_lows, ci_highs = [], [], []

        for method in methods:
            mean, low, high = compute_ci(results[method][metric_key])
            means.append(mean)
            ci_lows.append(mean - low)
            ci_highs.append(high - mean)

        bars = ax.bar(methods, means, color=colors, edgecolor="black", linewidth=0.5)
        ax.errorbar(
            methods, means,
            yerr=[ci_lows, ci_highs],
            fmt="none", capsize=6, capthick=1.5, ecolor="black", linewidth=1.5,
        )

        ax.set_ylabel(metric_label)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric_label)

        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    fig.suptitle(
        "Bootstrapped Metrics (100 rounds, 80% subsample) — 3FTx · ProtT5",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_path = output_dir / "bootstrapped_metrics_3ftx.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_trustworthiness_k1(results: dict, output_dir: Path):
    """Separate plot for just k=1 trustworthiness with error bars."""
    methods = ["PCA", "UMAP", "pPCA-ToxProt"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    means, ci_lows, ci_highs = [], [], []

    for method in methods:
        mean, low, high = compute_ci(results[method]["trust_k1"])
        means.append(mean)
        ci_lows.append(mean - low)
        ci_highs.append(high - mean)

    bars = ax.bar(methods, means, color=colors, edgecolor="black", linewidth=0.5)
    ax.errorbar(
        methods, means,
        yerr=[ci_lows, ci_highs],
        fmt="none", capsize=8, capthick=1.5, ecolor="black", linewidth=1.5,
    )

    ax.set_ylabel("Trustworthiness (k=1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Trustworthiness k=1 with 95% CI\n(100 bootstrap rounds, 80% subsample)")

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    plt.tight_layout()
    out_path = output_dir / "trustworthiness_k1_bootstrap_3ftx.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


# --- Main ---


def main():
    target_path = Path("data/3ftx/tmp/prot_t5.h5")
    background_path = Path("data/toxprot/prot_t5.h5")
    labels_path = Path("data/3ftx/3FTx_accession.csv")
    output_dir = Path("src/protspace/benchmark/results")

    print("Loading data...")
    target_data, target_ids = load_h5_embeddings_with_keys(target_path)
    background_data, _ = load_h5_embeddings_with_keys(background_path)

    labels_df = load_labels(labels_path, target_ids)
    matched_ids = labels_df["identifier"].tolist()
    mask = [i for i, ident in enumerate(target_ids) if ident in set(matched_ids)]
    target_data = target_data[mask]
    target_ids = [target_ids[i] for i in mask]
    labels = labels_df["group"].values

    print(f"  Target: {target_data.shape[0]} proteins, Background: {background_data.shape[0]}")
    print(f"  Groups: {np.unique(labels)}")

    print(f"\nRunning {N_BOOTSTRAP} bootstrap rounds ({int(SUBSAMPLE_FRAC*100)}% subsample)...")
    results = bootstrap_metrics(target_data, background_data, labels)

    # Save raw results as CSV
    rows = []
    for method in results:
        for i in range(N_BOOTSTRAP):
            rows.append({
                "method": method,
                "round": i,
                "trust_k1": results[method]["trust_k1"][i],
                "trust_k15": results[method]["trust_k15"][i],
                "knn_acc": results[method]["knn_acc"][i],
            })
    df = pd.DataFrame(rows)
    csv_path = output_dir / "bootstrapped_metrics_3ftx.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS (mean [95% CI])")
    print("=" * 60)
    for method in ["PCA", "UMAP", "pPCA-ToxProt"]:
        print(f"\n  {method}:")
        for metric in ["trust_k1", "trust_k15", "knn_acc"]:
            mean, low, high = compute_ci(results[method][metric])
            print(f"    {metric}: {mean:.4f} [{low:.4f}, {high:.4f}]")

    # Generate plots
    print("\nGenerating plots...")
    plot_bootstrapped_bars(results, output_dir)
    plot_trustworthiness_k1(results, output_dir)


if __name__ == "__main__":
    main()
