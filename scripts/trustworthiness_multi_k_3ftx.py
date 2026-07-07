"""Trustworthiness across multiple k values for 3FTx.

Shows how PCA, UMAP, and pPCA-ToxProt compare at different neighborhood sizes.
UMAP excels at low k (local), PCA catches up at higher k (global).
"""

import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

K_VALUES = [1, 5, 15, 30, 50]
N_BOOTSTRAP = 50
SUBSAMPLE_FRAC = 0.8
RANDOM_STATE = 42


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


def compute_ci(values):
    arr = np.array(values)
    mean = arr.mean()
    lower = np.percentile(arr, 2.5)
    upper = np.percentile(arr, 97.5)
    return mean, lower, upper


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

    print(f"  Target: {target_data.shape[0]} proteins")
    print(f"  K values: {K_VALUES}")
    print(f"  Bootstrap rounds: {N_BOOTSTRAP}")

    methods = ["PCA", "UMAP", "pPCA-ToxProt"]
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples = target_data.shape[0]
    n_sub = int(n_samples * SUBSAMPLE_FRAC)

    # results[method][k] = list of trustworthiness values
    results = {m: {k: [] for k in K_VALUES} for m in methods}

    for i in range(N_BOOTSTRAP):
        if (i + 1) % 10 == 0:
            print(f"  Bootstrap {i+1}/{N_BOOTSTRAP}")

        idx = rng.choice(n_samples, size=n_sub, replace=False)
        sub_target = target_data[idx]

        projections = {
            "PCA": run_pca(sub_target),
            "UMAP": run_umap(sub_target),
            "pPCA-ToxProt": run_ppca_external(sub_target, background_data),
        }

        for method_name, proj in projections.items():
            for k in K_VALUES:
                actual_k = min(k, n_sub - 2)
                t = float(sklearn_trustworthiness(
                    sub_target, proj, n_neighbors=actual_k, metric="euclidean"
                ))
                results[method_name][k].append(t)

    # Save raw results
    rows = []
    for method in methods:
        for k in K_VALUES:
            for i, val in enumerate(results[method][k]):
                rows.append({"method": method, "k": k, "round": i, "trustworthiness": val})
    df = pd.DataFrame(rows)
    csv_path = output_dir / "trustworthiness_multi_k_3ftx.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS (mean [95% CI])")
    print("=" * 60)
    for method in methods:
        print(f"\n  {method}:")
        for k in K_VALUES:
            mean, low, high = compute_ci(results[method][k])
            print(f"    k={k:2d}: {mean:.4f} [{low:.4f}, {high:.4f}]")

    # Plot: line chart
    colors = {"PCA": "#4C72B0", "UMAP": "#55A868", "pPCA-ToxProt": "#C44E52"}
    markers = {"PCA": "o", "UMAP": "s", "pPCA-ToxProt": "^"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:
        means, ci_lows, ci_highs = [], [], []
        for k in K_VALUES:
            mean, low, high = compute_ci(results[method][k])
            means.append(mean)
            ci_lows.append(mean - low)
            ci_highs.append(high - mean)

        ax.errorbar(
            K_VALUES, means,
            yerr=[ci_lows, ci_highs],
            label=method,
            color=colors[method],
            marker=markers[method],
            markersize=8,
            capsize=4,
            capthick=1.5,
            linewidth=2,
        )

    ax.set_xlabel("k (neighborhood size)", fontsize=12)
    ax.set_ylabel("Trustworthiness", fontsize=12)
    ax.set_title(
        "Trustworthiness vs. Neighborhood Size (k)\n"
        f"3FTx · ProtT5 · {N_BOOTSTRAP} bootstrap rounds, 80% subsample",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0.9, 1.0)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "trustworthiness_multi_k_3ftx.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
