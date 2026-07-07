"""Bootstrapped trustworthiness across multiple datasets and k values.

Compares PCA, UMAP, and pPCA (where applicable) on:
  - 3FTx (453 proteins, background: ToxProt)
  - ToxProt (1500 proteins, background: SwissProt RR)
  - PLA2G2 (448 proteins, background: ToxProt)
  - CATH S40 (3000 proteins, PCA vs UMAP only — no biological background)

Produces:
  1. A multi-k line plot per dataset (trustworthiness vs k)
  2. A summary comparison across all datasets at k=1 and k=15
  3. CSV with all raw results
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
OUTPUT_DIR = Path("src/protspace/benchmark/results")


# --- Data Loading ---


def load_h5_flat(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load flat HDF5 (each key is a 1D embedding)."""
    arrays, keys = [], []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            data = f[key]
            if isinstance(data, h5py.Group):
                sub_keys = list(data.keys())
                arr = np.array(data[sub_keys[0]])
            else:
                arr = np.array(data)
            arrays.append(arr)
            keys.append(key)
    return np.array(arrays, dtype=np.float64), keys


def load_cath_s40(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load CATH S40 (grouped HDF5: each protein has one sub-group with residue range)."""
    arrays, keys = [], []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            grp = f[key]
            sub_keys = list(grp.keys())
            arr = np.array(grp[sub_keys[0]])
            arrays.append(arr)
            keys.append(key)
    return np.array(arrays, dtype=np.float64), keys


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


# --- Bootstrap ---


def bootstrap_trustworthiness(
    target_data: np.ndarray,
    background_data: np.ndarray | None,
    include_ppca: bool = True,
):
    """Run bootstrapped trustworthiness for all k values."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples = target_data.shape[0]
    n_sub = int(n_samples * SUBSAMPLE_FRAC)

    methods = ["PCA", "UMAP"]
    if include_ppca and background_data is not None:
        methods.append("pPCA")

    # results[method][k] = list of values
    results = {m: {k: [] for k in K_VALUES} for m in methods}

    for i in range(N_BOOTSTRAP):
        if (i + 1) % 10 == 0:
            print(f"    Bootstrap {i+1}/{N_BOOTSTRAP}")

        idx = rng.choice(n_samples, size=n_sub, replace=False)
        sub_target = target_data[idx]

        projections = {
            "PCA": run_pca(sub_target),
            "UMAP": run_umap(sub_target),
        }
        if include_ppca and background_data is not None:
            projections["pPCA"] = run_ppca_external(sub_target, background_data)

        for method_name, proj in projections.items():
            for k in K_VALUES:
                actual_k = min(k, n_sub - 2)
                t = float(sklearn_trustworthiness(
                    sub_target, proj, n_neighbors=actual_k, metric="euclidean"
                ))
                results[method_name][k].append(t)

    return results


def compute_ci(values):
    arr = np.array(values)
    return arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5)


# --- Plotting ---


def plot_multi_k(results: dict, dataset_name: str, output_dir: Path):
    """Line plot: trustworthiness vs k for one dataset."""
    colors = {"PCA": "#4C72B0", "UMAP": "#55A868", "pPCA": "#C44E52"}
    markers = {"PCA": "o", "UMAP": "s", "pPCA": "^"}

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in results:
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
        f"Trustworthiness vs. k — {dataset_name}\n"
        f"{N_BOOTSTRAP} bootstrap rounds, {int(SUBSAMPLE_FRAC*100)}% subsample",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Dynamic y-axis: find the global min across all methods/k values
    all_lows = []
    for method in results:
        for k in K_VALUES:
            _, low, _ = compute_ci(results[method][k])
            all_lows.append(low)
    y_min = max(0.85, min(all_lows) - 0.02)
    ax.set_ylim(y_min, 1.005)

    plt.tight_layout()
    slug = dataset_name.lower().replace(" ", "_").replace("-", "_")
    out_path = output_dir / f"trustworthiness_multi_k_{slug}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")
    return out_path


def plot_cross_dataset_summary(all_results: dict, output_dir: Path):
    """Bar chart comparing all datasets at k=1 and k=15 side by side."""
    datasets = list(all_results.keys())
    k_show = [1, 15]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"PCA": "#4C72B0", "UMAP": "#55A868", "pPCA": "#C44E52"}

    global_min = 1.0
    for ax_idx, k in enumerate(k_show):
        ax = axes[ax_idx]
        x = np.arange(len(datasets))
        width = 0.25

        all_methods = set()
        for ds in datasets:
            all_methods.update(all_results[ds].keys())
        methods_ordered = [m for m in ["PCA", "UMAP", "pPCA"] if m in all_methods]

        for i, method in enumerate(methods_ordered):
            means, errors_low, errors_high = [], [], []
            positions = []
            for j, ds in enumerate(datasets):
                if method in all_results[ds]:
                    mean, low, high = compute_ci(all_results[ds][method][k])
                    means.append(mean)
                    errors_low.append(mean - low)
                    errors_high.append(high - mean)
                    global_min = min(global_min, low)
                else:
                    means.append(np.nan)
                    errors_low.append(0)
                    errors_high.append(0)
                positions.append(j)

            offset = (i - len(methods_ordered) / 2 + 0.5) * width
            pos_arr = np.array(positions) + offset
            means_arr = np.array(means)
            valid = ~np.isnan(means_arr)
            ax.bar(
                pos_arr[valid], means_arr[valid], width,
                label=method, color=colors[method],
                edgecolor="black", linewidth=0.5,
            )
            ax.errorbar(
                pos_arr[valid], means_arr[valid],
                yerr=[np.array(errors_low)[valid], np.array(errors_high)[valid]],
                fmt="none", capsize=4, capthick=1, ecolor="black",
            )

        ax.set_xlabel("Dataset", fontsize=11)
        ax.set_ylabel("Trustworthiness", fontsize=11)
        ax.set_title(f"Trustworthiness (k={k})", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")

    for ax in axes:
        ax.set_ylim(0, 1.05)

    fig.suptitle(
        f"Cross-Dataset Trustworthiness Comparison\n"
        f"{N_BOOTSTRAP} bootstrap rounds, {int(SUBSAMPLE_FRAC*100)}% subsample",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_path = output_dir / "trustworthiness_cross_dataset.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# --- Main ---


DATASETS = {
    "3FTx": {
        "target": Path("data/3ftx/tmp/prot_t5.h5"),
        "background": Path("data/toxprot/prot_t5.h5"),
        "loader": "flat",
        "ppca": True,
    },
    "ToxProt": {
        "target": Path("data/toxprot/prot_t5.h5"),
        "background": Path("data/swissprot_rr/prot_t5.h5"),
        "loader": "flat",
        "ppca": True,
    },
    "PLA2G2": {
        "target": Path("data/pla2g2/prot_t5.h5"),
        "background": Path("data/toxprot/prot_t5.h5"),
        "loader": "flat",
        "ppca": True,
    },
    "CATH S40": {
        "target": Path("data/cath_s40/prot_t5.h5"),
        "background": None,
        "loader": "cath",
        "ppca": False,
    },
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    all_rows = []

    for ds_name, cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        # Load target
        if cfg["loader"] == "cath":
            target_data, target_ids = load_cath_s40(cfg["target"])
        else:
            target_data, target_ids = load_h5_flat(cfg["target"])

        print(f"  Target: {target_data.shape[0]} proteins, dim={target_data.shape[1]}")

        # Load background if applicable
        background_data = None
        if cfg["ppca"] and cfg["background"] is not None:
            background_data, _ = load_h5_flat(cfg["background"])
            print(f"  Background: {background_data.shape[0]} proteins")

        # Run bootstrap
        print(f"  Running {N_BOOTSTRAP} bootstrap rounds...")
        results = bootstrap_trustworthiness(
            target_data, background_data, include_ppca=cfg["ppca"]
        )
        all_results[ds_name] = results

        # Print summary
        for method in results:
            print(f"    {method}:")
            for k in K_VALUES:
                mean, low, high = compute_ci(results[method][k])
                print(f"      k={k:2d}: {mean:.4f} [{low:.4f}, {high:.4f}]")

        # Save per-dataset plot
        plot_multi_k(results, ds_name, OUTPUT_DIR)

        # Collect rows for CSV
        for method in results:
            for k in K_VALUES:
                for i, val in enumerate(results[method][k]):
                    all_rows.append({
                        "dataset": ds_name,
                        "method": method,
                        "k": k,
                        "round": i,
                        "trustworthiness": val,
                    })

    # Save combined CSV
    df = pd.DataFrame(all_rows)
    csv_path = OUTPUT_DIR / "trustworthiness_multi_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Cross-dataset comparison plot
    print("\nGenerating cross-dataset comparison...")
    plot_cross_dataset_summary(all_results, OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
