"""Cross-dataset benchmarks: trustworthiness, kNN accuracy, silhouette.

Datasets:
  - 3FTx (453 proteins, 4 functional groups, bg: ToxProt)
  - SwissProt RR EC subset (500 proteins, 7 EC classes, bg: length extremes)
  - PLA2G2 (448 proteins, no labels, bg: ToxProt) — trustworthiness only
  - ToxProt (1500 proteins, no labels, bg: SwissProt RR) — trustworthiness only

Produces grouped bar charts comparing methods across datasets.
"""

import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

N_BOOTSTRAP = 50
SUBSAMPLE_FRAC = 0.8
RANDOM_STATE = 42
OUTPUT_DIR = Path("src/protspace/benchmark/results")

METHODS = ["PCA", "UMAP", "ρPCA"]
COLORS = {"PCA": "#4C72B0", "UMAP": "#55A868", "ρPCA": "#C44E52"}


def load_h5(path: Path) -> tuple[np.ndarray, list[str]]:
    arrays, keys = [], []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            data = f[key]
            if isinstance(data, h5py.Group):
                arr = np.array(data[list(data.keys())[0]])
            else:
                arr = np.array(data)
            arrays.append(arr)
            keys.append(key)
    return np.array(arrays, dtype=np.float64), keys


def run_pca(data):
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(data)


def run_umap(data):
    from umap import UMAP
    return UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(data)


def run_ppca(target, background):
    from protspace.utils.constants import DimensionReductionConfig
    from protspace.utils.reducers import PPCAReducer
    config = DimensionReductionConfig(n_components=2, random_state=42, regularization_mu=1e-3)
    object.__setattr__(config, "background_data", background)
    return PPCAReducer(config).fit_transform(target)


def compute_ci(values):
    arr = np.array(values)
    return arr.mean(), np.percentile(arr, 2.5), np.percentile(arr, 97.5)


def bootstrap_one_dataset(target, background, labels, n_bootstrap=N_BOOTSTRAP):
    rng = np.random.default_rng(RANDOM_STATE)
    n = target.shape[0]
    n_sub = int(n * SUBSAMPLE_FRAC)
    has_labels = labels is not None

    metrics = ["trust_k15"]
    if has_labels:
        metrics += ["knn_acc", "silhouette"]

    results = {m: {met: [] for met in metrics} for m in METHODS}

    for i in range(n_bootstrap):
        if (i + 1) % 10 == 0:
            print(f"    round {i+1}/{n_bootstrap}")

        idx = rng.choice(n, size=n_sub, replace=False)
        sub = target[idx]
        sub_labels = labels[idx] if has_labels else None

        projs = {
            "PCA": run_pca(sub),
            "UMAP": run_umap(sub),
            "ρPCA": run_ppca(sub, background),
        }

        for method, proj in projs.items():
            k = min(15, n_sub - 2)
            t = float(sklearn_trustworthiness(sub, proj, n_neighbors=k, metric="euclidean"))
            results[method]["trust_k15"].append(t)

            if has_labels:
                clf = KNeighborsClassifier(n_neighbors=k)
                loo = LeaveOneOut()
                correct = sum(
                    (clf.fit(proj[tr], sub_labels[tr]).predict(proj[te]) == sub_labels[te]).sum()
                    for tr, te in loo.split(proj)
                )
                results[method]["knn_acc"].append(correct / len(sub_labels))

                if len(np.unique(sub_labels)) >= 2:
                    results[method]["silhouette"].append(
                        float(silhouette_score(proj, sub_labels))
                    )
                else:
                    results[method]["silhouette"].append(float("nan"))

    return results


def load_3ftx():
    target, ids = load_h5(Path("data/3ftx/tmp/prot_t5.h5"))
    bg, _ = load_h5(Path("data/toxprot/prot_t5.h5"))
    df = pd.read_csv("data/3ftx/3FTx_accession.csv")
    id_set = set(df["identifier"])
    mask = [i for i, k in enumerate(ids) if k in id_set]
    target = target[mask]
    ids = [ids[i] for i in mask]
    df = df.set_index("identifier").loc[ids].reset_index()
    return target, bg, df["group"].values


def load_swissprot_ec(n_sample=500):
    target, ids = load_h5(Path("data/swissprot_rr/prot_t5.h5"))
    bg, _ = load_h5(Path("data/swissprot_rr/background_length_extremes_20pct.h5"))
    df = pd.read_csv("data/swissprot_rr/eval_labels.tsv", sep="\t")
    df = df[df["ec_top"].notna() & (df["ec_top"] != "no_ec")]
    id_to_idx = {k: i for i, k in enumerate(ids)}
    df = df[df["identifier"].isin(id_to_idx)]

    rng = np.random.default_rng(RANDOM_STATE)
    if len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=RANDOM_STATE)

    indices = [id_to_idx[i] for i in df["identifier"]]
    return target[indices], bg, df["ec_top"].values


def load_pla2g2():
    target, _ = load_h5(Path("data/pla2g2/prot_t5.h5"))
    bg, _ = load_h5(Path("data/toxprot/prot_t5.h5"))
    return target, bg, None


def load_toxprot():
    target, _ = load_h5(Path("data/toxprot/prot_t5.h5"))
    bg, _ = load_h5(Path("data/swissprot_rr/prot_t5.h5"))
    return target, bg, None


def plot_cross_dataset(all_results, metric, ylabel, title, filename, ylim=(0, 1.05)):
    datasets = [ds for ds in all_results if metric in next(iter(all_results[ds].values()))]
    if not datasets:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 2.5), 4.5))
    x = np.arange(len(datasets))
    width = 0.22
    offsets = {"PCA": -width, "UMAP": 0, "ρPCA": width}

    for method in METHODS:
        means, errs_lo, errs_hi, positions = [], [], [], []
        for j, ds in enumerate(datasets):
            vals = all_results[ds][method][metric]
            if not vals:
                continue
            mean, lo, hi = compute_ci(vals)
            means.append(mean)
            errs_lo.append(mean - lo)
            errs_hi.append(hi - mean)
            positions.append(j)

        pos = np.array(positions) + offsets[method]
        ax.bar(pos, means, width, label=method, color=COLORS[method], edgecolor="none")
        ax.errorbar(pos, means, yerr=[errs_lo, errs_hi],
                    fmt="none", capsize=5, capthick=1, ecolor="#333333", linewidth=1)

        for p, val in zip(pos, means):
            y_off = 0.015 if val >= 0 else -0.015
            va = "bottom" if val >= 0 else "top"
            ax.text(p, val + y_off, f"{val:.2f}", ha="center", va=va,
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=13, pad=10)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / filename
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loaders = {
        "3FTx": load_3ftx,
        "SwissProt EC": load_swissprot_ec,
        "PLA2G2": load_pla2g2,
        "ToxProt": load_toxprot,
    }

    all_results = {}
    for ds_name, loader in loaders.items():
        print(f"\n{'='*50}")
        print(f"  {ds_name}")
        print(f"{'='*50}")
        target, bg, labels = loader()
        print(f"  {target.shape[0]} proteins, labels: {'yes' if labels is not None else 'no'}")
        results = bootstrap_one_dataset(target, bg, labels)
        all_results[ds_name] = results

        for method in METHODS:
            print(f"  {method}:")
            for met in results[method]:
                mean, lo, hi = compute_ci(results[method][met])
                print(f"    {met}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    # Save CSV
    rows = []
    for ds in all_results:
        for method in METHODS:
            for met in all_results[ds][method]:
                for i, val in enumerate(all_results[ds][method][met]):
                    rows.append({"dataset": ds, "method": method, "metric": met, "round": i, "value": val})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "cross_dataset_benchmarks.csv", index=False)

    plot_cross_dataset(all_results, "trust_k15", "Trustworthiness",
                       "Trustworthiness (k=15)", "slide_cross_trustworthiness.png",
                       ylim=(0.55, 1.02))

    plot_cross_dataset(all_results, "knn_acc", "kNN Accuracy",
                       "kNN Classification Accuracy (k=15)", "slide_cross_knn.png",
                       ylim=(0.3, 1.02))

    sil_min = 1.0
    for ds in all_results:
        for method in METHODS:
            vals = all_results[ds][method].get("silhouette", [])
            if vals:
                _, lo, _ = compute_ci(vals)
                sil_min = min(sil_min, lo)
    sil_floor = min(-0.05, sil_min - 0.05)

    plot_cross_dataset(all_results, "silhouette", "Silhouette Score",
                       "Silhouette Score", "slide_cross_silhouette.png",
                       ylim=(sil_floor, max(0.3, -sil_floor)))


if __name__ == "__main__":
    main()
