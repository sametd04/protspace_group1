"""Per-protein neighborhood preservation analysis for 3FTx dataset.

Computes per-protein trustworthiness (k-NN overlap) for PCA, UMAP, and pPCA,
identifies outlier proteins, and produces scatter plots + distribution charts.
"""

import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- Data Loading ---


def load_h5_embeddings_with_keys(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load embeddings and their protein identifiers from an H5 file."""
    arrays = []
    keys = []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            keys.append(key)
            arrays.append(f[key][:])
    return np.array(arrays, dtype=np.float64), keys


def load_labels(csv_path: Path, identifiers: list[str]) -> pd.DataFrame:
    """Load and align labels with embedding order. Returns aligned DataFrame."""
    df = pd.read_csv(csv_path)
    id_to_idx = {ident: i for i, ident in enumerate(identifiers)}
    df = df[df["identifier"].isin(id_to_idx)]
    df = df.copy()
    df["_order"] = df["identifier"].map(id_to_idx)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return df


# --- Dimensionality Reduction ---


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


# --- Per-Protein Scoring ---


def compute_per_protein_scores(high_d: np.ndarray, low_d: np.ndarray, k: int = 15) -> np.ndarray:
    """Compute per-protein neighborhood preservation score.

    For each protein: |neighbors_highD ∩ neighbors_2D| / k
    """
    n_samples = high_d.shape[0]
    actual_k = min(k, n_samples - 1)

    nn_high = NearestNeighbors(n_neighbors=actual_k + 1, metric="euclidean")
    nn_high.fit(high_d)
    high_neighbors = nn_high.kneighbors(high_d, return_distance=False)[:, 1:]

    nn_low = NearestNeighbors(n_neighbors=actual_k + 1, metric="euclidean")
    nn_low.fit(low_d)
    low_neighbors = nn_low.kneighbors(low_d, return_distance=False)[:, 1:]

    scores = np.zeros(n_samples)
    for i in range(n_samples):
        overlap = len(set(high_neighbors[i]) & set(low_neighbors[i]))
        scores[i] = overlap / actual_k

    return scores


def identify_outliers(scores: np.ndarray, identifiers: list[str], percentile: float = 10.0) -> list[int]:
    """Return indices of proteins in the bottom percentile by score."""
    threshold = np.percentile(scores, percentile)
    return [i for i in range(len(scores)) if scores[i] <= threshold]


# --- Visualization ---


def plot_projection_colored_by_score(
    projection: np.ndarray,
    scores: np.ndarray,
    method_name: str,
    output_path: Path,
    outlier_indices: list[int] | None = None,
):
    """Scatter plot of 2D projection colored by per-protein score."""
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=scores,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        s=20,
        alpha=0.8,
    )

    if outlier_indices:
        ax.scatter(
            projection[outlier_indices, 0],
            projection[outlier_indices, 1],
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            s=40,
        )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Neighborhood Preservation (k=15)")
    ax.set_title(f"{method_name} — Per-Protein Neighborhood Preservation")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_score_distributions(scores_dict: dict[str, np.ndarray], output_path: Path):
    """Overlaid histograms of score distributions for all methods."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"PCA": "#4C72B0", "UMAP": "#55A868", "pPCA-ToxProt": "#C44E52"}

    for name, scores in scores_dict.items():
        ax.hist(scores, bins=20, alpha=0.5, label=f"{name} (mean={scores.mean():.3f})",
                color=colors.get(name, None), edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Per-Protein Score (k-NN overlap)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Per-Protein Neighborhood Preservation")
    ax.legend()
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_outlier_group_distribution(
    outlier_indices: list[int],
    labels_df: pd.DataFrame,
    output_path: Path,
):
    """Compare group distribution of pPCA outliers vs full dataset."""
    all_groups = labels_df["group"].value_counts(normalize=True)
    outlier_groups = labels_df.iloc[outlier_indices]["group"].value_counts(normalize=True)

    # Keep only groups that appear in outliers
    groups = outlier_groups.index.tolist()
    all_props = [all_groups.get(g, 0) for g in groups]
    outlier_props = [outlier_groups.get(g, 0) for g in groups]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, all_props, width, label="All proteins", color="#4C72B0", alpha=0.7)
    ax.bar(x + width / 2, outlier_props, width, label="pPCA outliers (bottom 10%)", color="#C44E52", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Proportion")
    ax.set_title("pPCA Outliers: Which Toxin Groups Are Over-/Under-represented?")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# --- Main ---


def main():
    target_path = Path("data/3ftx/tmp/prot_t5.h5")
    background_path = Path("data/toxprot/prot_t5.h5")
    labels_path = Path("data/3ftx/3FTx_accession.csv")
    output_dir = Path("src/protspace/benchmark/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading embeddings...")
    target_data, target_ids = load_h5_embeddings_with_keys(target_path)
    background_data, _ = load_h5_embeddings_with_keys(background_path)
    print(f"  Target: {target_data.shape[0]} proteins, Background: {background_data.shape[0]} proteins")

    # Load and align labels
    labels_df = load_labels(labels_path, target_ids)
    matched_ids = labels_df["identifier"].tolist()
    print(f"  Matched with labels: {len(matched_ids)}/{len(target_ids)} proteins")

    # Filter embeddings to only those with labels
    mask = [i for i, ident in enumerate(target_ids) if ident in set(matched_ids)]
    target_data = target_data[mask]
    target_ids = [target_ids[i] for i in mask]
    print(f"  Using {target_data.shape[0]} proteins for analysis")

    # Run projections
    print("\nRunning dimensionality reduction...")
    projections = {}

    print("  PCA...")
    projections["PCA"] = run_pca(target_data)

    print("  UMAP...")
    projections["UMAP"] = run_umap(target_data)

    print("  pPCA-ToxProt...")
    projections["pPCA-ToxProt"] = run_ppca_external(target_data, background_data)

    # Compute per-protein scores
    print("\nComputing per-protein scores (k=15)...")
    scores = {}
    for name, proj in projections.items():
        scores[name] = compute_per_protein_scores(target_data, proj, k=15)
        print(f"  {name}: mean={scores[name].mean():.3f}, min={scores[name].min():.3f}, max={scores[name].max():.3f}")

    # Build results CSV
    results_df = pd.DataFrame({
        "identifier": target_ids,
        "group": labels_df["group"].values,
        "sub_group": labels_df["sub_group"].values,
        "pca_score": scores["PCA"],
        "umap_score": scores["UMAP"],
        "ppca_score": scores["pPCA-ToxProt"],
    })
    csv_path = output_dir / "per_protein_trustworthiness_3ftx.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Identify outliers (bottom 10%)
    print("\nOutliers (bottom 10% per method):")
    for name in projections:
        outlier_idx = identify_outliers(scores[name], target_ids, percentile=10.0)
        col = {"PCA": "pca_score", "UMAP": "umap_score", "pPCA-ToxProt": "ppca_score"}[name]
        outlier_df = results_df.iloc[outlier_idx].sort_values(col)
        print(f"\n  {name} — {len(outlier_idx)} outliers:")
        print(f"    Worst 5: {outlier_df['identifier'].head(5).tolist()}")
        print(f"    Groups:  {outlier_df['group'].value_counts().head(5).to_dict()}")

    # Generate plots
    print("\nGenerating plots...")
    for name, proj in projections.items():
        outlier_idx = identify_outliers(scores[name], target_ids, percentile=10.0)
        safe_name = name.lower().replace("-", "_").replace(" ", "_")
        plot_projection_colored_by_score(
            proj, scores[name], name,
            output_dir / f"per_protein_{safe_name}_3ftx.png",
            outlier_indices=outlier_idx,
        )

    plot_score_distributions(scores, output_dir / "score_distributions_3ftx.png")

    ppca_outliers = identify_outliers(scores["pPCA-ToxProt"], target_ids, percentile=10.0)
    plot_outlier_group_distribution(ppca_outliers, labels_df, output_dir / "outlier_groups_ppca_3ftx.png")

    print("\nDone! All outputs in:", output_dir)


if __name__ == "__main__":
    main()
