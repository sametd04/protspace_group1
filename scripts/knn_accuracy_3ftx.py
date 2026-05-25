"""kNN classification accuracy benchmark for 3FTx dataset.

Trains a k-nearest-neighbors classifier (k=15) on the 2D projection using
toxin group labels and evaluates via leave-one-out cross-validation.
Tests whether the projection preserves class-separable structure.
"""

import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

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


# --- Evaluation ---


def compute_knn_accuracy_loo(projection: np.ndarray, labels: np.ndarray, k: int = 15) -> float:
    """Leave-one-out kNN classification accuracy on the 2D projection."""
    loo = LeaveOneOut()
    correct = 0
    total = 0

    for train_idx, test_idx in loo.split(projection):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(projection[train_idx], labels[train_idx])
        pred = clf.predict(projection[test_idx])
        correct += (pred == labels[test_idx]).sum()
        total += len(test_idx)

    return round(correct / total, 4)


# --- Main ---


def main():
    target_path = Path("data/3ftx/tmp/prot_t5.h5")
    background_path = Path("data/toxprot/prot_t5.h5")
    labels_path = Path("data/3ftx/3FTx_accession.csv")
    output_path = Path("src/protspace/benchmark/results/knn_accuracy_3ftx.csv")

    print("Loading embeddings...")
    target_data, target_ids = load_h5_embeddings_with_keys(target_path)
    background_data, _ = load_h5_embeddings_with_keys(background_path)
    print(f"  Target: {target_data.shape[0]} proteins, Background: {background_data.shape[0]} proteins")

    # Load and align labels (inner join)
    labels_df = load_labels(labels_path, target_ids)
    matched_ids = labels_df["identifier"].tolist()
    print(f"  Matched with labels: {len(matched_ids)}/{len(target_ids)} proteins")

    # Filter embeddings to only those with labels
    mask = [i for i, ident in enumerate(target_ids) if ident in set(matched_ids)]
    target_data = target_data[mask]
    target_ids = [target_ids[i] for i in mask]
    labels = labels_df["group"].values
    print(f"  Using {target_data.shape[0]} proteins for analysis")
    print(f"  Unique groups: {len(np.unique(labels))}")

    # Run projections
    print("\nRunning dimensionality reduction...")
    projections = {}

    print("  PCA...")
    projections["PCA"] = run_pca(target_data)

    print("  UMAP...")
    projections["UMAP"] = run_umap(target_data)

    print("  pPCA-ToxProt...")
    projections["pPCA-ToxProt"] = run_ppca_external(target_data, background_data)

    # Compute kNN accuracy (LOO CV, k=15)
    print("\nComputing kNN accuracy (LOO CV, k=15)...")
    results = []
    for name, proj in projections.items():
        acc = compute_knn_accuracy_loo(proj, labels, k=15)
        print(f"  {name}: accuracy = {acc}")
        results.append({"method": name, "knn_accuracy": acc})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
