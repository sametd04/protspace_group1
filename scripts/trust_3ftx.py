import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Load data

def load_h5_embeddings(path: Path) -> np.ndarray:
    """Load all embeddings from an H5 file, upcast to float64."""
    arrays = []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arrays.append(f[key][:])
    return np.array(arrays, dtype=np.float64)


# Dimensionality reduction methods

def run_pca(data: np.ndarray) -> np.ndarray:
    """PCA to 2D."""
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(data)


def run_umap(data: np.ndarray) -> np.ndarray:
    """UMAP to 2D."""
    from umap import UMAP
    return UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    ).fit_transform(data)


def run_ppca_external(target: np.ndarray, background: np.ndarray) -> np.ndarray:
    """ρPCA with ToxProt as external background."""
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



# Evaluation

def compute_trustworthiness(high_d: np.ndarray, low_d: np.ndarray) -> float:
    """Trustworthiness at k=15."""
    score = sklearn_trustworthiness(high_d, low_d, n_neighbors=15, metric="euclidean")
    return round(float(score), 4)


# Main

def main():
    target_path = Path("data/3ftx/tmp/prot_t5.h5")
    background_path = Path("data/toxprot/prot_t5.h5")
    output_path = Path("src/protspace/benchmark/results/trustworthiness_3ftx.csv")

    print("Loading embeddings...")
    target_data = load_h5_embeddings(target_path)
    background_data = load_h5_embeddings(background_path)
    print(f"  Target: {target_data.shape[0]} proteins, Background: {background_data.shape[0]} proteins")

    runs = {
        "PCA": lambda: run_pca(target_data),
        "UMAP": lambda: run_umap(target_data),
        "pPCA-ToxProt": lambda: run_ppca_external(target_data, background_data),
    }

    results = []
    for name, run_fn in runs.items():
        print(f"\nRunning {name}...")
        projection = run_fn()
        trust = compute_trustworthiness(target_data, projection)
        print(f"  → trustworthiness = {trust}")
        results.append({"method": name, "trustworthiness": trust})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
