"""Hyperparameter sweep for pPCA on 3FTx dataset.

Varies regularization_mu (5 values) and standard_scale (True/False)
to measure impact on trustworthiness.
"""

import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness as sklearn_trustworthiness

warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_h5_embeddings(path: Path) -> np.ndarray:
    arrays = []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arrays.append(f[key][:])
    return np.array(arrays, dtype=np.float64)


def run_ppca_with_params(
    target: np.ndarray,
    background: np.ndarray,
    mu: float,
    standard_scale: bool,
) -> np.ndarray:
    from protspace.utils.constants import DimensionReductionConfig
    from protspace.utils.reducers import PPCAReducer

    config = DimensionReductionConfig(
        n_components=2,
        random_state=42,
        background_strategy="external",
        regularization_mu=mu,
        standard_scale=standard_scale,
    )
    object.__setattr__(config, "background_data", background)

    reducer = PPCAReducer(config)
    return reducer.fit_transform(target)


def compute_trustworthiness(high_d: np.ndarray, low_d: np.ndarray) -> float:
    return round(float(sklearn_trustworthiness(high_d, low_d, n_neighbors=15, metric="euclidean")), 4)


def main():
    target_path = Path("data/3ftx/tmp/prot_t5.h5")
    background_path = Path("data/toxprot/prot_t5.h5")
    output_dir = Path("src/protspace/benchmark/results")

    print("Loading embeddings...")
    target_data = load_h5_embeddings(target_path)
    background_data = load_h5_embeddings(background_path)
    print(f"  Target: {target_data.shape[0]} proteins, Background: {background_data.shape[0]} proteins")

    mu_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    scale_values = [True, False]

    results = []
    for scale in scale_values:
        for mu in mu_values:
            print(f"  mu={mu:.0e}, standard_scale={scale}...", end=" ")
            try:
                projection = run_ppca_with_params(target_data, background_data, mu, scale)
                trust = compute_trustworthiness(target_data, projection)
            except Exception as e:
                print(f"FAILED: {e}")
                trust = float("nan")
            print(f"trustworthiness={trust}")
            results.append({
                "regularization_mu": mu,
                "standard_scale": scale,
                "trustworthiness": trust,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = output_dir / "hyperparam_sweep_3ftx.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    print(df.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    for scale in scale_values:
        subset = df[df["standard_scale"] == scale]
        label = f"standard_scale={scale}"
        ax.plot(subset["regularization_mu"], subset["trustworthiness"], "o-", label=label)

    ax.set_xscale("log")
    ax.set_xlabel("regularization_mu")
    ax.set_ylabel("Trustworthiness (k=15)")
    ax.set_title("pPCA Hyperparameter Sweep: Trustworthiness vs μ")
    ax.legend()
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "hyperparam_sweep_3ftx.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
