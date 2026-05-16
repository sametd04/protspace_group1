"""Shared Typer option type aliases for CLI commands."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer


class Metric(str, Enum):
    euclidean = "euclidean"
    cosine = "cosine"
    manhattan = "manhattan"


class BackgroundStrategy(str, Enum):
    """Sampling policy for the ρPCA background set."""

    external = "external"
    random = "random"
    uniform = "uniform"
    outlier = "outlier"


class Kernel(str, Enum):
    gaussian = "gaussian"
    inverse_distance = "inverse_distance"
    linear = "linear"


class KernelSource(str, Enum):
    embedding = "embedding"
    similarity = "similarity"
    precomputed = "precomputed"


# ---------------------------------------------------------------------------
# Shared option types
# ---------------------------------------------------------------------------

Opt_Verbose = Annotated[
    int,
    typer.Option("-v", "--verbose", count=True, help="Verbosity: -v=INFO, -vv=DEBUG."),
]

Opt_Methods = Annotated[
    list[str] | None,
    typer.Option(
        "-m",
        "--methods",
        help=(
            "DR methods. Comma-sep or repeat: -m pca2,umap2 or -m pca2 -m umap2. "
            "Inline params: -m 'umap2:n_neighbors=50;min_dist=0.1'."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_Similarity = Annotated[
    bool,
    typer.Option(
        "-s",
        "--similarity",
        help="Compute sequence similarity DR via MMseqs2.",
        rich_help_panel="Projection",
    ),
]
Opt_Metric = Annotated[
    Metric,
    typer.Option(help="Distance metric for UMAP/t-SNE.", rich_help_panel="Projection"),
]
Opt_RandomState = Annotated[
    int,
    typer.Option(help="Random seed.", rich_help_panel="Projection"),
]
Opt_NNeighbors = Annotated[
    int,
    typer.Option(
        help="UMAP/PaCMAP/LocalMAP neighbors. Larger=more global.",
        rich_help_panel="Projection",
        min=2,
    ),
]
Opt_MinDist = Annotated[
    float,
    typer.Option(
        help="UMAP min distance.", rich_help_panel="Projection", min=0.0, max=0.99
    ),
]
Opt_Perplexity = Annotated[
    float,
    typer.Option(
        help="t-SNE perplexity. Should be < n_samples/3.",
        rich_help_panel="Projection",
        min=5.0,
    ),
]
Opt_LearningRate = Annotated[
    float,
    typer.Option(help="t-SNE learning rate.", rich_help_panel="Projection", min=1.0),
]
Opt_MnRatio = Annotated[
    float,
    typer.Option(
        help="PaCMAP/LocalMAP mid-near ratio.",
        rich_help_panel="Projection",
        min=0.0,
        max=1.0,
    ),
]
Opt_FpRatio = Annotated[
    float,
    typer.Option(
        help="PaCMAP/LocalMAP further ratio.",
        rich_help_panel="Projection",
        min=0.0,
    ),
]
Opt_NInit = Annotated[
    int,
    typer.Option(help="MDS initializations.", rich_help_panel="Projection", min=1),
]
Opt_MaxIter = Annotated[
    int,
    typer.Option(help="MDS max iterations.", rich_help_panel="Projection", min=1),
]
Opt_Eps = Annotated[
    float,
    typer.Option(help="MDS convergence tolerance.", rich_help_panel="Projection"),
]
Opt_RegularizationMu = Annotated[
    float,
    typer.Option(
        "--regularization-mu",
        help=(
            "ρPCA Tikhonov regularization μ added to the background "
            "covariance Σ_B. Must be ≥ 0. Default 1e-3 is safe for PLM "
            "embeddings; reduce to 0 only if the background is well-"
            "conditioned (n_background ≫ n_features)."
        ),
        rich_help_panel="Projection",
        min=0.0,
    ),
]
Opt_BackgroundRatio = Annotated[
    float,
    typer.Option(
        "--background-ratio",
        help=(
            "ρPCA fraction of input samples used as background in "
            "auto-split modes. Ignored when --ppca-background is set. "
            "Must lie strictly in (0, 1)."
        ),
        rich_help_panel="Projection",
        min=0.0,
        max=1.0,
    ),
]
Opt_BackgroundStrategy = Annotated[
    BackgroundStrategy,
    typer.Option(
        "--background-strategy",
        help=(
            "ρPCA background construction policy. 'external' uses the "
            "dataset from --ppca-background; 'random', 'uniform', and "
            "'outlier' partition the target. If --ppca-background is "
            "set, this flag is overridden to 'external'."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_PpcaBackground = Annotated[
    Path | None,
    typer.Option(
        "--ppca-background",
        help=(
            "Path to an HDF5 file holding background embeddings for ρPCA "
            "(canonical mode per Carilli/Jackson/Pachter 2025). Must have "
            "the same embedding dimension as the target. Identifiers are "
            "not required to overlap with the target."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_StandardScale = Annotated[
    bool,
    typer.Option(
        "--standard-scale/--no-standard-scale",
        help=(
            "Per-set standard-scale target and background before computing "
            "covariances (paper convention). Strongly recommended for PLM "
            "embeddings whose dimensions have heterogeneous scale."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_KppcaKernel = Annotated[
    Kernel,
    typer.Option(
        "--kppca-kernel",
        help=(
            "k-ρPCA kernel function. 'gaussian' uses exp(-d²/2h²) with "
            "bandwidth h (see --kppca-kernel-bandwidth); 'inverse_distance' "
            "uses 1/(d+ε); 'linear' uses K=I (reduces to ρPCA, sanity check)."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_KppcaKernelSource = Annotated[
    KernelSource,
    typer.Option(
        "--kppca-kernel-source",
        help=(
            "Source of pairwise distances for the k-ρPCA kernel matrix. "
            "'embedding' (default) uses pairwise Euclidean distances in the "
            "standardised target embedding space — emphasises local pLM "
            "neighbourhoods. 'similarity' uses 1 - MMseqs2 identity from a "
            "similarity matrix (requires --similarity). 'precomputed' loads "
            "the matrix from --kppca-kernel-path."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_KppcaKernelBandwidth = Annotated[
    float,
    typer.Option(
        "--kppca-kernel-bandwidth",
        help=(
            "k-ρPCA Gaussian kernel bandwidth h. 0 (default) auto-resolves "
            "to sqrt(median pairwise distance) per the rhopca reference "
            "heuristic. Ignored for non-Gaussian kernels."
        ),
        rich_help_panel="Projection",
        min=0.0,
    ),
]
Opt_KppcaKernelPath = Annotated[
    Path | None,
    typer.Option(
        "--kppca-kernel-path",
        help=(
            "Path to an n_T × n_T precomputed kernel matrix (.npy, .h5, or "
            ".parquet). Only used when --kppca-kernel-source=precomputed."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_KppcaBackgroundKernel = Annotated[
    bool,
    typer.Option(
        "--kppca-background-kernel/--no-kppca-background-kernel",
        help=(
            "Apply the kernel matrix to the background covariance too. "
            "Off by default — paper 2 keeps Σ_B unweighted."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_BatchSize = Annotated[
    int,
    typer.Option(
        help="Sequences per Biocentral API call.", rich_help_panel="Embedding"
    ),
]
Opt_Fasta = Annotated[
    Path | None,
    typer.Option(
        "-f",
        "--fasta",
        help="FASTA for -s/--similarity when input is HDF5.",
        rich_help_panel="Input",
    ),
]