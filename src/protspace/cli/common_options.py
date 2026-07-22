"""Shared Typer option type aliases for CLI commands.

Import these in any CLI command to avoid duplicating option definitions.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer


class Metric(str, Enum):
    euclidean = "euclidean"
    cosine = "cosine"
    manhattan = "manhattan"


class NuisanceBlockNormalization(str, Enum):
    none = "none"
    trace = "trace"
    row_count = "row_count"


# ---------------------------------------------------------------------------
# Shared option types
# ---------------------------------------------------------------------------

Opt_Verbose = Annotated[
    int,
    typer.Option("-v", "--verbose", count=True, help="Verbosity: -v=INFO, -vv=DEBUG."),
]

# Projection options (shared by prepare and project)
Opt_Methods = Annotated[
    list[str] | None,
    typer.Option(
        "-m",
        "--methods",
        help=(
            "DR methods. Comma-sep or repeat: -m pca2,umap2 or -m pca2 -m umap2. "
            "Includes rhopca2/rhopca3 when --rhopca-background or --nuisance is supplied. "
            "Inline params: -m 'umap2:n_neighbors=50;min_dist=0.1'. "
            "Chain a pre-reduction with '>': -m 'rhopca50>umap2' reduces to the top-50 "
            "rhoPCA components, then runs UMAP (final stage must be 2- or 3-D)."
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

# ρPCA options. Only the final clean pathways are exposed:
#   1. --rhopca-background for an already-built explicit background
#   2. --nuisance for annotation-defined backgrounds inside `prepare`
Opt_RhoPcaBackground = Annotated[
    Path | None,
    typer.Option(
        "--rhopca-background",
        "--ppca-background",  # deprecated alias
        help=(
            "Explicit ρPCA background HDF5. Required for rhopca2/rhopca3 in "
            "`project`; optional in `prepare` when --nuisance is supplied."
        ),
        rich_help_panel="ρPCA",
    ),
]
# Deprecated alias for the option type (pre-rename import name).
Opt_PpcaBackground = Opt_RhoPcaBackground
Opt_Nuisance = Annotated[
    list[str] | None,
    typer.Option(
        "--nuisance",
        help=(
            "Annotation-defined ρPCA nuisance specification. Repeatable. "
            "Examples: --nuisance 'sp_in_embedding:type=binary;missing=zero' "
            "or --nuisance 'length:type=continuous;transform=log1p;basis=spline'. "
            "Only available in `prepare`, because it needs annotation context."
        ),
        rich_help_panel="ρPCA",
    ),
]
Opt_RegularizationMu = Annotated[
    float,
    typer.Option(
        "--regularization-mu",
        help="Tikhonov regularization μ added to the ρPCA background covariance.",
        rich_help_panel="ρPCA",
        min=0.0,
    ),
]
Opt_StandardScale = Annotated[
    bool,
    typer.Option(
        "--standard-scale/--no-standard-scale",
        help=(
            "Column-standardize target/background before the ρPCA eigensolve. "
            "Default is no standard scaling for ρPCA."
        ),
        rich_help_panel="ρPCA",
    ),
]
Opt_NuisanceRidgeAlpha = Annotated[
    float | None,
    typer.Option(
        "--nuisance-ridge-alpha",
        help="Ridge α for annotation→embedding nuisance models. Omit to auto-tune per "
             "nuisance via cross-validated GCV; pass a value to fix it.",
        rich_help_panel="ρPCA",
        min=0.0,
    ),
]
Opt_NuisanceCrossFit = Annotated[
    int,
    typer.Option(
        "--nuisance-cross-fit",
        help="Default cross-fitting folds for nuisance models. Use 0 or 1 to fit all rows.",
        rich_help_panel="ρPCA",
        min=0,
    ),
]
Opt_NuisanceBlockNormalization = Annotated[
    NuisanceBlockNormalization,
    typer.Option(
        "--nuisance-block-normalization",
        help="Per-nuisance block normalization before signing and stacking.",
        rich_help_panel="ρPCA",
    ),
]
Opt_NuisanceWriteBackground = Annotated[
    bool,
    typer.Option(
        "--nuisance-write-background/--no-nuisance-write-background",
        help="Write background.h5, manifest, diagnostics, and effect summary for --nuisance runs.",
        rich_help_panel="ρPCA",
    ),
]

# Embedding options (shared by prepare and embed)
Opt_BatchSize = Annotated[
    int,
    typer.Option(
        help="Sequences per Biocentral API call.", rich_help_panel="Embedding"
    ),
]

# Input options (shared by prepare and project)
Opt_Fasta = Annotated[
    Path | None,
    typer.Option(
        "-f",
        "--fasta",
        help="FASTA for -s/--similarity when input is HDF5.",
        rich_help_panel="Input",
    ),
]
