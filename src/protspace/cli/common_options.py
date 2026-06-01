"""Shared Typer option type aliases for CLI commands."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer


class Metric(str, Enum):
    euclidean = "euclidean"
    cosine = "cosine"
    manhattan = "manhattan"


class PpcaMode(str, Enum):
    """User-facing ρPCA background modes.

    explicit   : use an external HDF5 embedding set as Σ_B input.
    annotation : select background rows from the current dataset by annotation.
    derived    : build nuisance-only sequences from the current dataset and
                 embed them as the background.
    """

    explicit = "explicit"
    annotation = "annotation"
    derived = "derived"


Opt_Verbose = Annotated[
    int,
    typer.Option("-v", "--verbose", count=True, help="Verbosity: -v=INFO, -vv=DEBUG."),
]

Opt_Methods = Annotated[
    list[str] | None,
    typer.Option(
        "-m",
        "--methods",
        help="DR methods. Comma-sep or repeat: -m pca2,umap2 or -m pca2 -m umap2. Inline params: -m 'umap2:n_neighbors=50;min_dist=0.1'.",
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
        help="UMAP min distance.",
        rich_help_panel="Projection",
        min=0.0,
        max=0.99,
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
    typer.Option(help="PaCMAP/LocalMAP further ratio.", rich_help_panel="Projection", min=0.0),
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
        help="ρPCA Tikhonov regularization μ added to the background covariance Σ_B. Default 1e-3.",
        rich_help_panel="Projection",
        min=0.0,
    ),
]
Opt_PpcaMode = Annotated[
    PpcaMode,
    typer.Option(
        "--ppca-mode",
        help=(
            "ρPCA background mode: explicit=external HDF5 background; "
            "annotation=select background rows from current dataset by annotation; "
            "derived=embed nuisance-only sequence segments from the current dataset."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_PpcaBackground = Annotated[
    Path | None,
    typer.Option(
        "--ppca-background",
        help=(
            "Explicit ρPCA background HDF5 file. Required for --ppca-mode=explicit. "
            "Identifiers do not need to overlap with the target, but embedding "
            "dimensionality must match."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_PpcaBackgroundAnnotation = Annotated[
    str | None,
    typer.Option(
        "--ppca-background-annotation",
        help=(
            "Annotation column used by --ppca-mode=annotation to select background "
            "rows, and optionally by --ppca-mode=derived to choose which rows receive "
            "derived nuisance-only segments."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_PpcaBackgroundValues = Annotated[
    str | None,
    typer.Option(
        "--ppca-background-values",
        help=(
            "Comma-separated values in --ppca-background-annotation that define the "
            "background. If omitted, non-empty/non-missing/truthy values are selected."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_PpcaDerivedSegmentColumn = Annotated[
    str | None,
    typer.Option(
        "--ppca-derived-segment-column",
        help=(
            "For --ppca-mode=derived: annotation column containing the exact nuisance "
            "sequence to embed as background, e.g. signal_peptide_sequence. This is "
            "preferred when available."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_PpcaDerivedStartColumn = Annotated[
    str | None,
    typer.Option(
        "--ppca-derived-start-column",
        help=(
            "For --ppca-mode=derived: annotation column with 0-based inclusive segment "
            "start. If omitted, --ppca-derived-fixed-start is used."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_PpcaDerivedEndColumn = Annotated[
    str | None,
    typer.Option(
        "--ppca-derived-end-column",
        help=(
            "For --ppca-mode=derived: annotation column with 0-based exclusive segment "
            "end, e.g. SignalP cleavage position converted to Python slicing coordinates."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_PpcaDerivedFixedStart = Annotated[
    int,
    typer.Option(
        "--ppca-derived-fixed-start",
        help="For --ppca-mode=derived: fallback 0-based inclusive segment start.",
        rich_help_panel="Projection",
        min=0,
    ),
]
Opt_PpcaDerivedFixedEnd = Annotated[
    int,
    typer.Option(
        "--ppca-derived-fixed-end",
        help=(
            "For --ppca-mode=derived: fallback 0-based exclusive segment end. "
            "Use 0 to require --ppca-derived-end-column or --ppca-derived-segment-column."
        ),
        rich_help_panel="Projection",
        min=0,
    ),
]
Opt_PpcaDerivedMinLength = Annotated[
    int,
    typer.Option(
        "--ppca-derived-min-length",
        help="For --ppca-mode=derived: discard derived segments shorter than this length.",
        rich_help_panel="Projection",
        min=1,
    ),
]
Opt_PpcaDerivedEmbedder = Annotated[
    str | None,
    typer.Option(
        "--ppca-derived-embedder",
        help=(
            "For --ppca-mode=derived: embedding model used for nuisance-only segments. "
            "Defaults to the current embedding set name, so HDF5 inputs should be named "
            "after their model, e.g. -i prot_t5.h5:prot_t5."
        ),
        rich_help_panel="Projection",
    ),
]
Opt_StandardScale = Annotated[
    bool,
    typer.Option(
        "--standard-scale/--no-standard-scale",
        help="Per-set standard-scale target and background before computing covariances (ρPCA paper convention).",
        rich_help_panel="Projection",
    ),
]
Opt_BatchSize = Annotated[
    int,
    typer.Option(help="Sequences per Biocentral API call.", rich_help_panel="Embedding"),
]
Opt_Fasta = Annotated[
    Path | None,
    typer.Option(
        "-f",
        "--fasta",
        help="FASTA for -s/--similarity when input is HDF5; also supplies sequences for --ppca-mode=derived.",
        rich_help_panel="Input",
    ),
]