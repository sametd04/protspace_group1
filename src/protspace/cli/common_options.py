"""Shared Typer option type aliases for CLI commands."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer


class Metric(str, Enum):
    euclidean = "euclidean"
    cosine = "cosine"
    manhattan = "manhattan"

class PpcaStrategy(str, Enum):
    pool = "pool"
    complement = "complement"
    length_matched = "length_matched"
    stratified = "stratified"
    mixed = "mixed"
    isolate = "isolate"

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
    typer.Option("-s", "--similarity", help="Compute sequence similarity DR via MMseqs2.", rich_help_panel="Projection"),
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
    typer.Option(help="UMAP/PaCMAP/LocalMAP neighbors. Larger=more global.", rich_help_panel="Projection", min=2),
]
Opt_MinDist = Annotated[
    float,
    typer.Option(help="UMAP min distance.", rich_help_panel="Projection", min=0.0, max=0.99),
]
Opt_Perplexity = Annotated[
    float,
    typer.Option(help="t-SNE perplexity. Should be < n_samples/3.", rich_help_panel="Projection", min=5.0),
]
Opt_LearningRate = Annotated[
    float,
    typer.Option(help="t-SNE learning rate.", rich_help_panel="Projection", min=1.0),
]
Opt_MnRatio = Annotated[
    float,
    typer.Option(help="PaCMAP/LocalMAP mid-near ratio.", rich_help_panel="Projection", min=0.0, max=1.0),
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
Opt_PpcaBackground = Annotated[
    Path | None,
    typer.Option(
        "--ppca-background",
        help="Path to an HDF5 file holding background embeddings for ρPCA (e.g. SwissProt). Identifiers are not required to overlap with the target.",
        rich_help_panel="Projection",
    ),
]
Opt_StandardScale = Annotated[
    bool,
    typer.Option(
        "--standard-scale/--no-standard-scale",
        help="Per-set standard-scale target and background before computing covariances (paper convention).",
        rich_help_panel="Projection",
    ),
]
Opt_PpcaStrategy = Annotated[
    PpcaStrategy,
    typer.Option(
        "--ppca-strategy",
        help="ρPCA background construction strategy. 'isolate' builds an artifact-only variance background from the pool.",
        rich_help_panel="Projection",
    ),
]
Opt_PpcaTargetAnnotation = Annotated[str | None, typer.Option("--ppca-target-annotation", help="For --ppca-strategy=complement.", rich_help_panel="Projection")]
Opt_PpcaTargetValues = Annotated[str | None, typer.Option("--ppca-target-values", help="For --ppca-strategy=complement.", rich_help_panel="Projection")]
Opt_PpcaStratifyBy = Annotated[
    str | None,
    typer.Option(
        "--ppca-stratify-by",
        help="For --ppca-strategy=stratified, mixed, or isolate: comma-separated annotation columns to isolate/stratify the background by. E.g. 'signal_peptide'.",
        rich_help_panel="Projection",
    ),
]
Opt_PpcaMatchLength = Annotated[bool, typer.Option("--ppca-match-length/--no-ppca-match-length", help="Include sequence length.", rich_help_panel="Projection")]
Opt_PpcaSamplesPerTarget = Annotated[
    int,
    typer.Option(
        "--ppca-samples-per-target",
        help="For length_matched / stratified / mixed / isolate: number of pool samples drawn per target sample.",
        rich_help_panel="Projection",
        min=1,
    ),
]
Opt_BatchSize = Annotated[int, typer.Option(help="Sequences per Biocentral API call.", rich_help_panel="Embedding")]
Opt_Fasta = Annotated[Path | None, typer.Option("-f", "--fasta", help="FASTA for -s/--similarity when input is HDF5.", rich_help_panel="Input")]