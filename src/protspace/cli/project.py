"""protspace project — dimensionality reduction on HDF5 embeddings."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from protspace.cli.app import app, setup_logging
from protspace.cli.common_options import (
    Metric,
    Opt_Eps,
    Opt_Fasta,
    Opt_FpRatio,
    Opt_LearningRate,
    Opt_MaxIter,
    Opt_Methods,
    Opt_Metric,
    Opt_MinDist,
    Opt_MnRatio,
    Opt_NInit,
    Opt_NNeighbors,
    Opt_Perplexity,
    Opt_PpcaBackground,
    Opt_RandomState,
    Opt_RegularizationMu,
    Opt_Similarity,
    Opt_StandardScale,
    Opt_Verbose,
    PpcaStrategy,
    Opt_PpcaStrategy,
    Opt_PpcaTargetAnnotation,
    Opt_PpcaTargetValues,
    Opt_PpcaStratifyBy,
    Opt_PpcaMatchLength,
    Opt_PpcaSamplesPerTarget,
)

logger = logging.getLogger(__name__)


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(v.strip() for v in value.split(",") if v.strip())


@app.command()
def project(
    input: Annotated[
        list[str],
        typer.Option(
            "-i",
            "--input",
            help="HDF5 file(s). Repeat for multi-embedding. Colon syntax: -i file.h5:name",
        ),
    ],
    methods: Opt_Methods = None,
    output: Annotated[
        Path,
        typer.Option(
            "-o", "--output", help="Output directory for projection parquet files."
        ),
    ] = Path("."),
    similarity: Opt_Similarity = False,
    fasta: Opt_Fasta = None,
    metric: Opt_Metric = Metric.euclidean,
    random_state: Opt_RandomState = 42,
    n_neighbors: Opt_NNeighbors = 25,
    min_dist: Opt_MinDist = 0.1,
    perplexity: Opt_Perplexity = 30.0,
    learning_rate: Opt_LearningRate = 200.0,
    mn_ratio: Opt_MnRatio = 0.5,
    fp_ratio: Opt_FpRatio = 2.0,
    n_init: Opt_NInit = 4,
    max_iter: Opt_MaxIter = 300,
    eps: Opt_Eps = 1e-3,
    regularization_mu: Opt_RegularizationMu = 1e-3,
    ppca_strategy: Opt_PpcaStrategy = PpcaStrategy.pool,
    ppca_target_annotation: Opt_PpcaTargetAnnotation = None,
    ppca_target_values: Opt_PpcaTargetValues = None,
    ppca_stratify_by: Opt_PpcaStratifyBy = None,
    ppca_match_length: Opt_PpcaMatchLength = False,
    ppca_samples_per_target: Opt_PpcaSamplesPerTarget = 3,
    ppca_background: Opt_PpcaBackground = None,
    standard_scale: Opt_StandardScale = True,
    verbose: Opt_Verbose = 0,
) -> None:
    """Run dimensionality reduction on HDF5 embeddings.

    \b
    Outputs projections_metadata.parquet and projections_data.parquet
    to the output directory.
    """
    setup_logging(verbose)

    from collections import Counter

    import pyarrow.parquet as pq

    from protspace.cli.prepare import _parse_input_specs
    from protspace.data.loaders import EmbeddingSet, compute_similarity, load_h5
    from protspace.data.loaders.embedding_set import format_projection_name
    from protspace.data.processors.base_processor import BaseProcessor
    from protspace.data.processors.pipeline import (
        ReducerParams,
        _load_background_h5,
        _run_with_overridden_config,
        disambiguation_suffix,
        parse_methods_arg,
    )
    from protspace.utils import get_reducers
    from protspace.utils.constants import MDS_NAME, PPCA_NAME

    input_specs = _parse_input_specs(input)
    embedding_sets: list[EmbeddingSet] = []

    for path, name_override in input_specs:
        embedding_sets.append(load_h5([path], name_override=name_override))

    if not embedding_sets:
        raise typer.BadParameter("No valid HDF5 files found.")

    if similarity:
        if fasta is None:
            raise typer.BadParameter("--similarity requires --fasta.")
        sim_set = compute_similarity(fasta, embedding_sets[0].headers)
        embedding_sets.append(sim_set)

    from dataclasses import asdict

    method_specs = parse_methods_arg(methods or ["pca2"])

    # Resolve external background early so we can warn upfront and load once.
    bg_path_str = str(ppca_background) if ppca_background else ""
    bg_data = None
    if bg_path_str:
        # Unpack array since it now returns a tuple (arr, headers)
        bg_data, _ = _load_background_h5(Path(bg_path_str))

    reducer_params = ReducerParams(
        metric=metric.value,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        perplexity=perplexity,
        learning_rate=learning_rate,
        mn_ratio=mn_ratio,
        fp_ratio=fp_ratio,
        n_init=n_init,
        max_iter=max_iter,
        eps=eps,
        regularization_mu=regularization_mu,
        background_strategy=ppca_strategy.value,
        standard_scale=standard_scale,
        background_path=bg_path_str,
        target_annotation=ppca_target_annotation or "",
        target_values=_parse_csv(ppca_target_values),
        stratify_by=_parse_csv(ppca_stratify_by),
        match_length=ppca_match_length,
        samples_per_target=ppca_samples_per_target,
    )
    global_params = asdict(reducer_params)
    reducers = get_reducers()
    base = BaseProcessor(global_params, reducers)

    method_counts = Counter((s.method, s.dims) for s in method_specs)

    all_reductions = []
    headers = embedding_sets[0].headers
    for emb_set in embedding_sets:
        for spec in method_specs:
            method, dims = spec.method, spec.dims
            if emb_set.precomputed and method != MDS_NAME:
                logger.warning(
                    f"Skipping {method} for '{emb_set.name}' (only MDS for precomputed)"
                )
                continue
            if method not in reducers:
                logger.warning(f"Unknown method: {method}. Skipping.")
                continue

            effective_params = {**global_params, **spec.overrides_dict}
            if emb_set.precomputed:
                effective_params["precomputed"] = True

            # ρPCA: route external background array if early-loaded.
            if method == PPCA_NAME:
                if bg_data is not None:
                    effective_params["background_data"] = bg_data

            logger.info(f"Applying {method.upper()}{dims} to '{emb_set.name}'")
            reduction = _run_with_overridden_config(
                base, effective_params, method, dims, emb_set.data
            )
            reduction["name"] = format_projection_name(
                emb_set.name,
                method,
                dims,
                disambiguation_suffix(spec, method_counts),
            )
            all_reductions.append(reduction)

    output.mkdir(parents=True, exist_ok=True)

    metadata_table = base._create_projections_metadata_table(all_reductions)
    data_table = base._create_projections_data_table(all_reductions, headers)

    pq.write_table(metadata_table, str(output / "projections_metadata.parquet"))
    pq.write_table(data_table, str(output / "projections_data.parquet"))

    typer.echo(f"Saved {len(all_reductions)} projections to {output}")