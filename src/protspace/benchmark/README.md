# ProtSpace Benchmark Module

Benchmark dimensionality reduction methods on protein embeddings with timing and quality metrics.

## Quick Start

```bash
# Install dependencies
uv sync --extra benchmark

# Run benchmark with visualization (dataset selectable via --dataset)
uv run python -m protspace.benchmark.run --dataset 3ftx
uv run python -m protspace.benchmark.run --dataset toxprot

# Alternative benchmark CLI wrapper
uv run python -m protspace.benchmark.cli --data 3ftx --plot

# Visualize existing results
uv run python -m protspace.benchmark.cli --data 3ftx --plot-only
```

This will:
- Load embeddings from `data/3ftx/prot_t5.h5`
- Run all 6 DR methods (PCA, UMAP, t-SNE, PaCMAP, MDS, LocalMAP) twice (normalized & raw)
- Calculate trustworthiness and silhouette scores (if labels available)
- Save metrics to `src/protspace/benchmark/results/3ftx/metrics.csv`
- Generate side-by-side comparison plots in `src/protspace/benchmark/results/3ftx/`

## How benchmark execution works

`run.py` executes this pipeline per dataset:
1. Read embeddings from `data/<dataset>/prot_t5.h5`
2. Optionally read annotation labels from a source `.parquetbundle` (for silhouette)
3. Run all DR methods and compute metrics
4. Write outputs to `src/protspace/benchmark/results/<dataset>/`

Persistence model:
- Results are file-based only (`metrics.csv`, plots, `benchmark.parquetbundle`)
- No database or hidden state is used by the benchmark pipeline
- Input embeddings remain in `data/<dataset>/` and are not moved

## Python API

```python
from protspace.benchmark import benchmark_methods
from protspace.benchmark.metrics import calculate_trustworthiness
from protspace.data.loaders import load_h5
from protspace.utils.constants import DimensionReductionConfig

# Load embeddings
emb_set = load_h5(["path/to/embeddings.h5"])

# Configure DR
config = DimensionReductionConfig(
    n_components=2,
    random_state=42,
    n_neighbors=15,
    perplexity=30,
)

# Define metrics
metrics = {
    "trustworthiness": calculate_trustworthiness,
}

# Run benchmark
results = benchmark_methods(
    embeddings=emb_set.data,
    methods=["pca", "umap", "tsne"],
    config=config,
    normalize=True,  # Ensure comparable projections
    metric_functions=metrics,
)

# Access results
for method, result in results.items():
    print(f"{method}: {result.time_seconds:.3f}s")
    print(f"  Metrics: {result.metrics}")
```

## Available Methods

- `pca` - Principal Component Analysis
- `tsne` - t-SNE
- `umap` - UMAP
- `pacmap` - PaCMAP
- `mds` - Multidimensional Scaling
- `localmap` - LocalMAP

## Available Metrics

Import from `protspace.benchmark.metrics`:

- `calculate_trustworthiness` - Local neighborhood preservation (k-NN based) ✓ **Implemented**
- `calculate_continuity` - Placeholder (not implemented yet)
- `calculate_silhouette_score` / `make_silhouette_metric` - Silhouette on 2D projection with bundle labels ✓ **Implemented** (see ``run.py``)
- `calculate_knn_preservation` - Placeholder (not implemented yet)

All metrics in `AVAILABLE_METRICS` dictionary. Only use implemented metrics to avoid errors.

## Projection Normalization

When `normalize=True`, projections are made comparable by:
1. Centering at origin
2. Fixing sign ambiguity (positive means for x and y)

This ensures projections from different methods can be directly compared.

## Key Functions

### `benchmark_method(embeddings, method, config=None, normalize=True, metric_functions=None)`

Benchmark a single DR method.

**Returns:** `BenchmarkResult` with:
- `method`: Method name
- `projection`: 2D coordinates (n_samples, 2)
- `time_seconds`: Runtime
- `metrics`: Dict of metric values
- `params`: DR parameters used

### `benchmark_methods(embeddings, methods=None, config=None, normalize=True, metric_functions=None)`

Benchmark multiple methods at once.

**Returns:** `dict[str, BenchmarkResult]`

## Directory Structure

```
src/protspace/benchmark/
├── __init__.py           # Package exports
├── cli.py                # Command-line interface
├── run.py                # Benchmark orchestration
├── visualize.py          # Comparison plots
├── harness.py            # Core benchmarking functions
├── metrics.py            # Quality metrics
├── labels.py             # Label loading utilities
├── io/                   # I/O utilities
│   ├── paths.py          # Path resolution
│   └── headers.py        # Header resolution
└── results/              # Generated benchmark outputs
    └── <dataset>/
        ├── metrics.csv
        ├── normalization_comparison.png
        ├── normalization_comparison.pdf
        └── benchmark.parquetbundle

data/
└── <dataset>/            # Dataset-local outputs
    └── prot_t5.h5        # Input embeddings only
```

## Output Files

- `src/protspace/benchmark/results/<dataset>/metrics.csv` - Runtime and quality metrics per method
- `src/protspace/benchmark/results/<dataset>/normalization_comparison.png/pdf` - Side-by-side visualizations
- `src/protspace/benchmark/results/<dataset>/benchmark.parquetbundle` - Web bundle (persisted result dataset)
