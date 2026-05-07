# ProtSpace Benchmark Module

Benchmark dimensionality reduction methods on protein embeddings with timing and quality metrics.

## Quick Start

```bash
# Install dependencies
uv sync --extra benchmark

# Run benchmark with visualization
uv run python -m protspace.benchmark.cli --data 3ftx --plot

# Visualize existing results
uv run python -m protspace.benchmark.cli --data 3ftx --plot-only
```

This will:
- Load embeddings from `data/3ftx/tmp/prot_t5.h5`
- Run all 6 DR methods (PCA, UMAP, t-SNE, PaCMAP, MDS, LocalMAP) twice (normalized & raw)
- Calculate trustworthiness and silhouette scores (if labels available)
- Save metrics to `src/protspace/benchmark/results/3ftx/metrics.csv`
- Generate side-by-side comparison plots: `normalization_comparison.png`

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
└── results/              # Outputs (auto-created)
    └── <dataset>/
        ├── metrics.csv
        ├── normalization_comparison.png
        ├── normalization_comparison.pdf
        └── benchmark.parquetbundle
```

## Output Files

- `metrics.csv` - Runtime and quality metrics per method
- `normalization_comparison.png/pdf` - Side-by-side visualizations
- `benchmark.parquetbundle` - Web visualization bundle (optional)
