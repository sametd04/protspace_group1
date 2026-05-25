# Robustness Benchmarking for DR Methods

Quantifies stability of dimensionality reduction methods using k-NN overlap metrics.

## Quick Start

### One Command Workflow (Recommended)

```bash
# Run experiments AND generate comparison plot
uv run python -m protspace.benchmark.robustness.cli --methods all --plot

# Specific methods with plotting
uv run python -m protspace.benchmark.robustness.cli --methods umap tsne --plot

# Different dataset
uv run python -m protspace.benchmark.robustness.cli --methods all --dataset toxprot --plot
```

### Plot Only (from existing results)

```bash
uv run python -m protspace.benchmark.robustness.cli --skip-experiments --plot
# or
uv run python -m protspace.benchmark.robustness.plot --dataset 3ftx
```

## CLI Options

```
--methods <METHOD> [...]    DR methods: umap, tsne, pacmap, localmap, mds, all (default: all)
--dataset <DATASET>         Dataset: 3ftx (default), toxprot, pla2g2, cath_s40, swissprot_rr
--plot                      Generate comparison plot after experiments
--skip-experiments          Only generate plot from existing results
```

## Output Structure

```
results/{dataset}/robustness/
├── umap/
│   ├── results.csv      # Per-run results
│   ├── summary.csv      # Summary statistics
│   └── metadata.json    # Configuration
├── tsne/
│   └── ...
├── (other methods)
└── comparison_plot.png  # Comparison visualization
```

## Experiments

**Seed Robustness**: Tests stability across 5 different random seeds  
**Hyperparameter Robustness**: Tests sensitivity to key parameters
- **UMAP**: `n_neighbors`, `min_dist`
- **t-SNE**: `perplexity`, `learning_rate`
- **PaCMAP/LocalMAP**: `n_neighbors`, `mn_ratio`, `fp_ratio`
- **MDS**: `n_init`, `max_iter`, `eps`

## Metric

**k-NN Overlap** (k=15): Measures neighborhood preservation between projections
- `1.0` = Perfect stability (all neighbors preserved)
- `0.8-1.0` = High stability
- `0.5-0.8` = Moderate stability
- `<0.5` = Low stability

## Programmatic Usage

```python
from protspace.benchmark.robustness import RobustnessRunner
from pathlib import Path

runner = RobustnessRunner(
    method="umap",
    embeddings=embeddings,  # numpy array
    dataset="3ftx",
    output_base=Path("results/3ftx"),
)
runner.run()
```

## Module Structure

```
robustness/
├── __init__.py  # Package exports
├── config.py    # Method configurations (baselines, experiments)
├── run.py       # Core benchmarking (RobustnessRunner class)
├── cli.py       # Command-line interface
└── plot.py      # Visualization tools
```

## Configuration

Edit `config.py` to:
- Add new DR methods
- Modify baseline parameters
- Change experiment parameters (seeds, hyperparameter values)

## Performance

Approximate runtime on 3ftx dataset (836 proteins):
- All 5 methods: ~15-20 minutes
- Individual methods: 1-5 minutes each
