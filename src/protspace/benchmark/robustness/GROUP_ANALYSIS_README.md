# Group-Based Robustness Analysis

This module provides tools for analyzing DR method robustness at the protein group level, rather than globally across all proteins.

## Overview

Traditional robustness analysis computes a single robustness score averaged over all proteins. Group-based analysis computes separate robustness scores for each protein group, revealing group-specific patterns and vulnerabilities.

## Module Structure

```
src/protspace/benchmark/robustness/
├── data_utils.py          # Data loading and ID alignment utilities
├── group_analysis.py      # Core GroupDRRobustnessAnalyzer class
├── plot.py                # Heatmap visualization (extended)
├── config.py              # Shared configuration
└── __init__.py            # Public API exports

scripts/
└── run_group_robustness.py  # CLI wrapper (thin, ~170 lines)
```

## Quick Start

### Command Line

```bash
# Basic usage with seed robustness only
uv run python scripts/run_group_robustness.py \
  --embedding-path data/3FTx/prot_t5.h5 \
  --metadata-path data/3FTx/3FTx.csv \
  --output-dir results/3ftx_group_analysis \
  --dataset-name "3FTx"

# Test all DR methods with seed + hyperparameter experiments
uv run python scripts/run_group_robustness.py \
  --embedding-path data/3FTx/prot_t5.h5 \
  --metadata-path data/3FTx/3FTx.csv \
  --output-dir results/3ftx_full_analysis \
  --experiment-types seed n_neighbors min_dist perplexity learning_rate \
  --dataset-name "3FTx"

# Analyze specific methods and custom grouping column
uv run python scripts/run_group_robustness.py \
  --embedding-path data/toxprot/embeddings.h5 \
  --metadata-path data/toxprot/metadata.csv \
  --methods umap tsne pacmap \
  --group-column pfam_clan \
  --min-group-size 30 \
  --output-dir results/toxprot_analysis \
  --dataset-name "ToxProt"
```

### Python API

```python
from pathlib import Path
from protspace.benchmark.robustness import (
    load_and_prepare_data,
    GroupDRRobustnessAnalyzer,
    plot_group_heatmap,
)

# Load and align data
embeddings, metadata = load_and_prepare_data(
    embedding_path=Path("data/3FTx/prot_t5.h5"),
    metadata_path=Path("data/3FTx/3FTx.csv"),
)

# Initialize analyzer
analyzer = GroupDRRobustnessAnalyzer(
    embeddings=embeddings,
    metadata_df=metadata,
    group_column="major_group",
    min_group_size=15,
    output_dir=Path("results/analysis"),
)

# Run analysis
results = analyzer.run_all_methods(
    methods=["umap", "tsne", "pacmap"],
    experiment_types=["seed", "n_neighbors"],
)

# Save and visualize
analyzer.save_results(results)
analyzer.print_summary(results)

# Create heatmaps
for exp_type in results["experiment_type"].unique():
    plot_group_heatmap(
        results,
        experiment_type=exp_type,
        output_path=Path(f"results/heatmap_{exp_type}.png"),
        dataset_name="3FTx",
    )
```

## Features

### Data Utilities (`data_utils.py`)

- **ID Normalization**: Handles multiple ID formats (SwissProt, UniProt, simple)
- **Smart Alignment**: Automatically matches embeddings and metadata by ID
- **Validation**: Checks for alignment errors and reports statistics

### Group Analysis (`group_analysis.py`)

- **Adaptive k-NN**: Automatically adjusts k for small groups
- **Per-Group Metrics**: Computes robustness separately for each group
- **Multiple Experiments**: Supports seed and all hyperparameter variations
- **Comprehensive Output**: CSV files with detailed and summary results

### Visualization (`plot.py`)

- **Individual Heatmaps**: One per experiment type
- **Combined Heatmaps**: Averages across all hyperparameter experiments
- **Smart Filtering**: Excludes invalid 0.0 values for methods lacking certain hyperparameters
- **Customizable**: Dataset names, color scales, sizes

## Output Files

```
output_dir/
├── group_dr_robustness_results.csv    # Detailed results per method/group/experiment
├── method_summary.csv                 # Aggregated by method and experiment type
├── group_summary.csv                  # Aggregated by group and experiment type
├── heatmap_seed.png                   # Seed robustness heatmap
├── heatmap_n_neighbors.png            # Hyperparameter-specific heatmaps
├── heatmap_min_dist.png
├── ...
└── heatmap_hyperparameters_combined.png  # Average across all hyperparams
```

## Key Differences from Global Analysis

| Aspect | Global Analysis | Group Analysis |
|--------|----------------|----------------|
| **Granularity** | Single score per method | One score per (method, group) pair |
| **Use Case** | Overall method comparison | Identify group-specific vulnerabilities |
| **Filtering** | Uses all proteins | Filters by group size |
| **Output** | 1D comparison (methods) | 2D heatmap (methods × groups) |
| **k-NN Adaptation** | Fixed k | Adaptive k for small groups |

## Advanced Usage

### Custom ID Matching

```python
from protspace.benchmark.robustness import align_embeddings_metadata

embeddings, metadata = align_embeddings_metadata(
    embeddings=raw_embeddings,
    embedding_ids=raw_ids,
    metadata_df=raw_metadata,
    identifier_column="custom_id_column",
)
```

### Notebook Integration

```python
# Load results for further analysis
import pandas as pd
results = pd.read_csv("results/group_dr_robustness_results.csv")

# Filter to specific groups
toxic_groups = results[results["group"].str.contains("toxic")]

# Compare seed vs hyperparameter robustness
import seaborn as sns
sns.violinplot(data=results, x="method", y="robustness", hue="experiment_type")
```

## Implementation Notes

### ID Normalization Logic

The `extract_uniprot_id()` function handles:
- SwissProt format: `sp|O43653|PSCA_HUMAN` → `O43653`
- Custom format: `SP|O43653|Homo_sapiens` → `O43653`
- Simple format: `A0A0U5AUY6` → `A0A0U5AUY6`

### Hyperparameter Filtering

Methods don't support all hyperparameters. The code returns 0.0 for unsupported experiments, which are filtered out when computing combined averages:

```python
# Filter out invalid 0.0 values
valid_hyperparam = results[
    (results["experiment_type"] != "seed") & 
    (results["robustness"] > 0.0)
]
```

### Group Size Adaptation

For small groups, k-NN is automatically reduced:
```python
k_actual = min(knn_k, group_size - 1)
```

## Migration from Old Script

The original `scripts/run_3ftx_group_dr_robustness.py` (577 lines) has been refactored into:

1. **Library modules** (src/protspace/benchmark/robustness/):
   - `data_utils.py` (~130 lines): Data loading
   - `group_analysis.py` (~340 lines): Core analysis logic
   - `plot.py` (extended): Visualization

2. **CLI wrapper** (scripts/):
   - `run_group_robustness.py` (~170 lines): Thin CLI interface

### Benefits

- **Reusability**: Import and use in notebooks/scripts
- **Testability**: Each module can be unit tested
- **Maintainability**: Clear separation of concerns
- **Discoverability**: Public API via `__init__.py`
- **Dataset-agnostic**: No hardcoded paths or dataset names

## See Also

- `GROUP_ANALYSIS_QUICKSTART.md`: Quick reference guide
- `ANALYSIS_RECOMMENDATIONS.md`: Best practices for robustness analysis
- `config.py`: Method configurations and hyperparameter ranges
