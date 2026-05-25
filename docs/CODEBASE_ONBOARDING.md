# ProtSpace Codebase Onboarding Guide

## 1. High-Level Overview

### Purpose
ProtSpace is a bioinformatics visualization tool that helps researchers explore protein embeddings and similarity relationships. It takes protein sequences or pre-computed embeddings and creates interactive 2D/3D visualizations that can be colored by biological annotations (UniProt, InterPro, Taxonomy, etc.).

### Inputs
- **Protein sequences**: FASTA files or UniProt queries
- **Pre-computed embeddings**: HDF5 files containing high-dimensional protein language model (pLM) embeddings
- **Similarity matrices**: For protein similarity-based projections
- **Optional**: PDB/CIF structure files for 3D visualization

### Outputs
- **Parquet bundles** (`.parquetbundle`): Compressed archive containing:
  - Projection data (2D/3D coordinates)
  - Protein annotations
  - Visualization settings (colors, shapes)
- **Interactive Dash web app**: Local visualization server
- **Static plots**: PNG, PDF, SVG, HTML exports

### Key Value Proposition
Allows researchers to visualize thousands of proteins in 2D/3D space, identify clusters, explore relationships, and export publication-ready figures.

---

## 2. Architecture & Components

### 2.1 Core Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Layer (Typer)                    │
│  prepare, embed, project, annotate, bundle, serve       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│              Processing Pipeline Layer                  │
│  ReductionPipeline, BaseProcessor, AnnotationManager    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│                  Data Layer                              │
│  Loaders (FASTA, H5, Query), Embedders, Parsers         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│              Visualization Layer                         │
│  Dash App, Plotly, ArrowReader, UI Components           │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Key Components

#### **CLI Layer** (`src/protspace/cli/`)
- **Purpose**: User-facing commands
- **Files**:
  - `app.py`: Main Typer app with logging setup
  - `prepare.py`: Unified pipeline (most users start here)
  - `embed.py`: Generate embeddings from FASTA
  - `project.py`: Dimensionality reduction
  - `annotate.py`: Fetch biological annotations
  - `bundle.py`: Package outputs into `.parquetbundle`
  - `serve.py`: Launch web app
  - `style.py`: Customize colors/shapes

#### **Data Processing** (`src/protspace/data/`)
- **Loaders** (`loaders/`): Load data from different sources
  - `fasta.py`: Parse FASTA files
  - `h5.py`: Load HDF5 embeddings
  - `query.py`: Fetch sequences from UniProt
  - `similarity.py`: Compute protein similarity matrices
  - `embedding_set.py`: Unified container for embeddings
  
- **Embedding** (`embedding/`):
  - `biocentral.py`: Generate embeddings via Biocentral API (12 pLMs supported)
  
- **Annotations** (`annotations/`):
  - `manager.py`: Coordinate annotation retrieval from multiple sources
  - `retrievers/`: Fetch data from UniProt, InterPro, Taxonomy, TED, Biocentral
  - `transformers/`: Clean and format annotation data
  
- **Processors** (`processors/`):
  - `pipeline.py`: Main `ReductionPipeline` orchestrates the full workflow
  - `base_processor.py`: Handles dimensionality reduction (PCA, UMAP, t-SNE, etc.)
  
- **IO** (`io/`):
  - Bundle creation/extraction
  - Settings management
  - Parquet serialization

#### **Visualization** (`src/protspace/visualization/`)
- `plotting.py`: Create Plotly figures (2D/3D scatter plots)
- `molstar.py`: Protein structure viewer integration

#### **UI** (`src/protspace/ui/`)
- `layout.py`: Dash app layout structure
- `callbacks.py`: Interactive behavior (dropdown changes, selections, etc.)
- `styles.py`: CSS styling

#### **Utilities** (`src/protspace/utils/`)
- `arrow_reader.py`: **Critical**: Main data access layer, reads Parquet files and provides unified API
- Reducer implementations (UMAP, PCA, t-SNE, MDS, PaCMAP, LocalMAP)

---

## 3. Entry Points & Execution Flow

### 3.1 Main Entry Points

**CLI (production)**: `src/protspace/cli/app.py`
- Command: `protspace <subcommand>`
- Defined in `pyproject.toml`: `protspace = "protspace.cli.app:app"`

**Web App (legacy/direct)**: `src/protspace/main.py`
- Direct Python execution: `python -m protspace.main data.parquetbundle`
- Creates `ProtSpace` app instance and launches Dash server

### 3.2 Typical Execution Flows

#### **Flow 1: Quick Start (Most Common)**
```
protspace prepare -i sequences.fasta -e prot_t5 -m pca2,umap2 -o output/
```

1. `cli/prepare.py` → Entry point
2. Parse arguments, validate inputs
3. If FASTA input → `biocentral.py::embed_sequences()` (generates embeddings)
4. `pipeline.py::ReductionPipeline` → Orchestrate:
   - Load embeddings via `loaders/h5.py` or `loaders/fasta.py`
   - Fetch annotations via `annotations/manager.py`
   - Apply dimensionality reduction via `base_processor.py`
   - Bundle outputs via `io/bundle.py`
5. Output: `output/data.parquetbundle`

#### **Flow 2: Power User (Step-by-Step)**
```
protspace embed -i seq.fasta -e prot_t5 -o embeddings/
protspace project -i embeddings/prot_t5.h5 -m pca2,umap2 -o projections/
protspace annotate -i embeddings/prot_t5.h5 -a default -o annotations.parquet
protspace bundle -p projections/ -a annotations.parquet -o output.parquetbundle
```

Each command is independent and caches intermediate results.

#### **Flow 3: Web Visualization**
```
protspace serve output.parquetbundle --port 8050
```

1. `cli/serve.py` → Launches `main.py`
2. `main.py::detect_data_type()` → Extract bundle to temp dir
3. `app.py::ProtSpace` → Create Dash app
4. `ui/layout.py` → Build UI structure
5. `ui/callbacks.py` → Register interactive behaviors
6. `visualization/plotting.py` → Generate plots via `ArrowReader`

---

## 4. Most Important Files (Read in Order)

### Phase 1: Understanding the Data Model (30 mins)
1. **`src/protspace/utils/arrow_reader.py`** (300 lines)
   - **Why first**: Central data access layer. Everything flows through this.
   - **Key concepts**: 
     - Data structure: `protein_data`, `projections`, `visualization_state`
     - Parquet file schema (3 files + 1 JSON)
     - API methods: `get_projection_data()`, `get_protein_annotations()`, etc.

2. **`src/protspace/data/loaders/embedding_set.py`** (200 lines)
   - **Why**: Understand how embeddings are represented internally
   - **Key**: `EmbeddingSet` class, `format_projection_name()` naming conventions

3. **`src/protspace/core/constants.py`** & **`src/protspace/core/config.py`** (50 lines total)
   - **Why**: Configuration constants, marker shapes, colors
   - Quick reference for settings

### Phase 2: Understanding the Pipeline (45 mins)
4. **`src/protspace/data/processors/pipeline.py`** (600 lines)
   - **Why**: Heart of the processing logic
   - **Key**: 
     - `ReductionPipeline` class orchestrates everything
     - `ReducerParams`: DR algorithm parameters
     - `MethodSpec`: Method specification parsing (e.g., `umap2:n_neighbors=50`)
     - Caching logic, refetch stages

5. **`src/protspace/data/processors/base_processor.py`** (250 lines)
   - **Why**: Handles actual dimensionality reduction
   - **Key**: `process_reduction()` method, data transformations

6. **`src/protspace/cli/prepare.py`** (500 lines)
   - **Why**: Most-used CLI command, shows how everything connects
   - **Key**: Input parsing, pipeline configuration, caching strategies

### Phase 3: Annotations & Data Sources (30 mins)
7. **`src/protspace/data/annotations/manager.py`** (400 lines)
   - **Why**: Coordinates fetching from 5 different biological databases
   - **Key**: `AnnotationManager::retrieve_annotations()`, caching, merging

8. **`src/protspace/data/annotations/retrievers/uniprot.py`** (example retriever)
   - **Why**: Understand the retriever pattern
   - **Pattern repeats** for InterPro, Taxonomy, TED, Biocentral

### Phase 4: Visualization (30 mins)
9. **`src/protspace/visualization/plotting.py`** (350 lines)
   - **Why**: Core plotting logic
   - **Key**: `create_plot()`, 2D vs 3D handling, color/shape mapping

10. **`src/protspace/ui/callbacks.py`** (500 lines)
    - **Why**: Interactive behavior
    - **Key**: Dash callback decorators, selection handling, updates

11. **`src/protspace/app.py`** (130 lines)
    - **Why**: Main app class
    - **Key**: `ProtSpace` initialization, PDB loading, server launch

---

## 5. Core Algorithms

### 5.1 Dimensionality Reduction Pipeline

**Problem**: Protein embeddings are high-dimensional (1024-5120 dims). Need to visualize in 2D/3D.

**Solution**: Apply DR algorithms (PCA, UMAP, t-SNE, MDS, PaCMAP, LocalMAP)

**Implementation** (`base_processor.py::process_reduction()`):
```python
# Simplified flow:
1. Validate input data shape
2. Select reducer based on method name
3. Configure reducer with user params (n_neighbors, min_dist, etc.)
4. Fit and transform data
5. Return reduction dict with coordinates + metadata
```

**Key insight**: The system supports "precomputed" distance matrices for similarity-based projections (only MDS works with these).

### 5.2 Annotation Merging

**Problem**: Proteins can have annotations from 5 different sources. Need to merge into single dataframe.

**Solution** (`annotations/merging.py`):
```python
# Simplified:
1. Start with protein IDs as index
2. For each annotation source (UniProt, InterPro, etc.):
   - Fetch data (API or cache)
   - Transform to standard format
   - Left join on protein ID
3. Handle missing values → "<N/A>"
4. Apply custom transformations (e.g., taxonomy hierarchy)
```

### 5.3 Embedding Generation

**Problem**: Convert protein sequences to numerical vectors.

**Solution** (`data/embedding/biocentral.py`):
```python
# Simplified:
1. Batch sequences (default 1000)
2. Send to Biocentral API with model name (e.g., "prot_t5")
3. Parse response (JSON with embeddings)
4. Save to HDF5: {protein_id: embedding_vector}
5. Add metadata (model_name, sequence_length, etc.)
```

**Supported models**: 12 pLMs (ProtT5, ESM2 variants, Ankh, ESMC)

### 5.4 Projection Naming & Disambiguation

**Problem**: Multiple projections with same method/dims but different parameters need unique names.

**Solution** (`data/loaders/embedding_set.py::format_projection_name()`):
```python
# Examples:
"ProtT5 — PCA 2"
"ProtT5 — UMAP 2 (n=50, d=0.1)"  # when params differ
"Similarity — MDS 2"
```

**Disambiguation logic** (`pipeline.py::disambiguation_suffix()`):
- If multiple specs with same (method, dims): add param suffix only to override specs
- Plain spec keeps clean name

---

## 6. Non-Obvious Design Decisions

### 6.1 Parquet Over JSON
**Why**: v4.0+ migrated from JSON to Parquet for performance
- **Before**: Single large JSON file (>100MB for large datasets)
- **After**: 3 Parquet files + 1 small JSON
  - `selected_annotations.parquet`: Protein metadata (wide format)
  - `projections_metadata.parquet`: Projection configs
  - `projections_data.parquet`: 2D/3D coordinates
  - `visualization_state.json`: Colors, shapes (small, rarely changes)
- **Benefit**: Faster loading, lower memory, column-oriented access

### 6.2 Lazy Command Registration
**Why** (`cli/app.py::_register_commands()`):
```python
# Deferred imports to keep startup fast
def _register_commands():
    from protspace.cli import (
        annotate, bundle, embed, prepare, project, serve, style
    )
```
- Import subcommands only when needed
- Faster `protspace --help` response

### 6.3 Caching Strategy (Resumability)
**Decision**: Cache all intermediate results in `{output}/tmp/` by default
- **Files cached**:
  - `sequences.fasta`: UniProt query results
  - `*.h5`: Embeddings
  - `similarity_matrix.npy`: MMseqs2 results
  - `{embedding}_{method}{dims}_{hash}.npz`: DR projections
  - Annotation parquets per source
  
**Why**: Expensive operations (embedding, annotation fetching) shouldn't re-run on every tweak
  
**Trade-off**: Disk space vs time. Users can disable with `--no-keep-tmp` or invalidate with `--refetch <stage>`

### 6.4 Union vs Intersection for Multi-Input
**Problem**: What to do when `-i species_a.h5:prot_t5 -i species_b.h5:prot_t5`?
- **Same embedding name** → UNION (concatenate proteins) - v4.3.1 fix
- **Different names** → INTERSECTION (only common proteins)

**Why**: Use case dependent. Same model = comparing datasets. Different models = multi-view comparison.

### 6.5 Config Override Pattern
**Pattern** (`pipeline.py::_run_with_overridden_config()`):
```python
# Save original config
saved = base.config
base.config = effective_params  # Temp override
try:
    result = base.process_reduction(...)
finally:
    base.config = saved  # Always restore
```

**Why**: Parameter overrides (e.g., `umap2:n_neighbors=50`) shouldn't leak between reductions. Ensures isolation.

### 6.6 ArrowReader Dual Mode
**Pattern** (`utils/arrow_reader.py::__init__()`):
```python
if isinstance(source, dict):
    self.data = source  # Pre-built dict (legacy compatibility)
else:
    self._load_data()  # Load from Parquet directory
```

**Why**: Backward compatibility with legacy JSON-based system + support for in-memory data during tests.

---

## 7. Configuration & Dependencies

### 7.1 External Dependencies (Critical)
- **Biocentral API**: Embeddings generation (requires network)
- **UniProt REST API**: Annotation fetching (can be cached)
- **InterPro REST API**: Annotation fetching
- **PyMMseqs**: Protein similarity computation (local, fast)

### 7.2 Configuration Files
- **`pyproject.toml`**: Project metadata, dependencies, tool configs (pytest, ruff)
- **`uv.lock`**: Locked dependencies (use `uv` package manager)
- **`.parquetbundle`**: User-facing data format (ZIP-like archive)

### 7.3 Entry Point Registration
```toml
[project.scripts]
protspace = "protspace.cli.app:app"
```
After `pip install protspace`, `protspace` command is available globally.

---

## 8. Anti-Patterns & Pitfalls

### 8.1 Direct Config Mutation (AVOID)
**❌ Bad**:
```python
base.config["precomputed"] = True
base.process_reduction(...)
# Forgot to restore!
```

**✅ Good**:
```python
_run_with_overridden_config(base, {..., "precomputed": True}, ...)
```

### 8.2 Hardcoded Paths
**❌ Bad**: `"/tmp/cache"` (breaks on Windows)
**✅ Good**: `Path.home() / ".cache" / "protspace"`

### 8.3 Missing Annotation Validation
**Context**: User can pass arbitrary annotation names
**Risk**: Typos silently ignored
**Mitigation**: `annotations/configuration.py` validates against known groups

### 8.4 Large JSON in Memory
**Old pattern** (pre-v4.0): Load entire dataset as dict
**New pattern**: Use `ArrowReader` to lazy-load from Parquet

---

## 9. Step-by-Step Exploration Plan (<2 hours)

### Hour 1: Data Flow Understanding
**Goal**: Trace a protein from FASTA → visualization

1. **Run the quick start** (10 mins):
   ```bash
   # Use a small test dataset
   echo -e ">P12345\nMKLLILVLCFATCVLA" > test.fasta
   protspace prepare -i test.fasta -e prot_t5 -m pca2 -o test_output/
   ```
   - Observe console output
   - Check `test_output/tmp/` for cached files
   - Inspect `test_output/data.parquetbundle` (it's a ZIP!)

2. **Extract and inspect the bundle** (10 mins):
   ```bash
   unzip test_output/data.parquetbundle -d bundle_contents/
   ls -lh bundle_contents/
   # Use pandas or duckdb to peek at parquet files
   python -c "import pandas as pd; print(pd.read_parquet('bundle_contents/selected_annotations.parquet'))"
   ```

3. **Read the ArrowReader** (15 mins):
   - Open `src/protspace/utils/arrow_reader.py`
   - Trace `_load_data()` → `_build_data_structure()`
   - Understand the 3-parquet schema

4. **Trace one CLI command end-to-end** (25 mins):
   - Pick `protspace prepare` 
   - Start at `cli/prepare.py::prepare()`
   - Follow the call chain:
     ```
     prepare() → ReductionPipeline.run()
               → pipeline._run_reductions()
               → base_processor.process_reduction()
               → UMAP/PCA/etc.
     ```
   - Set breakpoints or add print statements to see data shapes

### Hour 2: Hands-On Experiments
**Goal**: Modify behavior, see results

5. **Experiment 1: Add a custom annotation** (20 mins):
   - Create a CSV: `custom_annot.csv` with columns `protein_id,custom_label`
   - Run: `protspace prepare -i test.fasta -a custom_annot.csv -m pca2 -o exp1/`
   - Verify annotation appears in bundle

6. **Experiment 2: Modify DR parameters** (15 mins):
   - Run: `protspace prepare -i test.h5 -m "umap2:n_neighbors=5" -m "umap2:n_neighbors=50" -o exp2/`
   - Launch: `protspace serve exp2/data.parquetbundle`
   - Compare projections in browser

7. **Experiment 3: Add a print to understand data flow** (15 mins):
   - Edit `src/protspace/data/processors/base_processor.py::process_reduction()`
   - Add: `print(f"Reducing {data.shape} with {method}")` at line ~50
   - Re-install: `pip install -e .`
   - Re-run a command, observe output

8. **Experiment 4: Write a minimal test** (10 mins):
   - Create `tests/test_my_understanding.py`:
     ```python
     from protspace.utils.arrow_reader import ArrowReader
     from pathlib import Path
     
     def test_arrow_reader_loads_bundle():
         reader = ArrowReader(Path("test_output/bundle_contents"))
         assert len(reader.get_projection_names()) > 0
     ```
   - Run: `pytest tests/test_my_understanding.py -v`

---

## 10. Small Experiments to Verify Understanding

### Experiment A: Manual Projection Creation
**Challenge**: Create a projection without using CLI
```python
from protspace.data.loaders import load_h5
from protspace.data.processors.base_processor import BaseProcessor
from protspace.utils import get_reducers

# Load embeddings
emb_set = load_h5([Path("embeddings/prot_t5.h5")])

# Configure processor
config = {"metric": "euclidean", "random_state": 42, "n_neighbors": 15}
processor = BaseProcessor(config, get_reducers())

# Run reduction
reduction = processor.process_reduction(emb_set.data, "umap", 2)
print(reduction["coordinates"])  # Should be Nx2 array
```

**Verify**: Output shape matches input protein count.

### Experiment B: Simulate Annotation Retrieval
**Challenge**: Fetch UniProt annotations manually
```python
from protspace.data.annotations.retrievers.uniprot import UniProtRetriever
import pandas as pd

retriever = UniProtRetriever()
protein_ids = ["P12345", "Q9Y6K9"]
annots = retriever.retrieve(protein_ids)
df = pd.DataFrame(annots)
print(df.columns)  # Should include gene_name, protein_name, etc.
```

**Verify**: Check against UniProt website for one protein.

### Experiment C: Custom Color Scheme
**Challenge**: Modify visualization colors programmatically
```python
from protspace.utils.arrow_reader import ArrowReader
from pathlib import Path

reader = ArrowReader(Path("test_output/bundle_contents"))
# Change color for a specific annotation value
reader.update_annotation_color("organism", "Homo sapiens", "#FF0000")
reader.save_data(Path("test_output/bundle_contents_modified"))
# Re-bundle and serve to see red color
```

**Verify**: Color persists after reload.

---

## 11. Confusing Parts for New Developers

### 11.1 Method Spec Parsing
**Confusing**: `"umap2:n_neighbors=50;min_dist=0.1"`
- **Parse**: `method="umap"`, `dims=2`, `overrides={"n_neighbors": 50, "min_dist": 0.1}`
- **Why confusing**: Mix of positional (method+dims) and key-value (params)
- **Read**: `pipeline.py::parse_method_spec()`

### 11.2 EmbeddingSet vs HDF5 Structure
**Confusing**: What's the difference?
- **HDF5 file**: On-disk format (protein_id → embedding vector)
- **EmbeddingSet**: In-memory Python object with `.data` (NxD array), `.headers` (protein IDs), `.name`
- **Loader**: `load_h5()` bridges the gap

### 11.3 Precomputed Flag
**Confusing**: When is `precomputed=True` used?
- **Answer**: When input is a distance/similarity matrix, not embeddings
- **Use case**: `--similarity` flag or manual similarity input
- **Why**: MDS can work directly on distances without needing raw embeddings

### 11.4 Projection Name Disambiguation
**Confusing**: Why "ProtT5 — UMAP 2" vs "ProtT5 — UMAP 2 (n=50)"?
- **Answer**: Only add param suffix when multiple specs exist for same (method, dims)
- **Edge case**: `-m umap2 -m umap2:n_neighbors=50` → first keeps clean name, second gets suffix
- **Read**: `pipeline.py::disambiguation_suffix()`

### 11.5 Annotation Groups vs Individual Names
**Confusing**: `-a default` vs `-a gene_name`?
- **Groups**: Predefined sets (default, all, uniprot, interpro, taxonomy, ted, biocentral)
- **Individual**: Specific annotation column names
- **Mix allowed**: `-a default,custom_column`
- **Read**: `annotations/configuration.py::resolve_annotation_spec()`

### 11.6 Cache Invalidation
**Confusing**: When does `--refetch` apply?
- **Stages**: query, embed, similarity, projections, uniprot, taxonomy, interpro, ted, biocentral
- **Shorthands**: all, annotations
- **Example**: `--refetch uniprot,interpro` → re-fetch only those annotations, use cached embeddings
- **Read**: `cli/prepare.py::_parse_refetch()`

### 11.7 Bundle vs Directory Output
**Confusing**: When bundled vs not?
- **Default**: `--bundled` (single `.parquetbundle` file)
- **Alternative**: `--no-bundled` (directory of parquet files)
- **Why both**: Bundle for distribution, directory for debugging/inspection
- **Extract**: Bundle is just a ZIP archive

---

## 12. Testing Strategy

### Test Organization
- **Unit tests**: `tests/test_*.py` (25 files)
- **Markers**:
  - `@pytest.mark.slow`: Long-running tests (skip with `-m "not slow"`)
  - `@pytest.mark.integration`: Requires external services

### Key Test Files
- `test_pipeline_utils.py`: Pipeline orchestration
- `test_annotation_manager.py`: Annotation fetching and merging
- `test_reducers.py`: DR algorithm outputs
- `test_interpro_annotation_retriever.py`: API interaction patterns

### Running Tests
```bash
# All tests
pytest

# Fast tests only
pytest -m "not slow"

# Specific file
pytest tests/test_arrow_reader.py -v

# With coverage
pytest --cov=src/protspace --cov-report=html
```

---

## 13. Quick Reference: Key Patterns

### Pattern 1: Loading Embeddings
```python
from protspace.data.loaders import load_h5
emb_set = load_h5([Path("embed.h5")], name_override="MyModel")
# emb_set.data: NxD numpy array
# emb_set.headers: List[str] protein IDs
# emb_set.name: "MyModel"
```

### Pattern 2: Running Dimensionality Reduction
```python
from protspace.data.processors.base_processor import BaseProcessor
from protspace.utils import get_reducers

config = {"metric": "euclidean", "random_state": 42}
processor = BaseProcessor(config, get_reducers())
reduction = processor.process_reduction(data, method="umap", dims=2)
# reduction: dict with "coordinates", "method", "dimensions", "info"
```

### Pattern 3: Reading Projection Data
```python
from protspace.utils.arrow_reader import ArrowReader
reader = ArrowReader(Path("bundle_dir/"))
proj_names = reader.get_projection_names()
coords = reader.get_projection_data(proj_names[0])
# coords: List[dict] with "identifier", "coordinates" {"x": ..., "y": ...}
```

### Pattern 4: Fetching Annotations
```python
from protspace.data.annotations.manager import AnnotationManager
manager = AnnotationManager(annotation_spec=["default"], ...)
df = manager.retrieve_annotations(protein_ids)
# df: pandas DataFrame with annotation columns
```

---

## 14. Development Workflow

### Setup
```bash
git clone <repo>
cd protspace_group1
pip install -e .  # Editable install
pip install -e ".[dev]"  # With dev dependencies
```

### Make Changes
1. Edit source files in `src/protspace/`
2. Run tests: `pytest tests/`
3. Check linting: `ruff check src/`
4. Format: `ruff format src/`

### Testing Changes
```bash
# CLI testing
protspace prepare -i test.fasta -m pca2 -o test_output/

# Python API testing
python -c "from protspace.app import ProtSpace; ..."

# Unit tests
pytest tests/test_specific.py -v
```

### Debugging
- Add `import pdb; pdb.set_trace()` for breakpoints
- Use `--verbose 2` flag for DEBUG logging
- Inspect `{output}/tmp/` for intermediate files

---

## 15. Common Tasks Cheatsheet

### Generate Embeddings
```bash
protspace embed -i seqs.fasta -e prot_t5 -e esm2_650m -o embeddings/
```

### Project with Custom Parameters
```bash
protspace project -i embed.h5 -m "umap2:n_neighbors=15" -m "umap2:n_neighbors=50" -o proj/
```

### Fetch Annotations Only
```bash
protspace annotate -i embed.h5 -a uniprot,interpro -o annots.parquet
```

### Bundle Manually
```bash
protspace bundle -p projections/ -a annots.parquet -o final.parquetbundle
```

### Serve Web App
```bash
protspace serve data.parquetbundle --port 8050
```

### Invalidate Cache
```bash
protspace prepare -i data.h5 -m pca2 --refetch projections -o output/
```

---

## 16. Where to Find Help

### Documentation
- `README.md`: Quick start
- `docs/cli.md`: CLI reference
- `docs/annotations.md`: Annotation catalog
- `docs/styling.md`: Customization guide
- `CHANGELOG.md`: Recent changes

### Code Comments
- Most modules have docstrings
- Complex algorithms have inline comments
- Check `cli/prepare.py` for well-documented CLI patterns

### Tests
- Look at `tests/test_*.py` for usage examples
- Integration tests show full workflows

---

## Summary: Your First 2 Hours

**0:00-0:15**: Run quick start, inspect outputs  
**0:15-0:30**: Read `arrow_reader.py` - understand data model  
**0:30-1:00**: Trace `prepare` command - understand pipeline  
**1:00-1:20**: Experiment 1 - Add custom annotation  
**1:20-1:35**: Experiment 2 - Modify DR parameters  
**1:35-1:50**: Experiment 3 - Add debug prints  
**1:50-2:00**: Write a test - Verify understanding  

**After 2 hours**: You should be able to:
- Explain the data flow from FASTA → bundle
- Understand the 3-parquet schema
- Modify DR parameters and see results
- Add custom annotations
- Navigate the codebase confidently

---

**Good luck onboarding! 🚀**
