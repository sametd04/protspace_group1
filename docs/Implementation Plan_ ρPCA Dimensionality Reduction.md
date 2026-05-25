# Implementation Plan: ρPCA (Rho PCA) Dimensionality Reduction
## Overview
ρPCA is a dimensionality reduction method that finds directions maximizing the ratio of target variance to background variance using a generalized eigenproblem: Σ_T v = λ Σ_B v. This requires both a target dataset and a background dataset, which differs from existing DR methods that only use a single dataset.
## Key Design Challenge
**Problem**: All existing DR methods accept a single data matrix, but ρPCA requires TWO datasets (target and background).
**Solution Options**:
1. **Background selection policy**: Automatically select background samples from the target data (e.g., random subset, outliers, uniform sampling)
2. **Extended interface**: Add optional background data parameter to the reducer interface
3. **Preprocessing step**: Allow users to specify background data via CLI/config before reduction
For initial implementation, recommend **Option 1** with configurable policies, as it fits the existing architecture.
## Files to Modify
### 1. src/protspace/utils/constants.py
**Changes**:
* Add `PPCA_NAME = "ppca"` constant (around line 15)
* Add `"ppca"` to `REDUCER_METHODS` list (line 17)
* Add ρPCA-specific parameters to `DimensionReductionConfig` dataclass:
    * `regularization_mu: float` - Tikhonov regularization for singular background covariance (default: 0.0)
    * `background_ratio: float` - Fraction of data to use as background (default: 0.3)
    * `background_strategy: Literal["random", "uniform", "outlier"]` - How to select background samples (default: "random")
* Add validation for new parameters in `__post_init__` method
### 2. src/protspace/utils/reducers.py
**Changes**:
* Import `PPCA_NAME` from constants (line 17)
* Create new `PPCAReducer` class (after `MDSReducer`, around line 380):
    * Inherit from `DimensionReducer`
    * Implement `fit_transform(data)` method:
        * Select background samples based on `background_strategy`
        * Compute sample covariance matrices for target and background
        * Apply Tikhonov regularization if needed
        * Solve generalized eigenproblem using `scipy.linalg.eigh`
        * Sort eigenvectors by descending eigenvalues
        * Project data onto top `n_components` eigenvectors
        * Return projected coordinates
    * Implement `get_params()` method:
        * Return dict with `n_components`, `regularization_mu`, `background_ratio`, `background_strategy`, `random_state`, and eigenvalue ratios
    * Add private helper methods:
        * `_select_background_samples(data)` - Select background indices
        * `_compute_covariance(data)` - Compute centered sample covariance
        * `_solve_generalized_eigenproblem(Sigma_T, Sigma_B)` - Core ρPCA math
* Update `parameters_by_method` function to include ρPCA in `method_map` (around line 108)
### 3. src/protspace/utils/**init**.py
**Changes**:
* Import `PPCA_NAME` from reducers (line 16)
* Import `PPCAReducer` from reducers (line 27)
* Add to `_REDUCERS` dict: `PPCA_NAME: PPCAReducer` (around line 36)
* Add `PPCA_NAME` to `_reducer_attrs` set (line 58)
* Add to `__dir__()` return list (line 84)
### 4. src/protspace/data/processors/base_processor.py
**Changes**:
* Add ρPCA-specific config parameters to `valid_config_keys` in `process_reduction` method (around line 33):
    * `"regularization_mu"`
    * `"background_ratio"`
    * `"background_strategy"`
* No special handling needed (unlike MDS) unless precomputed similarity matrices should be supported
### 5. tests/test_reducers.py
**Changes**:
* Import `PPCAReducer` from protspace.utils.reducers (line 17)
* Add `TestPPCAReducer` class (after `TestLocalMAPReducer`, around line 170):
    * `test_output_shape_2d` - Verify (N, 2) output
    * `test_output_shape_3d` - Verify (N, 3) output
    * `test_no_nan_values` - Check for NaN
    * `test_deterministic` - Same seed → same output
    * `test_get_params` - Verify params dict structure
    * `test_background_strategies` - Test random/uniform/outlier selection
    * `test_regularization` - Test singular background covariance handling
    * `test_eigenvalue_ratios` - Verify ratios are positive and descending
* Add `("ppca", PPCAReducer)` to `ALL_REDUCERS` list (line 177)
* The parametrized tests will automatically include ρPCA
### 6. src/protspace/cli/project.py
**Changes**:
* Add CLI options for ρPCA parameters (around line 60):
    * `--regularization-mu` (default: 0.0)
    * `--background-ratio` (default: 0.3)
    * `--background-strategy` (default: "random")
* Add to `ReducerParams` initialization (line 108)
* No other changes needed (method spec parsing already supports parameter overrides like `ppca2:background_ratio=0.5`)
### 7. src/protspace/data/processors/pipeline.py
**Changes**:
* Add ρPCA parameters to `ReducerParams` dataclass (around line 60):
    * `regularization_mu: float`
    * `background_ratio: float`
    * `background_strategy: str`
* Add to `_VALID_OVERRIDE_KEYS` set in `_coerce_value` function (around line 85)
* Add type coercion cases in `_coerce_value` for new parameters
## New Files to Create
None required. All implementation fits within existing files.
## Implementation Notes
### Mathematical Implementation
**Generalized Eigenproblem Solver**:
```python
import scipy.linalg as la
eigenvalues, eigenvectors = la.eigh(Sigma_T, Sigma_B)
# Sort by descending eigenvalue
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
```
**Tikhonov Regularization**:
```python
if regularization_mu > 0:
    Sigma_B_reg = Sigma_B + regularization_mu * np.eye(Sigma_B.shape[0])
else:
    Sigma_B_reg = Sigma_B
```
**Covariance Computation**:
```python
# Center data
data_centered = data - data.mean(axis=0)
# Sample covariance
Sigma = (data_centered.T @ data_centered) / (data_centered.shape[0] - 1)
```
### Background Selection Strategies
**Random** (default): Random subset of samples
**Uniform**: Samples uniformly across embedding space using k-means clustering
**Outlier**: Samples from low-density regions (inverse of local density)
### Edge Cases
* **Singular Sigma_B**: Apply regularization automatically if `la.eigh` fails
* **Small datasets**: Ensure background_ratio doesn't leave too few target samples
* **Negative eigenvalues**: Clip to zero and warn (shouldn't happen with proper covariance matrices)
## Testing Strategy
1. **Unit tests**: Individual reducer tests (shape, no NaN, deterministic)
2. **Integration tests**: Through `BaseProcessor.process_reduction`
3. **Sanity checks**:
    * Higher target variance → higher eigenvalues
    * Background orthogonality: V^T Sigma_B V ≈ I
    * Scale invariance: Scaling both covariances only scales eigenvalues
4. **Edge case tests**:
    * Singular background covariance
    * All background samples identical
    * Very small background sets
## CLI Usage Examples
After implementation, users will be able to use:
```warp-runnable-command
# Basic usage
protspace project -i embeddings.h5 -m ppca2
# With custom parameters
protspace project -i embeddings.h5 -m ppca2:background_ratio=0.4,regularization_mu=0.01
# Multiple methods including ρPCA
protspace prepare -i seq.fasta -e prot_t5 -m pca2,umap2,ppca2 -o output/
```
## Open Questions
1. **Background selection policy**: Should users have more control over background selection (e.g., specify background protein IDs)?
2. **Multiple replicates**: Should we support averaging multiple background/target covariances as mentioned in the theory?
3. **Visualization**: Should ρPCA projections have special visual indicators in the UI to show which samples were used as background?
4. **Documentation**: Where should we explain the biological interpretation of ρPCA results?
## Estimated Effort
* **Core implementation** (constants, reducer class, registration): 2-3 hours
* **Testing** (unit tests, edge cases): 1-2 hours
* **CLI integration** (parameters, validation): 1 hour
* **Documentation** (docstrings, examples): 1 hour
* **Total**: 5-7 hours
## Success Criteria
- [ ] All existing tests pass
- [ ] New ρPCA tests pass with >95% coverage
- [ ] Can run `protspace project -m ppca2` on existing datasets
- [ ] Output projections are finite, reproducible, and shape-correct
- [ ] Eigenvalue ratios are positive and descending
- [ ] Integration with existing pipeline requires no changes to other reducers
