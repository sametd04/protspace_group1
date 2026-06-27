# ρPCA / annotation-defined nuisance backgrounds in ProtSpace

This branch integrates **ρPCA** as a normal ProtSpace reduction method and adds support for building an explicit ρPCA background from user-declared annotation nuisances.

The final implementation intentionally supports only two ρPCA input modes:

1. `--nuisance`: build a background from annotations inside `protspace prepare`
2. `--ppca-background`: use an already existing explicit background HDF5

The old experimental modes are **not** part of this final integration:

```text
--ppca-mode annotation
--ppca-mode derived
--ppca-mode paired_delta
```

---

## Where ρPCA is integrated

ρPCA is integrated in the following parts of ProtSpace:

```text
protspace/utils/constants.py
```

Adds `PPCA_NAME = "ppca"` and registers `ppca` as a valid reducer method. It also defines the ρPCA defaults used by the reducer config:

```text
regularization_mu = 1e-6
standard_scale = False
```

```text
protspace/utils/reducers.py
```

Contains the mathematical ρPCA implementation:

```text
PPCAReducer
_sample_covariance
_standard_scale_columns
_solve_rho_eigenproblem
_project_full_input
```

The reducer receives the target embedding matrix `X` and an explicit background matrix `B`, then solves:

```text
Σ_T v = λ(Σ_B + μI)v
```

```text
protspace/utils/__init__.py
```

Registers `PPCAReducer` in `get_reducers()`, so `ppca2` and `ppca3` can be used like other reduction methods.

```text
protspace/utils/annotation_nuisance_background.py
```

Builds annotation-defined nuisance backgrounds. It parses `--nuisance` specifications, encodes annotation columns, fits annotation-to-embedding ridge models, predicts nuisance shifts, and creates signed background vectors.

```text
protspace/cli/common_options.py
```

Defines the final user-facing ρPCA CLI options:

```text
--nuisance
--ppca-background
--regularization-mu
--standard-scale / --no-standard-scale
--nuisance-ridge-alpha
--nuisance-cross-fit
--nuisance-block-normalization
--nuisance-write-background / --no-nuisance-write-background
```

```text
protspace/cli/prepare.py
```

Supports both final ρPCA background pathways:

```text
--nuisance          build background from annotations
--ppca-background   use existing explicit background
```

`prepare` is the main command for annotation-defined nuisance backgrounds because it has access to embeddings, FASTA, default annotations, and custom CSV annotations.

```text
protspace/cli/project.py
```

Supports only explicit background mode:

```text
--ppca-background
```

`project` does **not** support `--nuisance`, because it does not have the full annotation/FASTA context needed to build annotation-defined backgrounds.

```text
protspace/data/processors/pipeline.py
```

Wires everything together. For every requested `ppca` projection, the pipeline either loads an explicit background from `--ppca-background` or builds one from `--nuisance`.

---

## How to use ρPCA

ρPCA is requested like other ProtSpace reducers:

```bash
-m ppca2
```

or together with other methods:

```bash
-m pca2,umap2,ppca2
```

Usually, use PCA, UMAP, and ρPCA together so the ρPCA result can be compared against standard baselines.

---

## Option A: annotation-defined nuisance background

Use this when the nuisance is described by a UniProt/default annotation or a custom CSV column.

Example with a custom 3FTx CSV annotation:

```bash
protspace prepare \
  -i embeddings.h5:prot_t5 \
  -f 3FTx_sequences_full_or_mature.fasta \
  -a default \
  -a annotations_3ftx_ppca.csv \
  -m pca2,umap2,ppca2 \
  --nuisance "sp_in_embedding:type=binary;missing=zero;scale=0.5;cross_fit=5" \
  -o results/3ftx_rhopca
```

Example with sequence length as nuisance:

```bash
protspace prepare \
  -i toxprot_embeddings.h5:prot_t5 \
  -f toxprot_full.fasta \
  -a default \
  -m pca2,umap2,ppca2 \
  --nuisance "length:type=continuous;transform=log1p;basis=spline;n_knots=6;scale=0.5;cross_fit=5" \
  -o results/toxprot_length_rhopca
```

Example with multiple nuisances:

```bash
protspace prepare \
  -i toxprot_embeddings.h5:prot_t5 \
  -f toxprot_full.fasta \
  -a default \
  -m pca2,umap2,ppca2 \
  --nuisance "signal_peptide:type=binary;missing=zero;scale=0.5;cross_fit=5" \
  --nuisance "length:type=continuous;transform=log1p;basis=spline;n_knots=6;scale=0.5;cross_fit=5" \
  --nuisance "order:type=categorical;missing=category;min_count=10;scale=0.5;cross_fit=5" \
  -o results/toxprot_combined_rhopca
```

---

## Option B: explicit background

Use this when the background has already been created externally.

```bash
protspace prepare \
  -i embeddings.h5:prot_t5 \
  -f sequences.fasta \
  -a default \
  -m pca2,umap2,ppca2 \
  --ppca-background background.h5 \
  -o results/explicit_background_rhopca
```

For `project`, only explicit background mode is supported:

```bash
protspace project \
  -i embeddings.h5:prot_t5 \
  -m ppca2 \
  --ppca-background background.h5 \
  -o results/project_rhopca
```

---

## What happens if `--nuisance` is not defined?

If `ppca2` or `ppca3` is requested, ProtSpace needs a background.

Therefore:

```text
-m ppca2 + --nuisance          OK: background is built from annotations
-m ppca2 + --ppca-background   OK: explicit background is loaded
-m ppca2 without either        ERROR
```

If no ρPCA method is requested, then `--nuisance` and `--ppca-background` are not needed.

For example, this does not need any ρPCA background:

```bash
protspace prepare \
  -i embeddings.h5:prot_t5 \
  -m pca2,umap2 \
  -o results/no_rhopca
```

---

## Defaults for scaling and regularization

The intended ρPCA defaults are:

```text
regularization_mu = 1e-6
standard_scale = False
```

Therefore, this command:

```bash
protspace prepare \
  -i embeddings.h5:prot_t5 \
  -a default \
  -m pca2,umap2,ppca2 \
  --nuisance "sp_in_embedding:type=binary;missing=zero;scale=0.5;cross_fit=5" \
  -o output
```

is equivalent to:

```bash
protspace prepare \
  -i embeddings.h5:prot_t5 \
  -a default \
  -m pca2,umap2,ppca2 \
  --nuisance "sp_in_embedding:type=binary;missing=zero;scale=0.5;cross_fit=5" \
  --no-standard-scale \
  --regularization-mu 1e-6 \
  -o output
```

`--regularization-mu 1e-6` is the numerical default for the generalized eigenproblem:

```text
Σ_T v = λ(Σ_B + μI)v
```

`--no-standard-scale` is also the default for ρPCA. This means ρPCA runs in the original protein language model embedding space unless the user explicitly requests otherwise.

If a user intentionally wants to standardize every embedding dimension before ρPCA, they can opt in with:

```bash
--standard-scale
```

---

## Why ρPCA defaults to no standard scaling

For the ρPCA experiments, we want the target covariance and nuisance-background covariance to be computed in the same original protein language model embedding coordinate system.

The target covariance is:

```text
Σ_T = covariance of the target embeddings
```

The nuisance covariance is:

```text
Σ_B = covariance of the explicit or annotation-defined background
```

The annotation-defined background is estimated in the same embedding space as the target embeddings. If every embedding dimension is standard-scaled before the eigensolve, the relative scale of the original embedding dimensions is changed before comparing `Σ_T` and `Σ_B`.

For this reason, the final ρPCA default is:

```text
standard_scale = False
```

This is a ρPCA-specific default. It does not change PCA, UMAP, t-SNE, PaCMAP, MDS, or LocalMAP behavior. The `--standard-scale / --no-standard-scale` option is only used by the ρPCA reducer.

---

## Important parameters and hyperparameters

### Reduction method

Use one of:

```text
-m ppca2
-m ppca3
-m pca2,umap2,ppca2
```

Recommended for experiments:

```text
-m pca2,umap2,ppca2
```

### Background source

Exactly one background source is needed when using ρPCA:

```text
--nuisance
```

or:

```text
--ppca-background
```

Use `--nuisance` for annotation-defined backgrounds.

Use `--ppca-background` if the background HDF5 already exists.

### `--regularization-mu`

Used in the generalized eigenproblem:

```text
Σ_T v = λ(Σ_B + μI)v
```

Default:

```text
1e-6
```

Increase it if the background covariance is unstable or nearly singular.

### `--nuisance` definition

The general format is:

```text
--nuisance "column:type=<type>;option=value;option=value"
```

The `column` must exist in the annotation table after combining default annotations and custom CSV annotations.

Examples:

```text
sp_in_embedding:type=binary;missing=zero;scale=0.5;cross_fit=5
length:type=continuous;transform=log1p;basis=spline;n_knots=6;scale=0.5;cross_fit=5
order:type=categorical;missing=category;min_count=10;scale=0.5;cross_fit=5
pfam:type=multilabel;sep=semicolon;min_count=20;scale=0.5;cross_fit=5
```

Recommended nuisance types:

```text
binary        yes/no or true/false annotations
continuous    numeric annotations such as length
categorical   one label per protein, e.g. kingdom/order/family
multilabel    multiple labels per protein, e.g. Pfam/GO/keywords
auto          let the implementation infer the type
```

---

## Recommended optimization workflow

1. Always start with PCA, UMAP, and ρPCA together:

```bash
-m pca2,umap2,ppca2
```

2. Start with a single nuisance first.

Example:

```bash
--nuisance "sp_in_embedding:type=binary;missing=zero;scale=0.5;cross_fit=5"
```

3. Compare PCA vs ρPCA.

Check whether the known nuisance separation is reduced.

4. Tune `scale`.

Try:

```text
scale=0.25
scale=0.5
scale=1.0
```

5. Add additional nuisances only after the single-nuisance result is understood.

Example:

```bash
--nuisance "signal_peptide:type=binary;missing=zero;scale=0.5;cross_fit=5" \
--nuisance "length:type=continuous;transform=log1p;basis=spline;n_knots=6;scale=0.5;cross_fit=5"
```

6. Use negative controls if possible.

A good control is a shuffled nuisance annotation. If shuffled labels produce the same effect as real labels, the result is probably not meaningful.

---

## Practical recommended defaults

For 3FTx signal-peptide nuisance:

```bash
--nuisance "sp_in_embedding:type=binary;missing=zero;scale=0.5;cross_fit=5"
```

For ToxProt length nuisance:

```bash
--nuisance "length:type=continuous;transform=log1p;basis=spline;n_knots=6;scale=0.5;cross_fit=5"
```

For ToxProt taxonomy nuisance:

```bash
--nuisance "order:type=categorical;missing=category;min_count=10;scale=0.5;cross_fit=5"
```

For combined nuisance correction:

```bash
--nuisance "signal_peptide:type=binary;missing=zero;scale=0.5;cross_fit=5" \
--nuisance "length:type=continuous;transform=log1p;basis=spline;n_knots=6;scale=0.5;cross_fit=5" \
--nuisance "order:type=categorical;missing=category;min_count=10;scale=0.5;cross_fit=5"
```

The following options are now implicit defaults and do not need to be written unless you want to override them:

```bash
--no-standard-scale --regularization-mu 1e-6
```

---

## Interpretation

ρPCA does not delete or edit the embeddings.

Instead, it changes the projection objective.

The original embeddings stay unchanged. The annotation-defined nuisance background tells ρPCA which directions should be penalized in the 2D projection.

The correct interpretation is:

```text
ρPCA finds directions with high target variance and low declared-nuisance variance.
```

Not:

```text
ρPCA completely removes the nuisance from the data.
```

---

## Variables, hyperparameters, defaults, and behavior if absent

| Variable / option | Where used | Default value | What happens if absent? | Notes / optimization guidance |
|---|---|---:|---|---|
| `-m`, `--methods` | `prepare`, `project` | `pca2` | Only PCA2 is run. No ρPCA is run, so no background is needed. | Use `pca2,umap2,ppca2` for experiments. |
| `ppca2` / `ppca3` | method list | absent | ρPCA is not run. | If present, one background source is required. |
| `--nuisance` | `prepare` only | absent | If `ppca2/ppca3` is requested and no `--ppca-background` is supplied, ProtSpace raises an error. | Repeatable; use one declaration per nuisance. |
| `--ppca-background` | `prepare`, `project` | absent | If `ppca2/ppca3` is requested and no `--nuisance` is supplied, ProtSpace raises an error. In `project`, this is required for ρPCA. | Must match the embedding dimension of the target HDF5. |
| `--regularization-mu` | ρPCA eigensolve | `1e-6` | Uses `1e-6`. | Increase if `Σ_B + μI` is unstable or nearly singular. |
| `--standard-scale / --no-standard-scale` | ρPCA reducer only | `--no-standard-scale` (`standard_scale=False`) | ρPCA runs in the original embedding coordinate system. | Use `--standard-scale` only if you intentionally want column standardization before ρPCA. |
| `--nuisance-ridge-alpha` | annotation→embedding model | `10.0` | Each nuisance uses ridge alpha `10.0` unless overridden inside the nuisance spec by `ridge_alpha=...`. | Tune when nuisance predictions are too weak or too unstable. |
| `--nuisance-cross-fit` | annotation nuisance builder | `5` | Each nuisance uses 5-fold cross-fitting unless overridden inside the nuisance spec by `cross_fit=...`. | Use especially for categorical and multilabel nuisances. Use `0` or `1` to disable. |
| `--nuisance-block-normalization` | background assembly | `none` | Blocks are centered and signed but not trace- or row-count-normalized. | Alternatives: `trace`, `row_count`. Tune when combining very different nuisance blocks. |
| `--nuisance-write-background / --no-nuisance-write-background` | `prepare` nuisance mode | write enabled | Background HDF5, manifest, diagnostics, and effect summary are written for inspection. | Keep enabled for experiments and debugging. |
| `column` in `--nuisance` | nuisance spec | required | If absent or not found in annotations, ProtSpace raises an error. | Must match a column after merging default and custom annotations. |
| `type` | nuisance spec | `auto` | Type is inferred from values. | Prefer explicit `binary`, `continuous`, `categorical`, or `multilabel` for reproducibility. |
| `scale` | nuisance spec | `0.5` | Uses `0.5`. | Main tuning knob. Try `0.25`, `0.5`, `1.0`. Larger means stronger nuisance penalty. |
| `ridge_alpha` | nuisance spec | global `--nuisance-ridge-alpha` (`10.0`) | Uses global ridge alpha. | Per-nuisance override for the annotation→embedding model. |
| `cross_fit` | nuisance spec | global `--nuisance-cross-fit` (`5`) | Uses global cross-fit setting. | Per-nuisance override. Set `0` or `1` to fit all rows without cross-fitting. |
| `random_state` | nuisance spec / global reducer params | global `--random-state` (`42`) | Uses global random seed. | Controls fold assignment for cross-fitting. |
| `missing` | nuisance spec | `auto` | Default missing policy depends on inferred/declared type. | Defaults: continuous→`drop`, binary→`category`, categorical→`category`, multilabel→`empty`. |
| `missing_value` | nuisance spec | `__missing__` | Missing categorical/binary values use `__missing__` if encoded as a category. | Useful for reproducible labels. |
| `transform` | continuous nuisance | `log1p` | Numeric values are log-transformed before basis construction. | Use `identity` if raw scale is desired. |
| `basis` | continuous nuisance | `spline` | Continuous nuisance is encoded with spline basis. | Alternatives: `linear`, `polynomial`. |
| `n_knots` | continuous spline nuisance | `6` | Six spline knots are used. | Tune for smoother or more flexible length effects. |
| `degree` | continuous polynomial/spline nuisance | `3` | Cubic polynomial/spline degree is used. | Reduce for small datasets if unstable. |
| `min_count` | categorical/multilabel nuisance | `5` | Rare categories/tokens below this count are handled by `rare_policy`. | Increase for broad annotations such as Pfam/GO/taxonomy. |
| `max_features` | categorical/multilabel nuisance | `5000` | At most 5000 encoded features are kept. | Lower if memory or overfitting becomes an issue. |
| `rare_policy` | categorical/multilabel nuisance | `rare` | Rare labels are grouped into `rare_value`. | Alternative: `drop`. |
| `rare_value` | categorical/multilabel nuisance | `__rare__` | Rare labels are named `__rare__`. | Mostly diagnostic. |
| `sep` | multilabel nuisance | `;` | Multilabel entries are split on semicolons. | Named aliases supported: `semicolon`, `pipe`, `comma`, `tab`, `space`. |
| `strip_scores` | multilabel nuisance | `True` | Token suffix after `|` is stripped. | Useful for ProtSpace annotation strings with scores/evidence. |
| `strip_evidence` | multilabel nuisance | `True` | Evidence suffix after `|` is stripped. | Usually keep enabled. |
| `token_mode` | multilabel nuisance | `raw` | Tokens are used as raw strings after splitting/stripping. | Alternatives: `accession`, `name`. |
| `condition_on` / `condition` | nuisance spec | absent | No residualization/conditioning is applied. | Use only if one nuisance should be estimated after accounting for another. |
| `condition_type` | conditioning design | `auto` | Type is inferred for the conditioning variable. | Same logic as nuisance type inference. |
| `condition_missing` | conditioning design | `category` | Missing conditioning values become a category. | Conservative default. |
| `condition_min_count` | conditioning design | `5` | Rare conditioning categories below 5 are handled as rare/drop. | Tune for high-cardinality conditioning variables. |
| `condition_max_features` | conditioning design | `5000` | At most 5000 conditioning features are kept. | Lower if memory is high. |
| `fit_intercept` | nuisance ridge model | `True` | Intercept is fitted in the nuisance model. | Usually leave enabled. |
| `-a default` | annotation source | absent unless provided | Default annotations are not loaded unless requested. | Use `-a default` when nuisance columns come from default annotations or when plots should include them. |
| `-a custom.csv` | annotation source | absent unless provided | Custom CSV columns are not available. | Required when `--nuisance` references a custom column such as `sp_in_embedding`. |
| `-f`, `--fasta` | FASTA context | absent | Some annotation fetching or similarity steps may be unavailable. | Needed for FASTA-based workflows and sequence-dependent annotations. |