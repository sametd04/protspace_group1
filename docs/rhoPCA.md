# rhoPCA (ρPCA) — contrastive PCA in ProtSpace

rhoPCA is **contrastive PCA**: instead of keeping the directions of largest *total* variance (plain
PCA), it keeps directions with **high variance in your target embeddings but low variance in a
declared "background"** — so a known nuisance (signal peptide, sequence length, taxonomy, batch, …)
is suppressed and the biology you care about is surfaced.

## The idea

Given the target covariance `Σ_T` and a background covariance `Σ_B`, rhoPCA solves the generalized
eigenproblem

```
Σ_T v = λ (Σ_B + μI) v
```

and keeps the top eigenvectors, ranked by the ratio **ρ = vᵀΣ_T v / vᵀΣ_B v** (target variance ÷
background variance). Directions that are strong in the data but nearly absent from the background
score highest. `μ` (Tikhonov regularization) keeps `Σ_B` invertible.

Notes:
- The embeddings themselves are **not** modified — only the projection objective changes.
- The eigenvectors are **Σ_B-conjugate** (whitened by the background), not Euclidean-orthonormal.
- Method strings: **`rhopca2`** / **`rhopca3`** (2- or 3-D viewer projections). Larger `k` is allowed only
  as a *pre-reduction* stage (see below).
- **Backward-compatible aliases:** the old spellings `ppca2`/`ppca3`, `--ppca-background`, and the
  `PPCAReducer` class keep working — `ppca` is normalized to the canonical `rhopca` at parse time.

## Backgrounds — the decisive choice

rhoPCA needs a background. There are exactly two ways to supply one:

1. **External background HDF5** — `--rhopca-background bg.h5`. You built the background yourself (e.g. a
   paired Δ = E(full) − E(mature)). Works in both `prepare` and `project`.
2. **Annotation-defined background** — `--nuisance "<spec>"` (repeatable). `prepare` builds the
   background automatically: a ridge model predicts the embedding shift explained by the annotation,
   and a signed block `B = [+αG, −αG]` becomes the background. `prepare` only (needs annotations).

**One background per run.** A single `prepare` run builds **one** background per embedding set, shared
by *every* rhoPCA projection in that run — the viewer `rhopca2`/`rhopca3` and any pre-reduction
`rhopcaK>…` alike (repeated `--nuisance` specs combine into that one background). So you cannot get
`rhoPCA −length` and `rhoPCA −order` as two separate projections in the same bundle. To compare
*different* nuisance removals side by side, run them as **separate `prepare` runs / bundles** — one per
background — as the 3FTx and ToxProt experiment READMEs do.

The background HDF5 must have the **same feature dimension** as the target embeddings.

## CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `-m rhopca2` / `rhopca3` | — | request a rhoPCA projection (with any other methods) |
| `--rhopca-background PATH` | — | external background HDF5 (required for `rhopca` in `project`) |
| `--nuisance "SPEC"` | — | annotation-defined nuisance (repeatable; `prepare` only) |
| `--regularization-mu` | `1e-6` | Tikhonov μ added to `Σ_B` |
| `--standard-scale / --no-standard-scale` | `--no-standard-scale` | column-standardize before the eigensolve (off by default for rhoPCA, so `Σ_T` and `Σ_B` share scale) |
| `--nuisance-ridge-alpha` | `auto` | ridge α for annotation→embedding models; omit to auto-tune (CV-GCV), pass a value to fix |
| `--nuisance-cross-fit` | `1` | cross-fit folds (0/1 = fit all rows; set ≥2 for out-of-fold debiasing on high-dimensional designs) |
| `--nuisance-block-normalization` | `none` | `none` / `trace` / `row_count` per-nuisance block scaling |
| `--nuisance-write-background / --no-...` | write | write `background.h5` + manifest + diagnostics for `--nuisance` runs |
| `-m "rhopca2:rho_output_scale=V"` | `none` | output scaling (`none` / `target_var` / `unit_var`) — only matters for pre-reduction (see below) |

### Nuisance spec grammar (`--nuisance`)

```
<column>[:key=value;key=value;...]
```

The column name is required; everything after `:` is optional. **A bare `--nuisance "column"` works** —
the type is auto-detected and all other keys take their defaults (see below), so minimal specs Just
Work. Examples:

- `--nuisance "sp_in_embedding"`  (type auto-detected → binary)
- `--nuisance "sp_in_embedding:type=binary;missing=zero"`
- `--nuisance "length:type=continuous;transform=log1p;basis=spline"`
- `--nuisance "order:type=categorical;cross_fit=5"`

Pass `--nuisance` multiple times to remove several nuisances at once.

#### Keys reference

| key | applies to | default | meaning |
|-----|-----------|---------|---------|
| `type` | all | `auto` | `binary` / `categorical` / `continuous` / `multilabel`, or `auto` (detected) |
| `missing` | all | `auto` | how blank/absent values are handled (per-type default — see below) |
| `scale` | all | `1.0` | nuisance strength α in `B = [+αE, −αE]` (`Σ_B ∝ scale²`) |
| `ridge_alpha` | all | `auto` | ridge shrinkage of the annotation→embedding model; `auto` = CV-tuned (GCV), or a float to fix it |
| `cross_fit` | all | `1` | out-of-fold folds; `0/1` = fit-all; `≥2` = order-invariant debiasing |
| `transform` | continuous | `auto` | `auto` = CV-selected between `log1p` (skewed positives) and `none`/`identity` |
| `basis` | continuous | `spline` | feature expansion: `spline` / `poly` / `linear` |
| `n_knots` | continuous | `auto` | `auto` = CV-selected spline knots (from {4,6,8,12}); or a fixed int |
| `degree` | continuous | `3` | spline/poly degree |
| `min_count` | categorical/multilabel | `5` | fold levels/tokens rarer than this into `rare_policy` |
| `max_features` | categorical/multilabel | `5000` | cap on encoded columns |
| `rare_policy` | categorical/multilabel | `rare` | `rare` (bucket) or `drop` (remove those rows) |
| `sep`, `token_mode` | multilabel | `;`, `raw` | token splitting / normalization (`raw`/`accession`/`name`) |
| `condition_on` | all | — | partial out another annotation first (fit the nuisance on the residual) |
| `name` | all | column | label for the block in diagnostics |

#### Automatic type detection

With `type=auto` (the default), the column is classified by its values: **binary** if the values are
truthy/falsey (`yes/no`, `1/0`, `true/false`, …) or there are exactly two levels; **multilabel** if a
substantial fraction contains `;`-separated tokens; **continuous** if ≥90% parse as numbers with >10
distinct values; otherwise **categorical**. Example: `sp_in_embedding` (`yes`/`no`) → binary.

#### Missing-value policies

"Missing" means blank / `nan` / `none` / `null` / `na` / `n/a` (note: `unknown` is *not* treated as
missing). `missing=auto` resolves per type:

| type | `auto` default | notes |
|------|---------------|-------|
| binary | `category` | missing becomes its own `__missing__` level (doesn't assert absent); use `missing=zero` to force false |
| categorical | `category` | missing is its own level |
| continuous | `drop` | rows with no numeric value are dropped; also `median` / `zero` |
| multilabel | `empty` | missing → no tokens |

If a column has no missing values (like `sp_in_embedding`), the policy is a no-op.

`transform=log1p` on a **continuous** column with **negative** values automatically falls back to
`identity` (with a warning) rather than silently dropping those rows — so signed nuisances are safe with
the default spec; positive-only columns (length, counts) still use `log1p`.

**Cross-fit is order-invariant.** When `cross_fit ≥ 2`, folds are assigned by a canonical ordering of
the protein identifiers, not by row position — so the background (and the ρPCA projection) is
reproducible regardless of the input row order or identifier scheme. The default `cross_fit=1` fits the
ridge once on all rows (fit-all, deterministic); raise it to debias the effect estimate for
high-dimensional designs (many categories, spline bases).

#### `ridge_alpha` — the annotation→embedding model

`ridge_alpha` is the **L2 (Tikhonov) regularization strength of the ridge regression that estimates how
much the embedding is explained by the nuisance annotation** — the model that produces the effect `E`
used as the background. To build the background rhoPCA fits a linear model:

- **inputs** `Φ` (n × p): the *encoded annotation* (binary → one 0/1 column, categorical → one-hot,
  continuous → spline features);
- **target** `X` (n × d): the *centered embeddings*;
- **fit** weights `W` minimizing `‖X − ΦW‖² + ridge_alpha·‖W‖²` (sklearn `Ridge(alpha=ridge_alpha)`).

The prediction `E = ΦW` is the part of the embedding the annotation can linearly explain — this is the
`G` in `B = [+scale·E, −scale·E]`, and `Σ_B ∝ EᵀE`.

The `ridge_alpha·‖W‖²` penalty shrinks `W` toward zero — **large** = smaller/cleaner `E` (removes less,
won't chase noise), **small** = larger `E` (risks overfitting spurious directions, worse for
high-dimensional categorical/multilabel/spline designs).

**`ridge_alpha` is auto-tuned by default.** With the default `ridge_alpha=auto`, α is chosen **per
nuisance** by the ridge's own cross-validated predictive score (sklearn `RidgeCV`, efficient closed-form
GCV — no ρPCA eigensolve, no biology label, ~a fraction of a second). This picks the α that best
*generalizes* the annotation→embedding fit, so it helps exactly where regularization matters (many
categories / spline bases) and is a near-no-op where it doesn't (a 1-column binary design like
`sp_in_embedding`, where α only rescales a rank-1 effect and the projection is unchanged). The selection
is **order-invariant** (GCV is leave-one-out; the rare sparse fallback uses canonical-id folds) and
deterministic. **Pass a value to fix it**: per-spec `ridge_alpha=10`, or globally
`--nuisance-ridge-alpha 10`. The chosen α and its source (`auto_gcv` / `auto_kfold` / `explicit`) are
reported per nuisance in `diagnostics.md` / `effect_summary.csv` / the projection `info_json`.

Similarly, for **continuous** nuisances the `transform` (`log1p` vs `identity`) and spline `n_knots` are
CV-selected by the same score when left as `auto`; fixing either disables that part of the search. (No
optuna — the objective is a small, convex ridge CV, so GCV is cheaper, deterministic, and dependency-free.)

`ridge_alpha` regularizes *within* each fit, while `cross_fit` debiases *across* folds (out-of-fold
prediction) — two complementary guards against an over-optimistic background.

**Three distinct "regularization / α" knobs — don't confuse them:**

| knob | default | acts on | effect |
|------|---------|---------|--------|
| `ridge_alpha` | `10.0` | the annotation→embedding **ridge model** (`fit_ridge`) | shrinks the estimated effect `E` — changes its **magnitude *and* direction** (genuinely affects the projection) |
| `scale` (α in `[+αE,−αE]`) | `1.0` | multiplies `E` into the background | rescales `Σ_B ∝ scale²`; with tiny `μ` only rescales eigenvalues, **not directions** (near no-op) |
| `regularization_mu` | `1e-6` | added to `Σ_B` in the eigensolve `Σ_T v = λ(Σ_B+μI)v` | keeps `Σ_B` invertible — a numerical stabilizer, not a modeler |

**block-normalization** (`--nuisance-block-normalization`, default `none`) only matters with **several**
nuisances: `none` keeps each block at its natural predicted-effect magnitude (the stronger nuisance
dominates), `trace` rescales each block to equal total variance (balanced removal), `row_count`
upweights blocks with fewer rows. For a single nuisance all three are equivalent up to a global scale,
so `none` is the right default.

**standard-scale** (`--standard-scale`, default off) column-standardizes the embedding before the
eigensolve. Leave it **off** for rhoPCA: `Σ_T` and `Σ_B` come from the same embedding space and must
share scale; z-scoring would up-weight low-variance (often noise) dimensions of the pLM embedding.

## Commands (`protspace prepare`)

```bash
# 1) rhoPCA with an external background, alongside PCA & UMAP for comparison
protspace prepare -i emb.h5:prot_t5 \
  -m "pca2,umap2,rhopca2" \
  --rhopca-background background.h5 \
  --regularization-mu 1e-6 --no-standard-scale \
  -o out

# 2) rhoPCA with an annotation-defined background (built inside prepare)
protspace prepare -i emb.h5:prot_t5 \
  -m "pca2,rhopca2" \
  --nuisance "sp_in_embedding:type=binary;missing=zero;scale=0.5" \
  -o out

# 3) Remove several nuisances at once (length + taxonomy)
protspace prepare -i emb.h5:prot_t5 \
  -m "rhopca2" \
  --nuisance "length:type=continuous;transform=log1p;basis=spline" \
  --nuisance "order:type=categorical;scale=0.5" \
  -o out

# 4) 3-D rhoPCA
protspace prepare -i emb.h5:prot_t5 -m "rhopca3" --rhopca-background background.h5 -o out
```

`protspace project` works the same for `rhopca2`/`rhopca3` but requires `--rhopca-background` (it has no
annotation context to build one):

```bash
protspace project -i emb.h5 -m rhopca2 --rhopca-background background.h5 -o out
```

## rhoPCA as a pre-reduction (chaining)

rhoPCA can reduce the embedding to the **top-k components first**, then feed a downstream method — chain
stages with **`>`** in `-m`. The stage before `>` is the pre-reduction (any `k ≥ 2`); the **final**
stage is the viewer projection and must be **2- or 3-D**.

```bash
# rhoPCA → 30 dims (unified background), then UMAP → 2-D — defaults do the rest
protspace prepare -i emb.h5 -m "rhopca30>umap2" --nuisance sp_in_embedding -o out

# compare arms in one bundle: raw PCA/UMAP + rhoPCA-30 pre-reduction → PCA/UMAP
protspace prepare -i emb.h5 \
  -m "pca2,umap2,rhopca30>pca2,rhopca30>umap2" --nuisance sp_in_embedding -o out
```

Rules & tips:
- A **rhoPCA pre-stage must be first** (it needs the raw embedding to match its background).
- **Optimal k is data-dependent.** For 3FTx, an earlier k-sweep found the sweet spot at **k ≈ 25–30**
  (rhoPCA-30 maximised `major_group` separation). Pick k from your own nuisance/biology trade-off.
- **`rho_output_scale`** controls how the k axes are scaled before the downstream method (the axes are
  Σ_B-conjugate and ranked by ρ, not variance): the default `none` (raw, variance = ρ) works well and
  is what the clean commands use; `target_var` (project onto unit eigenvectors → PCA-like variance) is
  a near-equivalent alternative when feeding UMAP; `unit_var` z-scores each axis.
- The pre-reduction is cached within a run, so several downstream methods reuse one rhoPCA-k result.
- Any method can be a pre-stage (`pca50>umap2`, even `umap50>pca2`), but PCA-k / rhoPCA-k are the
  sensible linear choices; only rhoPCA removes the nuisance.

## Outputs

- `-m rhopca2` adds a `ρPCA 2` projection to the `.parquetbundle`; `get_params()` reports the eigenvalue
  ratios (ρ), background source, and sample count.
- With `--nuisance` (and background writing on), `prepare` writes to
  `<out>/…/rhopca_nuisance_backgrounds/<model>/`: `background.h5`, `background.manifest.json`,
  `diagnostics.md`, `effect_summary.csv`, and an aligned annotations CSV.

## Troubleshooting

- **`LinAlgError: Σ_B not positive definite`** → raise `--regularization-mu`, or use a larger /
  less-degenerate background.
- **Tuning `--regularization-mu`?** Rarely needed: the annotation background is low-rank (the nuisance
  spans only its design dimensions), so `μ` just floors the nuisance-free subspace and the projection is
  insensitive to its exact value across several orders of magnitude — the `1e-6` default sits well below
  the real nuisance eigenvalues. Only raise it to *soften* suppression or to recover from a
  `LinAlgError`.
- **Feature-dim mismatch** → the background must use the same embedding model/dimension as the target.
- **rhoPCA looks like PCA** → the background isn't capturing the nuisance; check the nuisance spec /
  `scale`, or inspect `diagnostics.md`.
- **Both `--rhopca-background` and `--nuisance` given** → choose one background source per run.

## Defaults at a glance
`n_components` 2/3 (viewer) or any k (pre-reduction) · `regularization_mu=1e-6` ·
`standard_scale=False` · `rho_output_scale=none` · nuisance `scale=1.0`, `ridge_alpha=auto`
(CV-GCV; pass a float to fix), `cross_fit=1` (fit-all; ≥2 = order-invariant out-of-fold),
continuous `transform=auto`/`n_knots=auto` (CV-selected), `basis=spline`, `degree=3`,
`min_count=5`, `max_features=5000`, `rare_policy=rare`, `block_normalization=none`.

Every default above has been audited as optimal for the intended use: `μ=1e-6` is the minimal
stabilizer, `standard_scale=False` keeps `Σ_T`/`Σ_B` on a shared scale, `scale`/`ridge_alpha`/
`cross_fit`/`transform`/`n_knots` were tuned or made self-tuning this cycle, and the categorical
knobs (`min_count`/`rare_policy`/…) preserve data (bucket rare levels rather than drop rows). No
default is a placeholder — override any per run when your data warrants it.
