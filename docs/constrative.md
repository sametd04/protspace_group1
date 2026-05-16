# Contrastive dimensionality reduction in ProtSpace

ProtSpace supports two contrastive dimension-reduction methods from the
[`pachterlab/rhopca`](https://github.com/pachterlab/rhopca) family:

| Method | Reducer name | Reference |
| --- | --- | --- |
| ρPCA   | `ppca`  | Carilli, Jackson & Pachter 2025 ([paper 1](https://doi.org/10.1101/2025.11.19.689125)) |
| k-ρPCA | `kppca` | Jackson, Carilli & Pachter 2026 ([paper 2](https://doi.org/10.64898/2026.04.08.717236)) |

Both maximise a Rayleigh quotient that compares **target** variance to
**background** variance, finding directions along which the target dataset is
spread relative to a reference distribution. Unlike PCA, the dominant axes
are not the directions of maximum total variance — they are the directions of
maximum *contrastive* variance.

This is useful for protein embeddings because the top PCs of a PLM dataset
are typically dominated by nuisance signals (sequence length, amino-acid
composition, signal-peptide presence, broad taxonomy). Contrastive DR lets
you subtract those out by choosing a background dataset that shares them and
then projecting away from the directions on which both vary.

## When to use which

**ρPCA** (`ppca`) is the linear baseline. Use it when you have a target
dataset and a separate background dataset and want to find directions where
the target is uniquely variable. This is the canonical case from paper 1.

**k-ρPCA** (`kppca`) adds a kernel matrix that weights the target covariance
by pairwise sample similarity. Use it when you want to emphasise *local*
contrastive structure — directions where clusters of related proteins vary
together — rather than the global axes that ρPCA recovers. In paper 2 the
kernel encodes spatial proximity (Visium spot coordinates); for protein
embeddings, ProtSpace exposes three kernel sources (see
[k-ρPCA kernel sources](#k-ρpca-kernel-sources) below).

A short rule of thumb: try `ppca` first. If the top eigenvalues are large
and the projection shows clean separation between known subgroups, that's
already a strong result. If the projection looks dominated by a small number
of outliers or by a single global axis that crosses your biological
subgroups, `kppca` is the next thing to try — it down-weights long-range
covariance contributions and tends to emphasise within-cluster structure.

## ρPCA

### Objective

For target data $X_T \in \mathbb{R}^{n_T \times d}$ and background
$X_B \in \mathbb{R}^{n_B \times d}$, ρPCA maximises the Rayleigh quotient

$$\mathcal{R}(v) \;=\; \frac{v^\top \hat\Sigma_T \, v}{v^\top \hat\Sigma_B \, v}$$

where $\hat\Sigma_T, \hat\Sigma_B$ are the per-set sample covariance matrices.
The maximisers are the top generalised eigenvectors of the problem
$\hat\Sigma_T v = \lambda \hat\Sigma_B v$, solved with
`scipy.linalg.eigh(Σ_T, Σ_B)`. The eigenvalues $\lambda_i$ are the
target/background variance ratios along each axis.

### Background sources

Two modes:

1. **External (Mode 1, canonical paper convention).** A separate background
   dataset is supplied via `--ppca-background`. The reducer receives it
   through `cfg.background_data` and uses it directly. Recommended.
2. **Auto-split (Mode 2, fallback).** No external background is given; the
   reducer subsamples the input itself using one of three strategies
   (`random`, `uniform`, `outlier`). This is a heuristic — the resulting
   $\hat\Sigma_B$ estimates the same distribution as $\hat\Sigma_T$, so the
   eigenvalues mostly reflect Tikhonov regularisation rather than biology.
   The pipeline emits a warning when this path is taken.

### Implementation details

- **Per-set standard-scaling.** Both target and background matrices are
  standardised independently (column mean 0, unit variance) before
  computing covariances. This matches the rhopca reference convention; it
  is critical for PLM embeddings where embedding dimensions have wildly
  heterogeneous scales. Disable with `--no-standard-scale` if needed.
- **Tikhonov regularisation.** A small $\mu I$ is added to $\hat\Sigma_B$
  before solving the generalised eigenproblem. Required whenever
  $n_B \leq d$, which is the default regime for PLM embeddings. The
  default $\mu = 10^{-3}$ is safe for PLM embeddings after standardisation.
- **Eigenvalue filtering.** Only positive finite eigenvalues are kept,
  matching `rhopca/utils/misc.py::generalized_eigen`. Tiny negative
  eigenvalues from numerical noise are discarded.
- **Sign normalisation.** Each eigenvector is sign-flipped so that the
  entry with largest absolute magnitude is positive. This gives
  deterministic output across runs but does not match the rhopca reference
  exactly (which leaves signs unnormalised); compare projections up to
  sign in any cross-validation.

## k-ρPCA

### Objective

Given an $n_T \times n_T$ kernel matrix $K$ encoding sample-similarity
weights between target samples, k-ρPCA maximises

$$\mathcal{R}(v) \;=\; \frac{v^\top \hat\Sigma_T^K v}{v^\top \hat\Sigma_B v}$$

with

$$\hat\Sigma_T^K \;=\; \frac{1}{n_T - 1} X_T^\top K \, X_T, \qquad
\hat\Sigma_B \;=\; \frac{1}{n_B - 1} X_B^\top X_B$$

after per-set centering. When $K = I$, this reduces exactly to ρPCA. When
$K$ encodes meaningful neighbourhood structure (proteins close in pLM space,
homologues by sequence similarity, structurally similar proteins by
Foldseek), $\hat\Sigma_T^K$ up-weights covariance contributions from pairs
of similar samples.

The intuition: a contrastive direction $v$ scores high under
$\hat\Sigma_T^K$ if pairs of *similar* target proteins disagree along $v$.
Pairs of unrelated proteins contribute little. So k-ρPCA recovers
directions where local groups of related proteins are spread out — and
ignores global axes (like signal-peptide presence) that cut indiscriminately
across all subgroups.

### k-ρPCA kernel sources

| `--kppca-kernel-source` | Distance metric | Notes |
| --- | --- | --- |
| `embedding` (default) | Euclidean in standardised target embedding space | Up-weights neighbours in pLM space. No extra input required. |
| `similarity` | $1 - \text{identity}$ from MMseqs2 | Up-weights homologues. Requires `--similarity` and `-f FASTA`. |
| `precomputed` | User-supplied $n_T \times n_T$ matrix | Most flexible. Plug in Foldseek TM-scores, GO Jaccard, phylogenetic distances, anything pairwise positive. Load via `--kppca-kernel-path`. |

The kernel function is selected separately with `--kppca-kernel`:

- `gaussian` (default): $K_{ij} = \exp(-d_{ij}^2 / 2h^2)$. Bandwidth $h$
  defaults to $\sqrt{\text{median}(d)}$, the rhopca reference heuristic;
  override with `--kppca-kernel-bandwidth`.
- `inverse_distance`: $K_{ij} = 1 / (d_{ij} + 10^{-6})$.
- `linear`: $K = I$. k-ρPCA reduces to ρPCA. Useful as a sanity check.

The kernel is applied to the target covariance only by default. Pass
`--kppca-background-kernel` to also kernel-weight $\hat\Sigma_B$ (uncommon;
paper 2 keeps $\hat\Sigma_B$ unweighted).

### Computational notes

k-ρPCA constructs a dense $n_T \times n_T$ kernel matrix in memory. For
typical targets ($n_T \lesssim 10^4$) this is fine — a 10,000-protein
target gives an 800 MB kernel. Beyond that, dense storage becomes
prohibitive (CATH S40 at ~30k proteins would need 7.2 GB). Sparse kernels
via $k$-NN or radius-neighbours graphs are not yet implemented in ProtSpace;
the rhopca reference supports them via
`sklearn.neighbors.radius_neighbors_graph` and is the right starting point
if you need to scale further.

## CLI reference

### Common flags (apply to both `ppca` and `kppca`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--ppca-background PATH` | (auto-split) | Background embeddings HDF5 file. Triggers Mode 1. |
| `--background-strategy {external,random,uniform,outlier}` | `outlier` | Auto-split policy when no `--ppca-background` is given. |
| `--background-ratio FLOAT` | `0.3` | Fraction of input used as background in auto-split mode. |
| `--regularization-mu FLOAT` | `1e-3` | Tikhonov $\mu$ added to $\hat\Sigma_B$. |
| `--standard-scale / --no-standard-scale` | on | Per-set standard-scaling. |

### k-ρPCA-only flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--kppca-kernel {gaussian,inverse_distance,linear}` | `gaussian` | Kernel function. |
| `--kppca-kernel-source {embedding,similarity,precomputed}` | `embedding` | Source of pairwise distances. |
| `--kppca-kernel-bandwidth FLOAT` | `0` (auto) | Gaussian bandwidth $h$. 0 auto-resolves to $\sqrt{\text{median}(d)}$. |
| `--kppca-kernel-path PATH` | — | Precomputed kernel matrix (`.npy`, `.h5`, `.parquet`). |
| `--kppca-background-kernel / --no-...` | off | Also kernel-weight $\hat\Sigma_B$. |

### Per-method overrides

The standard inline-parameter syntax works for both reducers, so you can run
multiple variants in a single command. For example:

```bash
protspace prepare \
  -i target.h5 --ppca-background bg.h5 \
  -m "kppca2:kernel=gaussian,kppca2:kernel=linear" \
  -o results/
```

produces two projections, one with a Gaussian kernel and one with $K = I$
(sanity check that k-ρPCA reduces to ρPCA).

## Usage examples

### ρPCA with an external background

```bash
protspace prepare \
  -i data/3ftx/prot_t5.h5:prot_t5 \
  --ppca-background data/toxprot/prot_t5.h5 \
  -m pca2,umap2,ppca2 \
  -a data/3ftx/3FTx_accession.csv \
  -o results/3ftx_ppca_vs_toxprot/
```

The output bundle contains PCA, UMAP, and ρPCA projections. The run log
records the eigenvalue ratios; for a real biological contrast (3FTx target
vs. toxprot background) these are typically in the 1000s.

### k-ρPCA with the default embedding-source kernel

```bash
protspace prepare \
  -i data/3ftx/prot_t5.h5:prot_t5 \
  --ppca-background data/toxprot/prot_t5.h5 \
  -m pca2,ppca2,kppca2 \
  -a data/3ftx/3FTx_accession.csv \
  -o results/3ftx_kppca_vs_toxprot/
```

Side-by-side with ρPCA. The Gaussian bandwidth auto-resolves to
$\sqrt{\text{median pairwise distance}}$; check `run.log` for the actual
value used.

### k-ρPCA with a sequence-similarity kernel

```bash
protspace prepare \
  -i data/3ftx/prot_t5.h5:prot_t5 \
  -f data/3ftx/sequences.fasta \
  --similarity \
  --ppca-background data/toxprot/prot_t5.h5 \
  --kppca-kernel-source similarity \
  -m kppca2 \
  -o results/3ftx_kppca_similarity/
```

`--similarity` runs MMseqs2 to build a pairwise identity matrix, which
k-ρPCA then converts to a Gaussian kernel via $d = 1 - \text{identity}$.

### k-ρPCA with a precomputed kernel

```bash
protspace prepare \
  -i data/3ftx/prot_t5.h5 \
  --ppca-background data/toxprot/prot_t5.h5 \
  --kppca-kernel-source precomputed \
  --kppca-kernel-path data/3ftx/foldseek_tm.npy \
  -m kppca2 \
  -o results/3ftx_kppca_foldseek/
```

The kernel matrix can be any positive symmetric $n_T \times n_T$ matrix.
The reducer symmetrises ($K \leftarrow (K + K^\top) / 2$) but does not
otherwise validate the values. Foldseek TM-scores are bounded in $[0, 1]$
and work directly as similarity weights.

## Diagnostics

Both reducers record post-fit diagnostics in the projections metadata,
which `protspace prepare` reads back into `run.log` automatically. For
ρPCA, you get:

- `eigenvalue_ratios`: the top $k$ eigenvalues, each = target/background
  variance ratio along its axis.
- `background_source`: `"external"` or one of the auto-split strategies.
- `n_background_samples`: $n_B$.

For k-ρPCA, additionally:

- `kernel_source`: which input was used.
- `kernel`: kernel function name.
- `kernel_bandwidth_used`: the actual $h$ value (after auto-resolution).
- `kernel_n_samples`: $n_T$ (size of $K$).

Use these to compare configurations across runs without re-opening the
viewer.

## Interpretation

The eigenvalues $\lambda_i$ have a clear semantic meaning: $\lambda_i$ is
the *ratio* of target variance to background variance along axis $i$. So:

- $\lambda_i \gg 1$ — strong contrastive signal; target spreads much
  further than background along this direction.
- $\lambda_i \approx 1$ — no contrastive signal; target and background have
  similar variance.
- $\lambda_i \ll 1$ — anti-contrastive; background spreads further than
  target. The corresponding eigenvectors at the *bottom* of the spectrum
  represent directions the target *lacks* relative to the background.
  (ProtSpace currently exposes only the top eigenvectors, but the bottom
  end is interpretable too.)

In Mode 1 with a biologically meaningful background, $\lambda_1$ values of
1000–10000 are typical. In auto-split mode, eigenvalues are
regularisation-dominated and not interpretable as variance ratios.

## Common pitfalls

**Auto-split eigenvalues look impressive but mean nothing.** If you see
$\lambda_1 \approx 7500$ without an external background, that ratio reflects
how Tikhonov shrinkage interacts with sub-sampling noise, not biological
contrast. Always check `background_source` in `run.log`.

**$n_B \ll d$ requires regularisation.** If your background has 500
samples and your embedding has 1024 dimensions, $\hat\Sigma_B$ is rank
$\leq 500$ and not invertible without $\mu I$. The default $\mu = 10^{-3}$
handles this. If `scipy.linalg.eigh` fails despite regularisation, increase
$\mu$ — the conditioning is the bottleneck.

**External background must use the same embedding model.** A target
embedded with ProtT5 and a background embedded with ESM-2 produce
covariances in different spaces; the generalised eigenproblem will run but
the eigenvectors are meaningless. ProtSpace checks that the embedding
dimensionality matches and refuses to run otherwise; check the embedding
model explicitly when constructing the background HDF5 file.

**Kernel bandwidth matters more than kernel function.** The auto-resolved
$h = \sqrt{\text{median}(d)}$ is a sensible default but is not always
optimal. If the target dataset has clear cluster structure, try
$h \approx$ within-cluster median distance to emphasise within-cluster
contrastive structure, or $h \approx$ between-cluster median to emphasise
between-cluster contrast. The `run.log` reports the actual bandwidth used,
so you can iterate.

**The viewer shows projections, not eigenvectors.** The 2D scatter plots
in the viewer are projections of the target onto the top two eigenvectors.
Two different background datasets will produce different eigenvectors and
therefore different scatter plots — there is no canonical layout to
compare across runs without explicit Procrustes alignment.

## References

- Carilli, M., Jackson, K. & Pachter, L. (2025). *The Rayleigh Quotient
  and Contrastive Principal Component Analysis I.* bioRxiv
  2025.11.19.689125. <https://doi.org/10.1101/2025.11.19.689125>
- Jackson, K., Carilli, M. & Pachter, L. (2026). *The Rayleigh Quotient
  and Contrastive Principal Component Analysis II.* bioRxiv
  2026.04.08.717236. <https://doi.org/10.64898/2026.04.08.717236>
- Reference implementation: <https://github.com/pachterlab/rhopca>
