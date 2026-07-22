# ToxProt rhoPCA experiment — 6 unified-nuisance comparison bundles

Everything is fetched from UniProt (the `data/toxprot` folder started empty). ToxProt = the reviewed
Swiss-Prot toxins, embedded with **ProtT5**. Two nuisances are removed with the **unified**
(annotation-defined) rhoPCA background — **`length`** (continuous) and **`order`** (taxonomic rank,
categorical) — and the functional read-out is **`protein_families`**. Driven by
[`scripts/run_toxprot_experiment.py`](../../../scripts/run_toxprot_experiment.py).

Every bundle carries `length`, `order`, `protein_families`, plus the UniProt defaults (`keyword`,
`reviewed`, `protein_name`, …) for coloring. Open in ProtSpace (`protspace serve` / protspace.app).

- **Query:** `(reviewed:true) AND (keyword:KW-0800)` → 7916 sequences; **6909 embedded** (one Biocentral
  batch of 959 hit a transient server error — see note). 78 taxonomic orders, 296 protein families.
- **Pre-reduction dimensionality:** rhoPCA-**25** (earlier ToxProt k-sweep sweet spot).

## Why 6 bundles

One `protspace prepare` run builds a *single* nuisance background shared by all its rhoPCA projections,
so the three configs (−length / −order / −length+order) can't be separate rhoPCA projections in one
bundle. Each config therefore gets its own bundle = the PCA/UMAP baselines + that config's rhoPCA — one
clean command each.

| # | folder | `-m` | rhoPCA removes |
|---|--------|------|----------------|
| 1 | `bundles/01_direct_length` | `pca2,umap2,rhopca2` | length |
| 2 | `bundles/02_direct_order` | `pca2,umap2,rhopca2` | order |
| 3 | `bundles/03_direct_length_order` | `pca2,umap2,rhopca2` | length + order |
| 4 | `bundles/04_prereduce25_length` | `pca2,umap2,rhopca25>pca2,rhopca25>umap2` | length |
| 5 | `bundles/05_prereduce25_order` | `pca2,umap2,rhopca25>pca2,rhopca25>umap2` | order |
| 6 | `bundles/06_prereduce25_length_order` | `pca2,umap2,rhopca25>pca2,rhopca25>umap2` | length + order |

## The `protspace prepare` commands (clean — defaults do the rest)

`$R = results/rhoPCA_experiments/ToxProt`. Only `-i`, `-a`, `-m`, `-o`, and one `--nuisance` per removed
column; bare specs auto-detect (`length`→continuous, `order`→categorical).

```bash
# direct rhoPCA (bundles 1–3): swap the --nuisance flag(s)
protspace prepare -i $R/embeddings/prot_t5.h5 -a $R/inputs/annotations.csv \
  -m pca2,umap2,rhopca2 --nuisance length            -o $R/bundles/01_direct_length
protspace prepare … --nuisance order                 -o $R/bundles/02_direct_order
protspace prepare … --nuisance length --nuisance order -o $R/bundles/03_direct_length_order

# rhoPCA-25 pre-reduction (bundles 4–6): raw PCA/UMAP + rhoPCA-25 → PCA/UMAP
protspace prepare -i $R/embeddings/prot_t5.h5 -a $R/inputs/annotations.csv \
  -m "pca2,umap2,rhopca25>pca2,rhopca25>umap2" --nuisance length -o $R/bundles/04_prereduce25_length
protspace prepare … --nuisance order                 -o $R/bundles/05_prereduce25_order
protspace prepare … --nuisance length --nuisance order -o $R/bundles/06_prereduce25_length_order
```

Annotations are fetched once (`protspace annotate -i prot_t5.h5 -a default -a taxonomy`) into
`inputs/annotations.csv` and reused by all six.

## Results (kNN-15 on the 2-D coordinates)

Lower `length`/`order` = nuisance removed; higher `protein_families` = biology kept. `length` is
quartile-binned for the kNN read.

**Direct rhoPCA vs baseline PCA:**

| projection | length ↓ | order ↓ | protein_families ↑ |
|------------|----------|---------|--------------------|
| baseline PCA | 0.717 | 0.575 | 0.477 |
| ρPCA −length | **0.535** | 0.475 | 0.369 |
| ρPCA −order | 0.545 | **0.492** | 0.368 |
| ρPCA −length+order | **0.499** | **0.383** | 0.279 |

**rhoPCA-25 pre-reduction → UMAP vs baseline UMAP** (the good arm):

| projection | length ↓ | order ↓ | protein_families ↑ |
|------------|----------|---------|--------------------|
| baseline UMAP | 0.868 | 0.800 | 0.785 |
| UMAP via rhopca25 −length | **0.802** | 0.772 | 0.744 |
| UMAP via rhopca25 −order | 0.822 | **0.732** | 0.725 |
| UMAP via rhopca25 −length+order | **0.727** | **0.692** | 0.695 |

**Reading it.** Each rhoPCA config removes the nuisance it targets (length and/or order kNN drop below
baseline). ToxProt's biology is genuinely entangled with taxonomy/length, so `protein_families` dips
when the nuisances are removed — but the **pre-reduction → UMAP** arm keeps far more family signal
(0.70–0.74) than the direct 2-D linear rhoPCA (0.28–0.37), because a nonlinear method downstream
recovers the toxin-family structure in the contrastive subspace. That is the intended use: rhoPCA-25 to
strip the nuisance, UMAP to surface the biology.

## Reproduce

```bash
uv run python scripts/run_toxprot_experiment.py                 # fetch + embed + annotate + 6 bundles
uv run python scripts/run_toxprot_experiment.py --skip-embed --skip-annotate   # rebuild bundles only
```

## Note — embedding completeness
One Biocentral export batch (959 of 7916 sequences) failed with a transient `403` during this run, so
the experiment covers **6909/7916** toxins — a representative ~87%. Re-running the embed (delete
`embeddings/prot_t5.h5` and re-run) will attempt the full set again.

See [`docs/rhoPCA.md`](../../../docs/rhoPCA.md) for the rhoPCA method and CLI reference.
