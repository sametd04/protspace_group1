# 3FTx rhoPCA experiment — 5 ProtSpace bundles

Five ProtSpace bundles built from `data/3FTx/rhoPCA_experiments/3FTx_data.xlsx` (1427 three-finger
toxins), re-embedded with **ProtT5** (Biocentral). Driven by
[`scripts/run_3ftx_core_experiment.py`](../../../scripts/run_3ftx_core_experiment.py). Every bundle
carries the annotations `sp_in_embedding` (signal peptide present in the embedded sequence: 783 yes /
644 no) and `major_group` (Short-chain / Plesiotypic / Non-standard / Long-chain / Ly-6), plus the
default UniProt columns (`protein_families`, `keyword`, `length`, …) fetched by accession.

Open any bundle in ProtSpace (`protspace serve` or [protspace.app](https://protspace.app)).

## The 5 bundles

| # | folder | projection(s) | idea |
|---|--------|---------------|------|
| 1 | `bundles/01_full_or_mature_pca_umap` | PCA 2 + UMAP 2 | full-or-mature set; PC1 tracks the signal peptide |
| 2 | `bundles/02_mature_only_positive_control` | PCA 2 + UMAP 2 | SP removed by **SignalP-6.0** → biology appears (positive control) |
| 3 | `bundles/03_full_or_mature_rhopca_manual_delta` | ρPCA 2 | recovers biology via a manual paired-Δ background |
| 4 | `bundles/04_full_or_mature_rhopca_unified` | ρPCA 2 | same, from just the `sp_in_embedding` annotation (unified background) |
| 5 | `bundles/05_full_or_mature_rhopca30_prereduction` | PCA 2 · UMAP 2 · PCA 2 (via rhopca30) · UMAP 2 (via rhopca30) | rhoPCA as a **pre-reduction** (unified) to 30-D, then PCA/UMAP, next to the raw baselines |

`full_or_mature` = `full_seq` where present (783, `sp_in_embedding=yes`) else `mature_seq` (644, `no`).
`mature_only` = SignalP-stripped full sequence for the 783, curated `mature_seq` passthrough for the 644.

## The `protspace prepare` commands — as clean as possible

Defaults do the rest; a flag appears only when it's actually required (`-i`, `-a`, `-m`, `-o`, plus a
background source for the ρPCA methods). No `-f`, `--keep-tmp`, `-v`, or model-name suffix needed.
`$R = results/rhoPCA_experiments/3FTx`; the embedding name resolves to `prot_t5` from the file, and the
`sp_in_embedding` nuisance auto-detects as binary.

```bash
# 1 — full-or-mature, PCA + UMAP
protspace prepare -i $R/embeddings/full_or_mature/prot_t5.h5 -a $R/inputs/annotations.csv \
  -m pca2,umap2 -o $R/bundles/01_full_or_mature_pca_umap

# 2 — mature-only (SignalP-stripped), PCA + UMAP   (positive control)
protspace prepare -i $R/embeddings/mature_only/prot_t5.h5 -a $R/inputs/annotations.csv \
  -m pca2,umap2 -o $R/bundles/02_mature_only_positive_control

# 3 — full-or-mature rhoPCA, manual paired-Δ background
protspace prepare -i $R/embeddings/full_or_mature/prot_t5.h5 -a $R/inputs/annotations.csv \
  -m rhopca2 --rhopca-background $R/inputs/manual_delta_background.h5 \
  -o $R/bundles/03_full_or_mature_rhopca_manual_delta

# 4 — full-or-mature rhoPCA, unified annotation background
protspace prepare -i $R/embeddings/full_or_mature/prot_t5.h5 -a $R/inputs/annotations.csv \
  -m rhopca2 --nuisance sp_in_embedding -o $R/bundles/04_full_or_mature_rhopca_unified

# 5 — rhoPCA-30 pre-reduction (unified) → PCA/UMAP, with raw baselines in the same bundle
protspace prepare -i $R/embeddings/full_or_mature/prot_t5.h5 -a $R/inputs/annotations.csv \
  -m "pca2,umap2,rhopca30>pca2,rhopca30>umap2" --nuisance sp_in_embedding \
  -o $R/bundles/05_full_or_mature_rhopca30_prereduction
```

## Bundle 5 — rhoPCA as a pre-reduction

rhoPCA reduces the 1024-D ProtT5 embedding to its top **30** contrastive (unified, SP-removing)
components first, then PCA and UMAP map that to 2-D. The bundle also holds the raw PCA/UMAP baselines so
you can compare in one view. **k = 30** is the optimum for 3FTx from an earlier k-sweep (`major_group`
separation peaked at k≈25–30); `rho_output_scale` is left at its default (`none` ≈ `target_var` here).

Measured on this run (kNN-15 on the 2-D coordinates; lower `sp` = nuisance removed, higher
`major_group` = biology kept):

| projection | sp kNN ↓ | major_group kNN ↑ |
|------------|----------|-------------------|
| PCA 2 (baseline) | 0.972 | 0.701 |
| UMAP 2 (baseline) | 0.965 | 0.861 |
| PCA 2 (via rhopca30) | **0.603** | 0.721 |
| UMAP 2 (via rhopca30) | **0.716** | **0.884** |

The pre-reduction strips the signal-peptide nuisance (UMAP sp 0.965 → 0.716, PCA 0.972 → 0.603) while
**retaining or improving** the toxin-group biology (UMAP major_group 0.861 → 0.884) — no hand-cropping
of the SP, just the unified annotation background.

## Step-by-step (what the driver does)

1. **Parse** `3FTx_data.xlsx` → per protein: `identifier`, `full_seq` (783 precursors), `mature_seq`,
   `major_group`, `uniprot_id`.
2. **Inputs:** `inputs/full_or_mature.fasta` (1427), `inputs/annotations.csv`
   (`identifier, sp_in_embedding, major_group` + UniProt defaults fetched by accession via
   `protspace annotate`), and `inputs/full_only.fasta` (the 783 SP precursors, for SignalP).
3. **Embed** full-or-mature with ProtT5 → `embeddings/full_or_mature/prot_t5.h5`.
4. **Phase A** (no SignalP): build bundles **1, 4, 5**.
5. **SignalP-6.0** (external, [services.healthtech.dtu.dk/services/SignalP-6.0](https://services.healthtech.dtu.dk/services/SignalP-6.0/))
   on `inputs/full_only.fasta`. Settings: **Organism = Eukarya, Output = Short, Model = Slow.** Place the
   downloaded results under `inputs/SignalP/` (the `output.json` / `processed_entries.fasta` are what the
   driver reads).
6. **Phase B** (from SignalP): derive `inputs/mature_only.fasta`, embed it, build the manual paired-Δ
   background `inputs/manual_delta_background.h5` (Δ = E(full) − E(mature), B = [+Δ, −Δ]), and build
   bundles **2, 3**.

## Reproduce

```bash
uv run python scripts/run_3ftx_core_experiment.py --phase a                        # bundles 1, 4, 5
uv run python scripts/run_3ftx_core_experiment.py --phase b \
  --signalp-output results/rhoPCA_experiments/3FTx/inputs/SignalP --skip-embed      # bundles 2, 3
```

See [`docs/rhoPCA.md`](../../../docs/rhoPCA.md) for the rhoPCA method and full CLI reference.
