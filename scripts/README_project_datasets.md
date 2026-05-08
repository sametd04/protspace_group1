# Project datasets (kurz)

## Download

```bash
# Standard (aktuelle Small-Subsets + H5)
uv run python scripts/download_project_datasets.py -v

# Nur bestimmte Datasets
uv run python scripts/download_project_datasets.py --datasets 3ftx toxprot -v

# Mehr Daten über Prozent (zusätzlich zu --*-max, kleineres Limit gewinnt)
uv run python scripts/download_project_datasets.py --percent 25 -v

# Alles vollständig
uv run python scripts/download_project_datasets.py --full -v

# Nur Größenbericht
uv run python scripts/download_project_datasets.py --report-only --datasets all
```

## Aktueller Stand (lokal) vs. Vollmenge

| Dataset | Aktuell FASTA | Aktuell Seqs | Vollmenge Seqs | Vollmenge FASTA (geschätzt) | Fehlt bis Vollmenge |
|---|---:|---:|---:|---:|---:|
| 3ftx | 46.4 KB | 300 | 461 | 71.3 KB | 161 |
| toxprot | 522.9 KB | 1,500 | 7,904 | 2.7 MB | 6,404 |
| pla2g2 | 59.6 KB | 448 | 448 | 59.6 KB | 0 |
| cath_s40 | 552.0 KB | 3,000 | 34,653 | 6.2 MB | 31,653 |
| swissprot_rr | 6.9 MB | 10,000 | 574,627 | 398.7 MB | 564,627 |

> Vollmengen-Größen sind Schätzungen auf Basis aktueller Dateigröße pro Sequenz.

## H5-Dateien

- H5 liegt direkt in `data/<dataset>/<embedder>.h5`
- Benchmark-Outputs liegen ebenfalls in `data/<dataset>/` (z. B. `metrics.csv`, Plots).
- Nur `benchmark.parquetbundle` bleibt in `src/protspace/benchmark/results/<dataset>/`.
