# Project-Datasets: Download + erster Research-Workflow

Diese Datei erklärt dir:
1. wie du die 4 Projekt-Datasets herunterlädst,
2. wie du alles **klein** hältst (Default),
3. was du konkret tun musst, um euer **erstes Research-Ziel** zu erreichen.

## 1) Erstes Research-Ziel (aus `proj5_dr_exploration`)

Für den Start ist das erste praktische Ziel aus *Setup*:
- bestehende ProtSpace-Visualisierung reproduzieren (z. B. 3FTx UMAP),
- Benchmark-Harness laufen lassen (Methoden + Metriken vergleichbar).

In eurem Repo heißt das konkret:
1. kleine Datasets laden,
2. Embeddings erzeugen,
3. `src/protspace/benchmark/run.py` auf Dataset setzen,
4. Benchmark laufen lassen und Metriken/Plots vergleichen.

### Aufgabe 1 (konkret): 3FTx-UMAP reproduzieren + Benchmark-Basis erzeugen

Das ist eure erste Team-Aufgabe. Ziel: ihr habt einen lauffähigen End-to-End-Flow für ein kleines,
reales Dataset und eine erste Vergleichsbasis für DR-Methoden.

Schritte:
1. **3FTx klein herunterladen** (oder mit den Default-Subsets aus diesem Script):
   ```bash
   uv run python scripts/download_project_datasets.py --datasets 3ftx -v
   ```
2. **Embeddings für 3FTx erzeugen**:
   ```bash
   uv run protspace embed -i data/project_datasets/3ftx/3ftx_reviewed.fasta -e prot_t5 -o output_3ftx/tmp
   ```
3. **Benchmark auf 3FTx laufen lassen** (`run.py` auf `DATASET = "3ftx"`):
   ```bash
   uv run python src/protspace/benchmark/run.py
   ```
4. **Ergebnisse prüfen**:
   - `src/protspace/benchmark/results/3ftx/metrics.csv`
   - `src/protspace/benchmark/results/3ftx/trustworthiness.png`
   - `src/protspace/benchmark/results/3ftx/knn_preservation.png`
   - `src/protspace/benchmark/results/3ftx/continuity.png`

**Definition of Done für Aufgabe 1:**
- Benchmark läuft ohne Fehler für 3FTx.
- CSV mit 2D/3D-Metriken pro Methode ist vorhanden.
- Die drei Plot-Dateien sind neu erzeugt und zeigen Unterschiede zwischen Methoden.
- Ihr könnt die gleiche Pipeline danach für `toxprot` wiederverwenden.

## 2) Downloader-Script

Script:
- `scripts/download_project_datasets.py`

Default-Verhalten ist jetzt **klein/subset** (nicht full):
- `3ftx`: max 300
- `toxprot`: max 1500
- `cath_s40`: max 3000
- `swissprot_rr` (reviewed): max 10000

Damit bleibt der Download typischerweise im niedrigen bis mittleren zweistelligen MB-Bereich statt hunderten MB.

## 3) Download ausführen

### Empfohlen (klein, alle Datasets)
```bash
uv run python scripts/download_project_datasets.py -v
```

### Nur einzelne Datasets
```bash
uv run python scripts/download_project_datasets.py --datasets 3ftx toxprot -v
```

### Noch kleiner machen
```bash
uv run python scripts/download_project_datasets.py \
  --3ftx-max 150 \
  --toxprot-max 600 \
  --cath-max 1200 \
  --swissprot-max 3000 \
  -v
```

### Vollständige Datasets (groß!)
```bash
uv run python scripts/download_project_datasets.py --full -v
```

## 4) Embeddings erzeugen (für Benchmark `run.py`)

`run.py` erwartet H5-Dateien unter:
- `output_<dataset>/tmp/prot_t5.h5`

Beispiel-Kommandos:

```bash
uv run protspace embed -i data/project_datasets/3ftx/3ftx_reviewed.fasta -e prot_t5 -o output_3ftx/tmp
uv run protspace embed -i data/project_datasets/toxprot/toxprot_reviewed.fasta -e prot_t5 -o output_toxprot/tmp
uv run protspace embed -i data/project_datasets/cath_s40/cath_s40.fa -e prot_t5 -o output_cath_s40/tmp
uv run protspace embed -i data/project_datasets/swissprot/swissprot_reviewed.fasta -e prot_t5 -o output_swissprot_rr/tmp
```

Wenn du SwissProt mit Redundanzreduktion erzeugt hast (`--swissprot-identity`), dann stattdessen:

```bash
uv run protspace embed -i data/project_datasets/swissprot/swissprot_rr.fasta -e prot_t5 -o output_swissprot_rr/tmp
```

## 5) Benchmark für erstes Ziel laufen lassen

In `src/protspace/benchmark/run.py`:
- `DATASET = "3ftx"` (oder `toxprot`, `cath_s40`, `swissprot_rr`)
- `EMBEDDING_MODEL = "prot_t5"`

Dann:
```bash
uv run python src/protspace/benchmark/run.py
```

Outputs:
- `src/protspace/benchmark/results/<dataset>/metrics.csv`
- `src/protspace/benchmark/results/<dataset>/*.png`

## 6) Empfohlene Reihenfolge für euer Team

1. Start mit `3ftx` (klein, schnell, gut für Reproduktion).
2. Danach `toxprot` (größer, realistischer).
3. Erst dann `cath_s40` / `swissprot_rr`.
4. Für schnelle Iteration immer zuerst mit kleinen Limits arbeiten.
