#!/usr/bin/env python3
"""Core ToxProt rhoPCA experiment — fetch from UniProt, 6 comparison bundles.

Everything is fetched from UniProt (the `data/toxprot` folder starts empty): the
reviewed Swiss-Prot toxin set, ProtT5 embeddings, and the annotations. The two
nuisances removed with the **unified** (annotation-defined) rhoPCA background are
**length** (continuous) and **order** (taxonomic rank, categorical); the functional
read-out is **protein_families**.

One `protspace prepare` run builds a single nuisance background shared by all its
rhoPCA projections, so the three nuisance configs (−length / −order / −length+order)
live in separate bundles — each bundle is the PCA/UMAP baselines plus one config's
rhoPCA, i.e. one clean command. Six bundles total:

  Direct rhoPCA:                 -m pca2,umap2,rhopca2 --nuisance <config>
    01_direct_length, 02_direct_order, 03_direct_length_order
  rhoPCA-25 pre-reduction:       -m "pca2,umap2,rhopca25>pca2,rhopca25>umap2" --nuisance <config>
    04_prereduce25_length, 05_prereduce25_order, 06_prereduce25_length_order

k=25 is the ToxProt pre-reduction sweet spot from an earlier k-sweep.

    uv run python scripts/run_toxprot_experiment.py            # full run
    uv run python scripts/run_toxprot_experiment.py --dry-run  # print the commands
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

from protspace.data.loaders.h5 import parse_identifier
from protspace.data.loaders.query import query_uniprot

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results" / "rhoPCA_experiments" / "ToxProt"
# Reviewed Swiss-Prot entries carrying the Toxin keyword (KW-0800) — the Tox-Prot set.
QUERY = "(reviewed:true) AND (keyword:KW-0800)"
PREREDUCE_K = 25  # ToxProt pre-reduction optimum (earlier k-sweep sweet spot)

# One bundle per (approach, nuisance config). Each is a single clean prepare command.
BUNDLES = [
    ("01_direct_length", "pca2,umap2,rhopca2", ["length"]),
    ("02_direct_order", "pca2,umap2,rhopca2", ["order"]),
    ("03_direct_length_order", "pca2,umap2,rhopca2", ["length", "order"]),
    ("04_prereduce25_length", f"pca2,umap2,rhopca{PREREDUCE_K}>pca2,rhopca{PREREDUCE_K}>umap2", ["length"]),
    ("05_prereduce25_order", f"pca2,umap2,rhopca{PREREDUCE_K}>pca2,rhopca{PREREDUCE_K}>umap2", ["order"]),
    ("06_prereduce25_length_order", f"pca2,umap2,rhopca{PREREDUCE_K}>pca2,rhopca{PREREDUCE_K}>umap2", ["length", "order"]),
]


def log(msg):
    print(f"[toxprot] {msg}", flush=True)


def run(cmd, dry_run):
    log("$ " + " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run([str(c) for c in cmd], cwd=str(REPO), check=True)


def fetch_fasta(out, dry_run):
    """Download the ToxProt FASTA and rewrite headers to bare accessions so the h5
    keys, annotations, and projections all align on the UniProt accession."""
    inp = out / "inputs"
    fasta = inp / "toxprot.fasta"
    if dry_run:
        log(f"(dry-run) would query UniProt {QUERY!r} → {fasta}")
        return fasta
    inp.mkdir(parents=True, exist_ok=True)
    _, raw = query_uniprot(QUERY, save_to=inp / "toxprot_raw.fasta")
    n = 0
    with open(raw) as src, fasta.open("w") as dst:
        for line in src:
            if line.startswith(">"):
                acc = parse_identifier(line[1:].strip().split()[0])
                dst.write(f">{acc}\n")
                n += 1
            else:
                dst.write(line)
    Path(raw).unlink(missing_ok=True)
    log(f"fetched {n} ToxProt sequences → {fasta}")
    return fasta


def embed(fasta, emb_dir, dry_run, skip_embed):
    h5 = emb_dir / "prot_t5.h5"
    if skip_embed and h5.exists():
        log(f"skip embed (exists): {h5}")
        return h5
    run(["protspace", "embed", "-i", fasta, "-e", "prot_t5", "-o", emb_dir], dry_run)
    return h5


def annotate(h5, out, dry_run, skip_annotate):
    """Fetch UniProt default + taxonomy annotations once → annotations.csv
    (identifier + length, order, protein_families, defaults)."""
    ann = out / "inputs" / "annotations.csv"
    if skip_annotate and ann.exists():
        log(f"skip annotate (exists): {ann}")
        return ann
    parq = out / "inputs" / "annotations.parquet"
    run(["protspace", "annotate", "-i", h5, "-a", "default", "-a", "taxonomy",
         "-o", parq, "--no-scores"], dry_run)
    if dry_run:
        return ann
    df = pq.read_table(parq).to_pandas()
    if "identifier" not in df.columns and "protein_id" in df.columns:
        df = df.rename(columns={"protein_id": "identifier"})
    df.to_csv(ann, index=False)
    for col in ("length", "order", "protein_families"):
        n = int((df[col].astype(str).str.strip().replace("nan", "").str.len() > 0).sum()) if col in df else 0
        log(f"annotation '{col}': {n}/{len(df)} non-empty")
    log(f"wrote {ann}  (cols: {list(df.columns)})")
    parq.unlink(missing_ok=True)
    return ann


def prepare(h5, ann, methods, out_dir, nuisances, dry_run):
    # Clean by design: only -i / -a / -m / -o plus one --nuisance per removed column
    # (bare specs auto-detect: length→continuous, order→categorical).
    extra = []
    for nz in nuisances:
        extra += ["--nuisance", nz]
    run(["protspace", "prepare", "-i", h5, "-a", ann, "-m", methods, "-o", out_dir, *extra],
        dry_run)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-embed", action="store_true",
                    help="reuse existing prot_t5.h5 instead of re-embedding")
    ap.add_argument("--skip-annotate", action="store_true",
                    help="reuse existing annotations.csv instead of re-fetching")
    args = ap.parse_args()

    fasta = fetch_fasta(args.out, args.dry_run)
    h5 = embed(fasta, args.out / "embeddings", args.dry_run, args.skip_embed)
    ann = annotate(h5, args.out, args.dry_run, args.skip_annotate)
    for name, methods, nuisances in BUNDLES:
        prepare(h5, ann, methods, args.out / "bundles" / name, nuisances, args.dry_run)
    log("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
