#!/usr/bin/env python3
"""Core 3FTx rhoPCA experiment — clean re-run from the raw table, 4 ProtSpace bundles.

From `data/3FTx/rhoPCA_experiments/3FTx_data.xlsx` this rebuilds everything and shells
out to the installed `protspace` CLI. Each of the 4 steps writes one
`data.parquetbundle` (open in ProtSpace), each carrying the default annotations plus
`sp_in_embedding` and `major_group`:

  Phase A (no SignalP):
    1. full-or-mature  → PCA + UMAP
    4. full-or-mature  → rhoPCA (unified annotation background from sp_in_embedding)
  Phase B (needs SignalP-6.0 output on the 783 full sequences):
    2. mature-only     → PCA + UMAP   (positive control; SignalP-stripped)
    3. full-or-mature  → rhoPCA (manual paired-Δ background, Δ = E(full) − E(mature))

full-or-mature = full_seq where present (783, sp_in_embedding=yes) else mature_seq
(644, no). SignalP is used only to derive the mature-only set + the Δ background.

    uv run python scripts/run_3ftx_core_experiment.py --phase a            # steps 1 & 4
    uv run python scripts/run_3ftx_core_experiment.py --phase b \
        --signalp-output <processed_entries.fasta|prediction_results.txt>  # steps 2 & 3
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "data" / "3FTx" / "rhoPCA_experiments" / "3FTx_data.xlsx"
DEFAULT_OUT = REPO / "results" / "rhoPCA_experiments" / "3FTx"
# Bare spec: type auto-detects to binary, no missing values, everything else default
# (scale=1.0, cross_fit=1, ridge_alpha=auto). As clean as it gets.
NUISANCE = "sp_in_embedding"
# Optimal rhoPCA pre-reduction dimensionality for 3FTx (earlier k-sweep: major_group
# kNN peaked at k=30); default rho_output_scale is used (none ≈ target_var here).
PREREDUCE_K = 30
# UniProt-style accession regex (protspace only fetches defaults for bare accessions).
ACCESSION = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
DEFAULT_COLS = ["ec", "keyword", "length", "protein_families", "reviewed",
                "gene_name", "protein_name", "uniprot_kb_id"]


def log(msg):
    print(f"[3ftx] {msg}", flush=True)


def run(cmd, dry_run):
    log("$ " + " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run([str(c) for c in cmd], cwd=str(REPO), check=True)


def write_fasta(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for ident, seq in records:
            fh.write(f">{ident}\n{seq}\n")
    log(f"wrote {path}  ({len(records)} records)")


def clean_seq(v):
    return "".join(str(v).split()).upper().replace("*", "") if pd.notna(v) else ""


def load_table(data_path):
    df = pd.read_excel(data_path)
    df["_full"] = df["full_seq"].map(clean_seq)
    df["_mature"] = df["mature_seq"].map(clean_seq)
    df["_id"] = df["identifier"].astype(str)
    df["_has_full"] = df["_full"].str.len() > 0
    df["_embed_seq"] = np.where(df["_has_full"], df["_full"], df["_mature"])
    df["_acc"] = df["uniprot_id"].astype("string").where(
        df["uniprot_id"].astype("string").str.match(ACCESSION, na=False))
    df = df[df["_embed_seq"].str.len() > 0].reset_index(drop=True)
    return df


def fetch_default_annotations(df, out, dry_run):
    """Fetch protspace `-a default` (UniProt) columns for the accession-bearing rows.

    Identifiers here are `SP|P01385|Species`, which protspace's accession filter
    skips — so `-a default` in prepare yields empty columns. We instead fetch keyed
    on the bare accession (uniprot_id, == the identifier's middle field) and return
    the default columns joined back per identifier. Rows without an accession get
    empty strings. Returns a DataFrame indexed by identifier with DEFAULT_COLS.
    """
    empty = pd.DataFrame({"_id": df["_id"], **dict.fromkeys(DEFAULT_COLS, "")})
    sub = df.dropna(subset=["_acc"]).drop_duplicates("_acc")
    if dry_run or sub.empty:
        log(f"(defaults) {len(sub)} accession rows would be fetched via protspace annotate")
        return empty
    tmp = out / "inputs" / "_accession.fasta"
    parq = out / "inputs" / "_defaults.parquet"
    write_fasta(list(zip(sub["_acc"], sub["_embed_seq"], strict=True)), tmp)
    run(["protspace", "annotate", "-i", tmp, "-a", "default", "-o", parq,
         "--no-scores"], dry_run)
    fetched = pq.read_table(parq).to_pandas().rename(columns={"identifier": "_acc"})
    keep = ["_acc"] + [c for c in DEFAULT_COLS if c in fetched.columns]
    merged = df[["_id", "_acc"]].merge(fetched[keep], on="_acc", how="left")
    for c in DEFAULT_COLS:
        if c not in merged.columns:
            merged[c] = ""
    merged[DEFAULT_COLS] = merged[DEFAULT_COLS].fillna("")
    n = int((merged["protein_families"].astype(str).str.len() > 0).sum())
    log(f"(defaults) fetched UniProt annotations for {len(sub)} accessions "
        f"({n} rows now carry protein_families)")
    tmp.unlink(missing_ok=True)
    parq.unlink(missing_ok=True)
    return merged[["_id", *DEFAULT_COLS]]


# --------------------------------------------------------------------------- #
# Shared inputs
# --------------------------------------------------------------------------- #
def build_common_inputs(df, out, dry_run):
    inp = out / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    fasta = inp / "full_or_mature.fasta"
    ann = inp / "annotations.csv"
    write_fasta(list(zip(df["_id"], df["_embed_seq"], strict=True)), fasta)
    # SP-bearing precursors only (headers = identifier) → run SignalP-6.0 on this file;
    # its output feeds Phase B (parse_signalp_mature keys on the same identifier).
    sp = df[df["_has_full"]]
    write_fasta(list(zip(sp["_id"], sp["_full"], strict=True)), inp / "full_only.fasta")
    defaults = fetch_default_annotations(df, out, dry_run)
    ann_df = pd.DataFrame({
        "identifier": df["_id"],
        "sp_in_embedding": np.where(df["_has_full"], "yes", "no"),
        "major_group": df["major_group"].astype(str),
    }).merge(defaults.rename(columns={"_id": "identifier"}), on="identifier", how="left")
    ann_df.to_csv(ann, index=False)
    log(f"wrote {ann}  (sp_in_embedding: {int(df['_has_full'].sum())} yes / "
        f"{int((~df['_has_full']).sum())} no; cols: {list(ann_df.columns)})")
    return fasta, ann


def embed(fasta, emb_dir, dry_run, skip_embed):
    h5 = emb_dir / "prot_t5.h5"
    if skip_embed and h5.exists():
        log(f"skip embed (exists): {h5}")
        return h5
    run(["protspace", "embed", "-i", fasta, "-e", "prot_t5", "-o", emb_dir], dry_run)
    return h5


def prepare(h5, ann, methods, out_dir, dry_run, extra=()):
    # Kept intentionally minimal — defaults do the rest. `-a default` is omitted (the
    # identifiers `SP|acc|Species` aren't bare accessions, so protspace's default fetch
    # skips them); the default UniProt columns are pre-merged into `ann` by accession in
    # fetch_default_annotations, so the CSV is the single annotation source.
    cmd = ["protspace", "prepare", "-i", h5, "-a", ann, "-m", methods, "-o", out_dir, *extra]
    run(cmd, dry_run)


# --------------------------------------------------------------------------- #
# Phase A — bundles 1, 4, 5 (no SignalP needed)
# --------------------------------------------------------------------------- #
def phase_a(df, out, dry_run, skip_embed):
    fasta, ann = build_common_inputs(df, out, dry_run)
    h5 = embed(fasta, out / "embeddings" / "full_or_mature", dry_run, skip_embed)
    prepare(h5, ann, "pca2,umap2",
            out / "bundles" / "01_full_or_mature_pca_umap", dry_run)
    prepare(h5, ann, "rhopca2",
            out / "bundles" / "04_full_or_mature_rhopca_unified", dry_run,
            extra=["--nuisance", NUISANCE])
    # Bundle 5 — rhoPCA-30 pre-reduction (unified background) → PCA/UMAP 2-D, with the
    # raw PCA/UMAP baselines in the same bundle for direct comparison.
    k = PREREDUCE_K
    prepare(h5, ann, f"pca2,umap2,rhopca{k}>pca2,rhopca{k}>umap2",
            out / "bundles" / "05_full_or_mature_rhopca30_prereduction", dry_run,
            extra=["--nuisance", NUISANCE])
    return fasta, ann, h5


# --------------------------------------------------------------------------- #
# Phase B — steps 2 & 3 (needs SignalP output)
# --------------------------------------------------------------------------- #
def _find_signalp_source(p):
    """Resolve a SignalP path (file or results dir) to the best usable source."""
    if p.is_dir():
        for cand in ("output_all_results/output.json", "output.json",
                     "processed_entries.fasta", "output_all_results/output.gff3"):
            if (p / cand).exists():
                return p / cand
        raise SystemExit(f"No SignalP output found under {p}")
    return p


def _restore_id(key, unmangle, full_by_id):
    """Map a (possibly `|`→`_` mangled) SignalP header back to the original id."""
    if key in full_by_id:
        return key
    return unmangle.get(key)


def parse_signalp_mature(signalp_path, df):
    """Return {identifier: mature_seq} from SignalP-6.0 output.

    Prefers `output.json` (all inputs, exact cleavage sites). Also accepts
    `processed_entries.fasta` (SP-stripped) or a gff3/prediction table (cleavage
    sites → mature = full[cs:]). SignalP replaces `|` with `_` in headers, so ids are
    restored via a reverse map. Proteins predicted with no SP keep the full sequence.
    """
    import json
    import re

    full_by_id = {i: s for i, s in zip(df["_id"], df["_full"], strict=True) if s}
    unmangle = {i.replace("|", "_"): i for i in full_by_id}
    cs_re = re.compile(r"pos\.?\s*(\d+)\s+and")  # "between pos. 21 and 22" → cut at 21
    p = _find_signalp_source(Path(signalp_path))
    mature, n_sp, n_nosp = {}, 0, 0

    if p.suffix == ".json":
        seqs = json.loads(p.read_text())["SEQUENCES"]
        for key, rec in seqs.items():
            oid = _restore_id(key, unmangle, full_by_id)
            if oid is None:
                continue
            m = cs_re.search(rec.get("CS_pos", "") or "")
            if m:
                mature[oid] = full_by_id[oid][int(m.group(1)):]
                n_sp += 1
            else:  # "Other" — no signal peptide, nothing to strip
                mature[oid] = full_by_id[oid]
                n_nosp += 1
    elif p.suffix in (".fasta", ".fa"):
        ident, seq = None, []
        for line in p.read_text().splitlines():
            if line.startswith(">"):
                if ident is not None:
                    mature[ident] = "".join(seq)
                ident = _restore_id(line[1:].split()[0], unmangle, full_by_id)
                seq = []
            elif ident is not None:
                seq.append(line.strip())
        if ident is not None:
            mature[ident] = "".join(seq)
        n_sp = len(mature)
    else:  # gff3 / prediction table with a cleavage-site column
        for line in p.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            oid = _restore_id(line.split("\t")[0].split()[0], unmangle, full_by_id)
            m = cs_re.search(line) or re.search(r"CS pos[^0-9]*(\d+)", line)
            if oid and m:
                mature[oid] = full_by_id[oid][int(m.group(1)):]
                n_sp += 1

    mature = {k: "".join(v.split()).upper() for k, v in mature.items() if v}
    log(f"parsed SignalP from {p.name}: {len(mature)} sequences "
        f"({n_sp} SP-cleaved, {n_nosp} no-SP → full kept)")
    return mature


def build_mature_only(df, signalp_mature, out):
    fasta = out / "inputs" / "mature_only.fasta"
    records, n_sig, n_fallback = [], 0, 0
    for _id, has_full, mature_col in zip(df["_id"], df["_has_full"], df["_mature"],
                                         strict=True):
        if has_full and _id in signalp_mature and signalp_mature[_id]:
            records.append((_id, signalp_mature[_id]))
            n_sig += 1
        else:
            records.append((_id, mature_col))  # 644 no-full, or missing SignalP
            n_fallback += 1
    write_fasta(records, fasta)
    log(f"mature_only: {n_sig} SignalP-stripped / {n_fallback} passthrough")
    return fasta


def build_manual_delta(full_h5, mature_h5, df, out, dry_run):
    """Δ = E(full) − E(mature) over the 783 SP proteins; B = [+Δ, −Δ]."""
    bg = out / "inputs" / "manual_delta_background.h5"
    if dry_run:
        log(f"(dry-run) would build manual-Δ background → {bg}")
        return bg
    with h5py.File(full_h5) as f:
        efull = {k: f[k][:] for k in f}
    with h5py.File(mature_h5) as f:
        emat = {k: f[k][:] for k in f}
    rows, ids = [], []
    for _id, has_full in zip(df["_id"], df["_has_full"], strict=True):
        if has_full and _id in efull and _id in emat:
            d = efull[_id].astype(np.float64) - emat[_id].astype(np.float64)
            rows.append(d)
            ids.append(_id)
    delta = np.vstack(rows)
    signed = np.vstack([delta, -delta])
    bg.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(bg, "w") as f:
        f.attrs["model_name"] = "background"
        f.attrs["background_kind"] = "manual_paired_delta"
        for i, _id in enumerate(ids):
            f.create_dataset(f"{_id}__delta__pos", data=signed[i].astype(np.float32))
            f.create_dataset(f"{_id}__delta__neg",
                             data=signed[len(ids) + i].astype(np.float32))
    log(f"wrote {bg}  ({signed.shape[0]} rows from {len(ids)} SP pairs)")
    return bg


def phase_b(df, out, signalp_output, dry_run, skip_embed):
    if not signalp_output and not dry_run:
        raise SystemExit("Phase B needs --signalp-output <processed_entries.fasta | "
                         "prediction_results.txt>")
    fasta_full = out / "inputs" / "full_or_mature.fasta"
    ann = out / "inputs" / "annotations.csv"
    if not fasta_full.exists() and not dry_run:
        raise SystemExit("Run --phase a first (need full_or_mature inputs + embeddings).")
    signalp_mature = parse_signalp_mature(signalp_output, df) if signalp_output else {}
    fasta_mat = build_mature_only(df, signalp_mature, out)
    h5_mat = embed(fasta_mat, out / "embeddings" / "mature_only", dry_run, skip_embed)
    h5_full = out / "embeddings" / "full_or_mature" / "prot_t5.h5"
    bg = build_manual_delta(h5_full, h5_mat, df, out, dry_run)
    prepare(h5_mat, ann, "pca2,umap2",
            out / "bundles" / "02_mature_only_positive_control", dry_run)
    prepare(h5_full, ann, "rhopca2",
            out / "bundles" / "03_full_or_mature_rhopca_manual_delta", dry_run,
            extra=["--rhopca-background", str(bg)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--phase", choices=["a", "b", "all"], default="a")
    ap.add_argument("--signalp-output", type=Path, default=None,
                    help="SignalP-6.0 output for phase B (fasta or prediction table)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-embed", action="store_true",
                    help="reuse existing prot_t5.h5 instead of re-embedding")
    args = ap.parse_args()

    df = load_table(args.data)
    log(f"loaded {len(df)} proteins from {args.data.name}")
    if args.phase in ("a", "all"):
        phase_a(df, args.out, args.dry_run, args.skip_embed)
    if args.phase in ("b", "all"):
        phase_b(df, args.out, args.signalp_output, args.dry_run, args.skip_embed)
    log("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
