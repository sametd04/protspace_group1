#!/usr/bin/env python3
"""Prepare final 3FTx FASTA and annotation files for ProtSpace ρPCA experiments.

Inputs
------
An Excel sheet with at least these columns:
    identifier, id_new, uniprot_id, full_seq, mature_seq

Outputs
-------
    3ftx_full_or_mature.fasta
    3ftx_mature_only.fasta
    3ftx_annotations_full_or_mature.csv
    3ftx_annotations_mature_only.csv
    3ftx_preparation_summary.txt

Identifier policy
-----------------
FASTA headers and CSV `identifier` values are made identical and stable:
    - use `uniprot_id` when available
    - otherwise use `FTX_<id_new>`
This avoids ProtSpace/UniProt-style pipe parsing surprises and still lets default
annotations work for rows with real UniProt accessions.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

VALID_AA_RE = re.compile(r"[^A-Z*]")


def clean_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    seq = str(value).strip().upper()
    seq = re.sub(r"\s+", "", seq)
    seq = VALID_AA_RE.sub("", seq)
    return seq


def clean_identifier(value: object) -> str:
    s = "" if pd.isna(value) else str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    # Keep UniProt-style accessions intact, make fallback IDs FASTA-safe.
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.:-]", "_", s)
    return s


def wrap_fasta(seq: str, width: int = 80) -> Iterable[str]:
    for i in range(0, len(seq), width):
        yield seq[i : i + width]


def write_fasta(records: list[tuple[str, str, str]], path: Path) -> None:
    with path.open("w") as f:
        for ident, seq, desc in records:
            if desc:
                f.write(f">{ident} {desc}\n")
            else:
                f.write(f">{ident}\n")
            for line in wrap_fasta(seq):
                f.write(line + "\n")


def make_ids(df: pd.DataFrame) -> list[str]:
    ids: list[str] = []
    seen: dict[str, int] = {}
    for _, row in df.iterrows():
        uid = clean_identifier(row.get("uniprot_id", ""))
        if uid:
            base = uid
        else:
            id_new = clean_identifier(row.get("id_new", ""))
            if id_new:
                base = f"FTX_{id_new}"
            else:
                base = clean_identifier(row.get("identifier", "")) or f"FTX_ROW_{len(ids)+1}"
        seen[base] = seen.get(base, 0) + 1
        ident = base if seen[base] == 1 else f"{base}_{seen[base]}"
        ids.append(ident)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xlsx", required=True, type=Path, help="Input 3FTx Excel file")
    ap.add_argument("--sheet", default=0, help="Excel sheet name or index; default: first sheet")
    ap.add_argument("--out", required=True, type=Path, help="Output directory")
    args = ap.parse_args()

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df = pd.read_excel(args.xlsx, sheet_name=sheet)

    required = ["identifier", "mature_seq"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in Excel file: {missing}")
    if "full_seq" not in df.columns:
        df["full_seq"] = pd.NA
    if "uniprot_id" not in df.columns:
        df["uniprot_id"] = pd.NA
    if "id_new" not in df.columns:
        df["id_new"] = range(1, len(df) + 1)

    args.out.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["original_identifier"] = df["identifier"].astype(str)
    df["identifier"] = make_ids(df)
    df["protspace_id"] = df["identifier"]

    mature = df["mature_seq"].map(clean_sequence)
    full = df["full_seq"].map(clean_sequence)

    has_mature = mature.ne("")
    has_full = full.ne("")
    if not has_mature.all():
        bad = df.loc[~has_mature, ["identifier", "original_identifier"]].head(20)
        raise SystemExit("Some rows have no mature_seq. First examples:\n" + bad.to_string(index=False))

    full_or_mature_seq = full.where(has_full, mature)
    source = pd.Series("mature_seq", index=df.index)
    source.loc[has_full] = "full_seq"

    # Signal peptide is in the embedding exactly when the full sequence was used.
    sp_in_embedding_full = has_full.map(lambda x: "yes" if bool(x) else "no")
    sp_in_embedding_mature = pd.Series("no", index=df.index)

    signal_peptide_length = (full_or_mature_seq.str.len() - mature.str.len()).clip(lower=0)
    # If no full sequence was available, no SP length can be inferred from this table.
    signal_peptide_length = signal_peptide_length.where(has_full, 0)

    base_cols = ["identifier", "protspace_id", "original_identifier"] + [
        c for c in df.columns if c not in {"identifier", "protspace_id", "original_identifier"}
    ]
    ann_full = df[base_cols].copy()
    ann_mature = df[base_cols].copy()

    ann_full["sp_in_embedding"] = sp_in_embedding_full
    ann_full["signal_peptide"] = sp_in_embedding_full
    ann_full["sequence_source"] = source
    ann_full["target_sequence_length"] = full_or_mature_seq.str.len()
    ann_full["mature_sequence_length"] = mature.str.len()
    ann_full["full_sequence_length"] = full.str.len().where(has_full, 0)
    ann_full["signal_peptide_length_inferred"] = signal_peptide_length.astype(int)

    ann_mature["sp_in_embedding"] = sp_in_embedding_mature
    ann_mature["signal_peptide"] = "no"
    ann_mature["sequence_source"] = "mature_seq"
    ann_mature["target_sequence_length"] = mature.str.len()
    ann_mature["mature_sequence_length"] = mature.str.len()
    ann_mature["full_sequence_length"] = full.str.len().where(has_full, 0)
    ann_mature["signal_peptide_length_inferred"] = 0

    full_records = []
    mature_records = []
    for i, row in df.iterrows():
        ident = row["identifier"]
        desc = f"original={row['original_identifier']} source={source.loc[i]}"
        full_records.append((ident, full_or_mature_seq.loc[i], desc))
        mature_records.append((ident, mature.loc[i], f"original={row['original_identifier']} source=mature_seq"))

    fasta_full = args.out / "3ftx_full_or_mature.fasta"
    fasta_mature = args.out / "3ftx_mature_only.fasta"
    csv_full = args.out / "3ftx_annotations_full_or_mature.csv"
    csv_mature = args.out / "3ftx_annotations_mature_only.csv"
    summary = args.out / "3ftx_preparation_summary.txt"

    write_fasta(full_records, fasta_full)
    write_fasta(mature_records, fasta_mature)
    ann_full.to_csv(csv_full, index=False)
    ann_mature.to_csv(csv_mature, index=False)

    with summary.open("w") as f:
        f.write("3FTx final input preparation summary\n")
        f.write("=====================================\n")
        f.write(f"Input rows: {len(df)}\n")
        f.write(f"Rows with full_seq: {int(has_full.sum())}\n")
        f.write(f"Rows without full_seq, using mature_seq fallback: {int((~has_full).sum())}\n")
        f.write(f"Rows with mature_seq: {int(has_mature.sum())}\n")
        f.write(f"sp_in_embedding=yes in full_or_mature: {int((sp_in_embedding_full == 'yes').sum())}\n")
        f.write(f"sp_in_embedding=no in full_or_mature: {int((sp_in_embedding_full == 'no').sum())}\n")
        f.write(f"Wrote: {fasta_full}\n")
        f.write(f"Wrote: {fasta_mature}\n")
        f.write(f"Wrote: {csv_full}\n")
        f.write(f"Wrote: {csv_mature}\n")

    print(summary.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
