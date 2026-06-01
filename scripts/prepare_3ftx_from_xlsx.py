#!/usr/bin/env python3
"""Prepare the 3FTx Excel dataset for ProtSpace/ρPCA experiments.

This version additionally creates a full-only FASTA for paired-delta ρPCA.

Outputs:
  1. sequences_full_or_mature.fasta
     Target embedding: full_seq if available, otherwise mature_seq.

  2. sequences_mature_only.fasta
     Mature-only positive-control embedding.

  3. sequences_full_only.fasta
     Full-precursor embeddings for rows with a usable full_seq. This is used
     together with the mature-only embeddings to build signed paired deltas
     E(full) - E(mature).

  4. annotations_3ftx_ppca.csv
     Custom ProtSpace annotations, including signal-peptide fields for ρPCA.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


AA_RE = re.compile(r"[A-Z]")
UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:"
    r"[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]|"
    r"A0A[A-Z0-9]{6,10}"
    r")$"
)


def clean_seq(value: object) -> str:
    """Return an uppercase amino-acid-like sequence string."""
    return "".join(AA_RE.findall(str(value).upper()))


def clean_text(value: object) -> str:
    return str(value).strip()


def is_uniprot_accession(value: str) -> bool:
    return bool(UNIPROT_ACCESSION_RE.match(value.strip().upper()))


def parse_accession_from_original_identifier(value: object) -> str:
    """Extract accession from values like SP|P01385|Acanthophis_antarcticus."""
    text = clean_text(value)
    parts = text.split("|")
    if len(parts) >= 2:
        candidate = parts[1].strip().upper()
        if is_uniprot_accession(candidate):
            return candidate
    candidate = text.strip().upper()
    if is_uniprot_accession(candidate):
        return candidate
    return ""


def make_fallback_id(row: pd.Series, row_index: int) -> str:
    """Create a stable non-UniProt fallback ID for rows without accessions."""
    for col in ("id_new", "id", "protein_id"):
        if col in row and clean_text(row[col]):
            raw = clean_text(row[col])
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
            return f"3FTx_{safe}"
    return f"3FTx_row_{row_index + 1:04d}"


def choose_protspace_id(row: pd.Series, row_index: int) -> str:
    """Choose the ID used by both FASTA headers and annotation CSV."""
    if "uniprot_id" in row and clean_text(row["uniprot_id"]):
        candidate = clean_text(row["uniprot_id"]).upper()
        if is_uniprot_accession(candidate):
            return candidate

    candidate = parse_accession_from_original_identifier(row.get("identifier", ""))
    if candidate:
        return candidate

    return make_fallback_id(row, row_index)


def deduplicate_ids(ids: list[str]) -> list[str]:
    """Ensure FASTA/annotation identifiers are unique."""
    counts: dict[str, int] = {}
    out: list[str] = []
    for identifier in ids:
        n = counts.get(identifier, 0)
        counts[identifier] = n + 1
        if n == 0:
            out.append(identifier)
        else:
            out.append(f"{identifier}_dup{n + 1}")
    return out


def write_fasta(df: pd.DataFrame, path: Path, seq_col: str) -> int:
    n = 0
    with path.open("w") as handle:
        for _, row in df.iterrows():
            seq = row[seq_col]
            if not seq:
                continue
            identifier = row["identifier"]
            handle.write(f">{identifier}\n")
            for i in range(0, len(seq), 80):
                handle.write(seq[i : i + 80] + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=None,
        help=(
            "Path to 3FTx_data.xlsx. Default: data/3Ftx/3FTx_data.xlsx "
            "or data/3FTx/3FTx_data.xlsx."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: the Excel file's parent directory.",
    )
    args = parser.parse_args()

    if args.xlsx is None:
        candidates = [
            Path("data/3Ftx/3FTx_data.xlsx"),
            Path("data/3FTx/3FTx_data.xlsx"),
        ]
        xlsx_path = next((p for p in candidates if p.exists()), candidates[0])
    else:
        xlsx_path = args.xlsx

    if not xlsx_path.exists():
        raise FileNotFoundError(f"Could not find Excel file: {xlsx_path}")

    out_dir = args.out_dir if args.out_dir is not None else xlsx_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    out_full_or_mature = out_dir / "sequences_full_or_mature.fasta"
    out_mature_only = out_dir / "sequences_mature_only.fasta"
    out_full_only = out_dir / "sequences_full_only.fasta"
    out_annotations = out_dir / "annotations_3ftx_ppca.csv"

    df = pd.read_excel(xlsx_path, dtype=str).fillna("")

    required = ["identifier", "full_seq", "mature_seq"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in {xlsx_path}: {missing}")

    df["original_identifier"] = df["identifier"].map(clean_text)

    protspace_ids = [choose_protspace_id(row, i) for i, row in df.iterrows()]
    df["identifier"] = deduplicate_ids(protspace_ids)
    df["protspace_id"] = df["identifier"]
    df["identifier_is_uniprot_accession"] = df["identifier"].map(
        lambda x: "yes" if is_uniprot_accession(str(x).split("_dup")[0]) else "no"
    )

    df["full_seq_clean"] = df["full_seq"].map(clean_seq)
    df["mature_seq_clean"] = df["mature_seq"].map(clean_seq)

    df["target_sequence"] = df.apply(
        lambda row: row["full_seq_clean"]
        if row["full_seq_clean"]
        else row["mature_seq_clean"],
        axis=1,
    )
    df["sequence_source"] = df.apply(
        lambda row: "full_seq" if row["full_seq_clean"] else "mature_seq",
        axis=1,
    )

    sp_sequences: list[str] = []
    sp_starts: list[object] = []
    sp_ends: list[object] = []
    sp_status: list[str] = []
    sp_notes: list[str] = []

    for _, row in df.iterrows():
        full = row["full_seq_clean"]
        mature = row["mature_seq_clean"]

        if full and mature:
            pos = full.find(mature)
            if pos > 0:
                sp_sequences.append(full[:pos])
                sp_starts.append(0)
                sp_ends.append(pos)
                sp_status.append("yes")
                sp_notes.append("full_seq_prefix_before_mature_seq")
            elif pos == 0:
                sp_sequences.append("")
                sp_starts.append("")
                sp_ends.append("")
                sp_status.append("no")
                sp_notes.append("mature_seq_starts_at_position_0")
            else:
                sp_sequences.append("")
                sp_starts.append("")
                sp_ends.append("")
                sp_status.append("unknown")
                sp_notes.append("mature_seq_not_found_in_full_seq")
        else:
            sp_sequences.append("")
            sp_starts.append("")
            sp_ends.append("")
            sp_status.append("no")
            sp_notes.append("no_full_seq_available_target_uses_mature_seq")

    df["signal_peptide_sequence"] = sp_sequences
    df["sp_start"] = sp_starts
    df["sp_end"] = sp_ends
    df["sp_in_embedding"] = sp_status
    df["signal_peptide"] = sp_status
    df["sp_derivation_note"] = sp_notes
    df["target_sequence_length"] = df["target_sequence"].str.len()
    df["mature_sequence_length"] = df["mature_seq_clean"].str.len()
    df["full_sequence_length"] = df["full_seq_clean"].str.len()
    df["signal_peptide_length"] = df["signal_peptide_sequence"].str.len()

    # Paired-delta full FASTA: only rows where full_seq exists. The actual
    # paired_delta mode can further restrict to sp_in_embedding=yes.
    df_full_only = df[df["full_seq_clean"].str.len() > 0].copy()

    n_target = write_fasta(df, out_full_or_mature, "target_sequence")
    n_mature = write_fasta(df, out_mature_only, "mature_seq_clean")
    n_full = write_fasta(df_full_only, out_full_only, "full_seq_clean")

    preferred_cols = [
        "identifier",
        "protspace_id",
        "original_identifier",
        "uniprot_id",
        "identifier_is_uniprot_accession",
        "sp_in_embedding",
        "signal_peptide",
        "signal_peptide_sequence",
        "signal_peptide_length",
        "sp_start",
        "sp_end",
        "sp_derivation_note",
        "sequence_source",
        "target_sequence_length",
        "mature_sequence_length",
        "full_sequence_length",
        "major_group",
        "sub_group",
        "cysteine_group",
        "number_cysteines",
        "evolutionary_order",
        "taxon_of_interest",
        "family",
        "genus",
        "species",
        "taxon_id",
        "data_origin",
        "db",
        "name",
        "name_infered_activity",
        "activity",
        "receptor",
        "membran_prediction",
        "representative",
        "Basal",
        "Dimeric",
        "Derived",
        "Short-chain",
        "Long-chain",
        "Non-standard",
        "S-type",
        "P-type",
        "Density-based clustering (ε=0.55)",
        "Density-based clustering (ε=0.59)",
        "Kmeans (n=7)",
    ]
    annotation_cols = [col for col in preferred_cols if col in df.columns]

    annotations = df[annotation_cols].copy()
    annotations.to_csv(out_annotations, index=False)

    n_uniprot = int((df["identifier_is_uniprot_accession"] == "yes").sum())
    n_non_uniprot = int((df["identifier_is_uniprot_accession"] == "no").sum())

    print(f"Wrote {n_target:,} target full-or-mature sequences to {out_full_or_mature}")
    print(f"Wrote {n_mature:,} mature-only control sequences to {out_mature_only}")
    print(f"Wrote {n_full:,} full-only paired-delta sequences to {out_full_only}")
    print(f"Wrote annotations to {out_annotations}")
    print()
    print("Identifier summary:")
    print(f"  UniProt-accession identifiers: {n_uniprot:,}")
    print(f"  fallback non-UniProt identifiers: {n_non_uniprot:,}")
    print()
    print("Sequence-source summary:")
    print(df["sequence_source"].value_counts(dropna=False).to_string())
    print()
    print("SP-in-target-embedding summary:")
    print(df["sp_in_embedding"].value_counts(dropna=False).to_string())
    print()
    print("SP length summary for derived background:")
    yes = df.loc[df["sp_in_embedding"] == "yes", "signal_peptide_length"]
    print(yes.describe().to_string())
    print()
    print("First five FASTA/annotation identifiers:")
    print(df[["identifier", "original_identifier", "uniprot_id"]].head().to_string(index=False))


if __name__ == "__main__":
    main()