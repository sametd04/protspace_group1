#!/usr/bin/env python3
"""Generate group metadata CSV files for robustness analysis.

This script fetches UniProt and InterPro annotations for each dataset and creates
CSV files similar to data/3FTx/3FTx.csv with grouping information for robustness analysis.

Usage:
    uv run python scripts/generate_group_metadata.py --dataset toxprot
    uv run python scripts/generate_group_metadata.py --dataset pla2g2
    uv run python scripts/generate_group_metadata.py --dataset cath_s40
    uv run python scripts/generate_group_metadata.py --dataset swissprot_rr
    uv run python scripts/generate_group_metadata.py --all
"""

import argparse
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset configurations
DATASET_CONFIGS = {
    "toxprot": {
        "h5_path": "data/toxprot/prot_t5.h5",
        "fasta_path": "data/toxprot/toxprot_reviewed.fasta",
        "output_csv": "data/toxprot/toxprot.csv",
        "annotations": "protein_families,pfam,pfam_clan,keyword,family,genus",
    },
    "pla2g2": {
        "h5_path": "data/pla2g2/prot_t5.h5",
        "fasta_path": "data/pla2g2/pla2g2.fasta",
        "output_csv": "data/pla2g2/pla2g2.csv",
        "annotations": "protein_families,pfam,pfam_clan,keyword,family,genus",
    },
    "cath_s40": {
        "h5_path": "data/cath_s40/prot_t5.h5",
        "fasta_path": "data/cath_s40/cath_s40.fa",
        "output_csv": "data/cath_s40/cath_s40.csv",
        "annotations": "protein_families,pfam,pfam_clan,cath,superfamily,family",
    },
    "swissprot_rr": {
        "h5_path": "data/swissprot_rr/prot_t5.h5",
        "fasta_path": "data/swissprot_rr/swissprot_reviewed.fasta",
        "output_csv": "data/swissprot_rr/swissprot_rr.csv",
        "annotations": "protein_families,pfam,pfam_clan,keyword,family",
    },
}


def generate_metadata_for_dataset(dataset_name: str) -> None:
    """Generate metadata CSV for a specific dataset using protspace annotate."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    h5_path = PROJECT_ROOT / config["h5_path"]
    fasta_path = PROJECT_ROOT / config["fasta_path"]
    output_csv = PROJECT_ROOT / config["output_csv"]

    if not h5_path.exists():
        raise FileNotFoundError(
            f"H5 file not found for {dataset_name}: {h5_path}\n"
            f"Please run: uv run python scripts/download_project_datasets.py --datasets {dataset_name}"
        )

    if not fasta_path.exists():
        raise FileNotFoundError(
            f"FASTA file not found for {dataset_name}: {fasta_path}\n"
            f"Please run: uv run python scripts/download_project_datasets.py --datasets {dataset_name}"
        )

    logger.info("=" * 80)
    logger.info("Generating metadata for dataset: %s", dataset_name)
    logger.info("H5: %s", h5_path.relative_to(PROJECT_ROOT))
    logger.info("FASTA: %s", fasta_path.relative_to(PROJECT_ROOT))
    logger.info("Output CSV: %s", output_csv.relative_to(PROJECT_ROOT))
    logger.info("Annotations: %s", config["annotations"])
    logger.info("=" * 80)

    # Create temp directory for annotations
    tmp_dir = PROJECT_ROOT / "tmp" / f"annotate_{dataset_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Run protspace annotate to fetch annotations
    # Note: Use FASTA file as input to get proper UniProt accessions from headers
    # and sequences for InterPro annotations
    cmd = [
        "uv",
        "run",
        "protspace",
        "annotate",
        "-i",
        str(fasta_path),
        "-a",
        config["annotations"],
        "-o",
        str(tmp_dir / "annotations.parquet"),
        "-v",
    ]

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    # Convert parquet to CSV
    logger.info("Converting parquet to CSV...")
    import pandas as pd

    parquet_path = tmp_dir / "annotations.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Annotation parquet not created: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    
    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    logger.info("Metadata CSV saved: %s (%d proteins)", output_csv, len(df))
    logger.info("Columns: %s", list(df.columns))
    
    # Show preview
    logger.info("\nPreview (first 5 rows):")
    print(df.head())
    
    # Show grouping column value counts
    logger.info("\nGrouping column summaries:")
    for col in ["protein_families", "pfam_clan", "family", "keyword"]:
        if col in df.columns:
            non_empty = df[col].notna() & (df[col] != "") & (df[col] != "<N/A>")
            count = non_empty.sum()
            unique = df[col][non_empty].nunique() if count > 0 else 0
            logger.info("  %s: %d proteins with values, %d unique", col, count, unique)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate group metadata CSV files for robustness analysis"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to generate metadata for",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate metadata for all datasets",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Must specify either --dataset or --all")

    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    datasets = list(DATASET_CONFIGS.keys()) if args.all else [args.dataset]

    for dataset in datasets:
        try:
            generate_metadata_for_dataset(dataset)
        except Exception as e:
            logger.error("Failed to generate metadata for %s: %s", dataset, e)
            if not args.all:
                raise
            logger.warning("Continuing with next dataset...")

    logger.info("\nDone! Generated metadata CSV files.")


if __name__ == "__main__":
    main()
