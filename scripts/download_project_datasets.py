#!/usr/bin/env python3
"""Download project datasets (3FTx, ToxProt, CATH S40, SwissProt).

Usage examples:
    uv run python scripts/download_project_datasets.py
    uv run python scripts/download_project_datasets.py --datasets toxprot cath_s40
    uv run python scripts/download_project_datasets.py --toxprot-max 2000 --cath-max 5000
    uv run python scripts/download_project_datasets.py --swissprot-max 50000
    uv run python scripts/download_project_datasets.py --full
    uv run python scripts/download_project_datasets.py --swissprot-identity 0.4
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
CATH_S40_FASTA_URL = (
    "https://download.cathdb.info/cath/releases/latest-release/"
    "non-redundant-data-sets/cath-dataset-nonredundant-S40.fa"
)
CATH_S40_LIST_URL = (
    "https://download.cathdb.info/cath/releases/latest-release/"
    "non-redundant-data-sets/cath-dataset-nonredundant-S40.list"
)
THREE_FTX_CSV_URL = (
    "https://raw.githubusercontent.com/tsenoner/protspace/main/data/3FTx/3FTx.csv"
)
ALL_DATASETS = ("3ftx", "toxprot", "cath_s40", "swissprot_rr")
DEFAULT_SMALL_MAX = {
    "threeftx_max": 300,
    "toxprot_max": 1500,
    "cath_max": 3000,
    "swissprot_max": 10000,
}


def _parse_next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return match.group(1) if match else None


def _download_file(url: str, output_path: Path, timeout: int = 120) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, output_path)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _count_fasta_sequences(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def _trim_fasta(path: Path, max_sequences: int) -> int:
    if max_sequences <= 0:
        raise ValueError("max_sequences must be > 0")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    written = 0
    keep_entry = False
    with path.open() as src, tmp_path.open("w") as dst:
        for line in src:
            if line.startswith(">"):
                if written >= max_sequences:
                    break
                written += 1
                keep_entry = True
            if keep_entry:
                dst.write(line)
    tmp_path.replace(path)
    return written


def _download_uniprot_fasta(
    query: str,
    output_path: Path,
    *,
    max_sequences: int | None,
    batch_size: int = 500,
    timeout: int = 120,
    append: bool = False,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    total_written = 0
    next_url: str | None = UNIPROT_SEARCH_URL
    first_call = True
    with output_path.open(mode) as out:
        while next_url:
            params = None
            if first_call:
                params = {"query": query, "format": "fasta", "size": batch_size}
            with requests.get(
                next_url, params=params, stream=True, timeout=timeout
            ) as response:
                response.raise_for_status()
                keep_entry = True
                for raw_line in response.iter_lines(decode_unicode=True):
                    line = raw_line or ""
                    if line.startswith(">"):
                        if max_sequences is not None and total_written >= max_sequences:
                            logger.info(
                                "Reached max_sequences=%s for query '%s'",
                                max_sequences,
                                query,
                            )
                            return total_written
                        total_written += 1
                        keep_entry = True
                    if keep_entry:
                        out.write(line + "\n")
                next_url = _parse_next_link(response.headers.get("Link"))
                first_call = False
            if max_sequences is not None and total_written >= max_sequences:
                break
    return total_written


def _read_3ftx_accessions(csv_path: Path) -> list[str]:
    accessions: set[str] = set()
    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            identifier = (row.get("identifier") or "").strip()
            if not identifier:
                continue
            parts = identifier.split("|")
            if len(parts) < 2:
                continue
            db, accession = parts[0].upper(), parts[1]
            if db == "SP" and accession:
                accessions.add(accession)
    return sorted(accessions)


def _chunked(values: Iterable[str], size: int) -> Iterable[list[str]]:
    chunk: list[str] = []
    for value in values:
        chunk.append(value)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _download_3ftx(
    out_dir: Path, *, max_sequences: int | None, timeout: int, batch_size: int
) -> None:
    dataset_dir = out_dir / "3ftx"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dataset_dir / "3FTx.csv"
    local_csv = Path("data/3FTx/3FTx.csv")
    if local_csv.exists():
        shutil.copy2(local_csv, csv_path)
        logger.info("Copied local 3FTx CSV -> %s", csv_path)
    else:
        _download_file(THREE_FTX_CSV_URL, csv_path, timeout=timeout)

    accessions = _read_3ftx_accessions(csv_path)
    if max_sequences is not None:
        accessions = accessions[:max_sequences]
    logger.info("3FTx: downloading %s Swiss-Prot sequences", len(accessions))

    output_fasta = dataset_dir / "3ftx_reviewed.fasta"
    if output_fasta.exists():
        output_fasta.unlink()

    total = 0
    for i, batch in enumerate(_chunked(accessions, 100), start=1):
        terms = " OR ".join(f"(accession:{acc})" for acc in batch)
        query = f"({terms})"
        written = _download_uniprot_fasta(
            query,
            output_fasta,
            max_sequences=None,
            batch_size=batch_size,
            timeout=timeout,
            append=i > 1,
        )
        total += written
    logger.info("3FTx FASTA saved: %s (%s sequences)", output_fasta, total)


def _download_toxprot(
    out_dir: Path, *, max_sequences: int | None, timeout: int, batch_size: int
) -> None:
    dataset_dir = out_dir / "toxprot"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    query = "(keyword:KW-0800) AND (reviewed:true)"
    output_fasta = dataset_dir / "toxprot_reviewed.fasta"
    count = _download_uniprot_fasta(
        query,
        output_fasta,
        max_sequences=max_sequences,
        batch_size=batch_size,
        timeout=timeout,
    )
    logger.info("ToxProt FASTA saved: %s (%s sequences)", output_fasta, count)


def _download_cath_s40(
    out_dir: Path, *, max_sequences: int | None, timeout: int
) -> None:
    dataset_dir = out_dir / "cath_s40"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = dataset_dir / "cath_s40.fa"
    list_path = dataset_dir / "cath_s40.list"

    _download_file(CATH_S40_FASTA_URL, fasta_path, timeout=timeout)
    _download_file(CATH_S40_LIST_URL, list_path, timeout=timeout)

    if max_sequences is not None:
        trimmed = _trim_fasta(fasta_path, max_sequences)
        logger.info("CATH S40 FASTA trimmed to %s sequences", trimmed)
    else:
        logger.info("CATH S40 FASTA sequences: %s", _count_fasta_sequences(fasta_path))


def _run_mmseqs_redundancy_reduction(
    input_fasta: Path, output_fasta: Path, identity: float, tmp_dir: Path
) -> None:
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "mmseqs not found in PATH. Install mmseqs2 or skip --swissprot-identity."
        )
    tmp_dir.mkdir(parents=True, exist_ok=True)
    prefix = tmp_dir / "swissprot_rr"
    cmd = [
        "mmseqs",
        "easy-cluster",
        str(input_fasta),
        str(prefix),
        str(tmp_dir / "work"),
        "--min-seq-id",
        str(identity),
        "-c",
        "0.8",
        "--cov-mode",
        "1",
    ]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    rep_fasta = Path(f"{prefix}_rep_seq.fasta")
    if not rep_fasta.exists():
        raise RuntimeError("mmseqs did not create representative FASTA output")
    shutil.copy2(rep_fasta, output_fasta)


def _download_swissprot(
    out_dir: Path,
    *,
    max_sequences: int | None,
    timeout: int,
    batch_size: int,
    identity: float | None,
) -> None:
    dataset_dir = out_dir / "swissprot"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    full_fasta = dataset_dir / "swissprot_reviewed.fasta"

    query = "reviewed:true"
    count = _download_uniprot_fasta(
        query,
        full_fasta,
        max_sequences=max_sequences,
        batch_size=batch_size,
        timeout=timeout,
    )
    logger.info("SwissProt FASTA saved: %s (%s sequences)", full_fasta, count)

    if identity is not None:
        rr_fasta = dataset_dir / "swissprot_rr.fasta"
        _run_mmseqs_redundancy_reduction(
            full_fasta, rr_fasta, identity=identity, tmp_dir=dataset_dir / "tmp_mmseqs"
        )
        rr_count = _count_fasta_sequences(rr_fasta)
        logger.info(
            "SwissProt redundancy-reduced FASTA saved: %s (%s sequences)",
            rr_fasta,
            rr_count,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download project datasets (3FTx, ToxProt, CATH S40, SwissProt)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=[*ALL_DATASETS, "all"],
        help="Datasets to download (default: all).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data/project_datasets"),
        help="Output directory (default: data/project_datasets).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="UniProt page size (default: 500).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout seconds (default: 120).",
    )
    parser.add_argument(
        "--3ftx-max",
        dest="threeftx_max",
        type=int,
        default=DEFAULT_SMALL_MAX["threeftx_max"],
        help=(
            "Max Swiss-Prot sequences to fetch for 3FTx accessions "
            f"(default: {DEFAULT_SMALL_MAX['threeftx_max']})."
        ),
    )
    parser.add_argument(
        "--toxprot-max",
        type=int,
        default=DEFAULT_SMALL_MAX["toxprot_max"],
        help=f"Max ToxProt sequences (default: {DEFAULT_SMALL_MAX['toxprot_max']}).",
    )
    parser.add_argument(
        "--cath-max",
        type=int,
        default=DEFAULT_SMALL_MAX["cath_max"],
        help=f"Max CATH S40 sequences (default: {DEFAULT_SMALL_MAX['cath_max']}).",
    )
    parser.add_argument(
        "--swissprot-max",
        type=int,
        default=DEFAULT_SMALL_MAX["swissprot_max"],
        help=(
            "Max SwissProt reviewed sequences "
            f"(default: {DEFAULT_SMALL_MAX['swissprot_max']})."
        ),
    )
    parser.add_argument(
        "--swissprot-identity",
        type=float,
        default=None,
        help=(
            "Optional MMseqs2 redundancy reduction identity threshold "
            "(e.g. 0.4 for ~S40-like filtering)."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Disable subset limits and download full datasets.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    if args.full:
        args.threeftx_max = None
        args.toxprot_max = None
        args.cath_max = None
        args.swissprot_max = None

    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    selected = set(ALL_DATASETS if "all" in args.datasets else args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if "3ftx" in selected:
        _download_3ftx(
            args.output_dir,
            max_sequences=args.threeftx_max,
            timeout=args.timeout,
            batch_size=args.batch_size,
        )
    if "toxprot" in selected:
        _download_toxprot(
            args.output_dir,
            max_sequences=args.toxprot_max,
            timeout=args.timeout,
            batch_size=args.batch_size,
        )
    if "cath_s40" in selected:
        _download_cath_s40(
            args.output_dir,
            max_sequences=args.cath_max,
            timeout=args.timeout,
        )
    if "swissprot_rr" in selected:
        _download_swissprot(
            args.output_dir,
            max_sequences=args.swissprot_max,
            timeout=args.timeout,
            batch_size=args.batch_size,
            identity=args.swissprot_identity,
        )

    logger.warning("Done. Files written to: %s", args.output_dir)


if __name__ == "__main__":
    main()
