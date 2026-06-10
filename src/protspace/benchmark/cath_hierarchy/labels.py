"""CATH hierarchy label loading for the cath_s40 dataset.

The CATH S40 HDF5 file uses a grouped layout:
    outer key: ``cath|4_4_0|12asA00``  (domain name encoded in last pipe segment)
    inner key: ``4-330``               (residue range — used as the embedding)

This module:
1. Reads the HDF5 file extracting domain IDs (``12asA00``) and embeddings.
2. Downloads the official CATH domain list (``cath-domain-list.txt``) which maps
   every domain to its four-level classification (C.A.T.H).
3. Aligns embeddings with hierarchy labels so every array row has a consistent index.

The ``generate_group_metadata.py`` / ``protspace annotate`` pipeline does **not**
work for CATH because domain IDs (``12asA00``) are not UniProt accessions.
We therefore go directly to the CATH database.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import requests

logger = logging.getLogger(__name__)

CATH_DOMAIN_LIST_URL = (
    "https://download.cathdb.info/cath/releases/latest-release/"
    "cath-classification-data/cath-domain-list.txt"
)
_CACHE_DIR = Path.home() / ".cache" / "protspace" / "cath"
_CACHE_MAX_AGE_DAYS = 30


# ---------------------------------------------------------------------------
# Public data container
# ---------------------------------------------------------------------------


@dataclass
class CATHLabels:
    """Per-protein CATH hierarchy labels aligned with an embedding matrix.

    All arrays have length ``n`` and share the same row order.

    Attributes:
        identifiers: CATH domain IDs (e.g. ``"12asA00"``).
        embeddings:  Float32 embedding matrix, shape ``(n, dim)``.
        homology:    4-level code, e.g. ``"1.10.10.10"``.  Empty string = unknown.
        topology:    3-level code, e.g. ``"1.10.10"``.
        architecture: 2-level code, e.g. ``"1.10"``.
        cath_class:  1-level code, e.g. ``"1"``.
        valid_mask:  Boolean array; True where a CATH code was found.
    """

    identifiers: list[str]
    embeddings: np.ndarray
    homology: np.ndarray
    topology: np.ndarray
    architecture: np.ndarray
    cath_class: np.ndarray
    valid_mask: np.ndarray

    @property
    def n_valid(self) -> int:
        """Number of proteins with a known CATH classification."""
        return int(self.valid_mask.sum())


# ---------------------------------------------------------------------------
# CATH domain list download + parsing
# ---------------------------------------------------------------------------


def _fetch_cath_domain_list() -> dict[str, tuple[str, str, str, str]]:
    """Return ``{domain_id: (C, A, T, H)}`` from the CATH domain list.

    Downloads and caches the file for up to ``_CACHE_MAX_AGE_DAYS`` days.

    The CLF 2.0 format has space-separated columns:
        domain  C  A  T  H  S35  S60  S95  S100  S100_count  length  resolution
    """
    cache_dir = _CACHE_DIR
    cache_file = cache_dir / "cath_domain_list.json"
    ts_file = cache_dir / "cath_domain_list.timestamp"

    # --- Try cache first ---
    if cache_file.exists() and ts_file.exists():
        try:
            age_days = (time.time() - float(ts_file.read_text())) / 86_400
            if age_days < _CACHE_MAX_AGE_DAYS:
                logger.info(
                    "Loading CATH domain list from cache (%.1f days old)", age_days
                )
                raw = json.loads(cache_file.read_text())
                return {k: tuple(v) for k, v in raw.items()}  # type: ignore[return-value]
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            logger.warning("Cache read failed (%s); re-downloading", exc)

    # --- Download ---
    logger.info("Downloading CATH domain list from %s …", CATH_DOMAIN_LIST_URL)
    try:
        resp = requests.get(CATH_DOMAIN_LIST_URL, timeout=120)
        resp.raise_for_status()
        text = resp.text
    except Exception as exc:  # noqa: BLE001
        logger.warning("Download failed: %s", exc)
        if cache_file.exists():
            logger.info("Using stale cache as fallback")
            raw = json.loads(cache_file.read_text())
            return {k: tuple(v) for k, v in raw.items()}  # type: ignore[return-value]
        raise

    mapping = _parse_domain_list(text)

    # --- Persist cache ---
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(mapping))
        ts_file.write_text(str(time.time()))
        logger.info("Cached %d CATH domain entries", len(mapping))
    except OSError as exc:
        logger.warning("Could not write CATH cache: %s", exc)

    return mapping


def _parse_domain_list(text: str) -> dict[str, tuple[str, str, str, str]]:
    """Parse CLF 2.0 text → ``{domain_id: (C, A, T, H)}``."""
    mapping: dict[str, tuple[str, str, str, str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 5:
            continue
        domain = cols[0]  # e.g. "12asA00"
        c, a, t, h = cols[1], cols[2], cols[3], cols[4]
        mapping[domain] = (c, a, f"{c}.{a}.{t}", f"{c}.{a}.{t}.{h}")
    return mapping


# ---------------------------------------------------------------------------
# H5 loader — CATH-specific
# ---------------------------------------------------------------------------


def _load_cath_h5(h5_path: Path) -> tuple[list[str], np.ndarray]:
    """Load embeddings from the CATH HDF5 file.

    The file uses a **two-level group layout**:
        root → ``cath|4_4_0|12asA00`` (Group) → ``4-330`` (Dataset, shape (1024,))

    The existing ``load_h5`` utility uses the inner key (``4-330``) as the
    identifier, which loses the domain name.  This loader uses the outer key
    and extracts the domain ID from the last pipe-separated segment.

    Returns:
        identifiers: list of domain IDs (e.g. ``"12asA00"``)
        embeddings:  float32 array, shape ``(n, dim)``
    """
    identifiers: list[str] = []
    embeddings: list[np.ndarray] = []

    with h5py.File(h5_path, "r") as f:
        for outer_key, item in f.items():
            if not isinstance(item, h5py.Group):
                # Unexpected flat layout — skip
                logger.debug("Skipping non-group key: %s", outer_key)
                continue

            # Extract domain ID from key like "cath|4_4_0|12asA00"
            domain_id = outer_key.split("|")[-1] if "|" in outer_key else outer_key

            # Take the first (and usually only) sub-dataset
            sub_items = list(item.values())
            if not sub_items:
                logger.debug("Empty group: %s", outer_key)
                continue

            emb = np.array(sub_items[0]).flatten()
            if emb.ndim != 1:
                logger.warning(
                    "Unexpected embedding shape for %s: %s", domain_id, emb.shape
                )
                continue

            identifiers.append(domain_id)
            embeddings.append(emb)

    if not embeddings:
        raise ValueError(f"No embeddings loaded from {h5_path}")

    arr = np.stack(embeddings).astype(np.float32)

    # Upcast float16 → float32 (pLM outputs are often stored as float16)
    if arr.dtype == np.float16:
        arr = arr.astype(np.float32)

    logger.info(
        "Loaded %d embeddings (dim=%d) from %s", len(identifiers), arr.shape[1], h5_path
    )
    return identifiers, arr


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_cath_labels(h5_path: Path) -> CATHLabels:
    """Load embeddings and align CATH hierarchy labels.

    Downloads the official CATH domain list on first call (cached locally).

    Args:
        h5_path: Path to the CATH S40 HDF5 embedding file.

    Returns:
        :class:`CATHLabels` with embeddings and 4-level CATH codes aligned.
    """
    identifiers, embeddings = _load_cath_h5(h5_path)

    # Fetch CATH classifications
    domain_map = _fetch_cath_domain_list()

    # Align labels with embedding order
    cath_class_arr, arch_arr, topo_arr, homo_arr = [], [], [], []
    n_missing = 0

    for domain_id in identifiers:
        hit = domain_map.get(domain_id)
        if hit is None:
            cath_class_arr.append("")
            arch_arr.append("")
            topo_arr.append("")
            homo_arr.append("")
            n_missing += 1
        else:
            c_code, a_code, t_code, h_code = hit
            cath_class_arr.append(c_code)
            arch_arr.append(a_code)
            topo_arr.append(t_code)
            homo_arr.append(h_code)

    if n_missing:
        logger.warning(
            "%d / %d domains had no CATH classification (they may be obsolete or "
            "non-standard entries); they will be excluded from labelled metrics.",
            n_missing,
            len(identifiers),
        )

    valid_mask = np.array([bool(h) for h in homo_arr])

    logger.info(
        "%d / %d domains have full CATH hierarchy labels",
        valid_mask.sum(),
        len(identifiers),
    )

    return CATHLabels(
        identifiers=identifiers,
        embeddings=embeddings,
        homology=np.array(homo_arr),
        topology=np.array(topo_arr),
        architecture=np.array(arch_arr),
        cath_class=np.array(cath_class_arr),
        valid_mask=valid_mask,
    )
