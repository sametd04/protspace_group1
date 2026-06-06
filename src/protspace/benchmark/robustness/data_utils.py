"""Data utilities for robustness analysis.

Provides utilities for loading, aligning, and preparing data for group-based
robustness analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from protspace.data.loaders import load_h5


def extract_uniprot_id(id_str: str) -> str:
    """Extract UniProt accession from various ID formats.
    
    Handles multiple ID formats:
    - "sp|O43653|PSCA_HUMAN" -> "O43653"
    - "SP|O43653|Homo_sapiens" -> "O43653"
    - "A0A0U5AUY6" -> "A0A0U5AUY6" (already simple)
    
    Args:
        id_str: Identifier string
        
    Returns:
        Normalized UniProt accession
    """
    if '|' in id_str:
        parts = id_str.split('|')
        if len(parts) >= 2:
            return parts[1]  # Middle part is UniProt ID
    return id_str


def align_embeddings_metadata(
    embeddings: np.ndarray,
    embedding_ids: list[str],
    metadata_df: pd.DataFrame,
    identifier_column: str = "identifier",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Align embeddings and metadata using identifier matching.
    
    Normalizes IDs, finds intersection, and reorders both embeddings and metadata
    to match in a consistent sorted order.
    
    Args:
        embeddings: Embedding matrix (N, D)
        embedding_ids: List of embedding identifiers
        metadata_df: Metadata DataFrame with identifier column
        identifier_column: Name of identifier column in metadata
        
    Returns:
        Tuple of (aligned_embeddings, aligned_metadata)
    """
    # Normalize IDs
    embedding_ids_normalized = [extract_uniprot_id(id_) for id_ in embedding_ids]
    metadata_df = metadata_df.copy()
    metadata_df['identifier_normalized'] = metadata_df[identifier_column].apply(extract_uniprot_id)
    
    # Find intersection
    embedding_ids_set = set(embedding_ids_normalized)
    metadata_ids_set = set(metadata_df['identifier_normalized'].unique())
    common_ids = embedding_ids_set & metadata_ids_set
    
    print(f"  Embeddings: {len(embedding_ids_normalized)}")
    print(f"  Metadata: {len(metadata_df)}")
    print(f"  Common proteins (intersection): {len(common_ids)}")
    
    if len(common_ids) == 0:
        raise ValueError("No common identifiers found between embeddings and metadata!")
    
    # Filter metadata
    metadata_df = metadata_df[metadata_df['identifier_normalized'].isin(common_ids)].copy()
    
    # Filter and reorder embeddings
    id_to_emb_idx = {id_: idx for idx, id_ in enumerate(embedding_ids_normalized)}
    
    # Sort IDs for consistency
    common_ids_sorted = sorted(common_ids)
    emb_indices = [id_to_emb_idx[id_] for id_ in common_ids_sorted]
    embeddings = embeddings[emb_indices]
    
    # Reorder metadata to match
    id_to_meta_order = {id_: i for i, id_ in enumerate(common_ids_sorted)}
    metadata_df['sort_order'] = metadata_df['identifier_normalized'].map(id_to_meta_order)
    metadata_df = metadata_df.sort_values('sort_order').reset_index(drop=True)
    metadata_df = metadata_df.drop(columns=['sort_order', 'identifier_normalized'])
    
    print(f"  ✓ Aligned dataset size: {len(metadata_df)} proteins")
    
    return embeddings, metadata_df


def load_and_prepare_data(
    embedding_path: Path,
    metadata_path: Path,
    identifier_column: str = "identifier",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load embeddings and metadata, align them, and return.
    
    Complete pipeline for loading and preparing data for robustness analysis.
    
    Args:
        embedding_path: Path to HDF5 embedding file
        metadata_path: Path to metadata CSV file
        identifier_column: Name of identifier column in metadata
        
    Returns:
        Tuple of (aligned_embeddings, aligned_metadata)
    """
    print("Loading data...")
    
    # Load embeddings
    emb_set = load_h5([embedding_path])
    embeddings = emb_set.data
    embedding_ids = emb_set.headers
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    
    # Align
    embeddings, metadata_df = align_embeddings_metadata(
        embeddings, embedding_ids, metadata_df, identifier_column
    )
    
    return embeddings, metadata_df
