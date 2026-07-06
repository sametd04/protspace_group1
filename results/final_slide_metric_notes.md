# Final slide metric notes

The final slide plots are clean scatter plots without embedded metric boxes. Metrics are written to CSV and standalone metric-card PNGs so they can be shown only if useful.

Metrics used:
- Silhouette: direct annotation-label separation in the 2D plot. For nuisance labels, lower is better.
- Label variance η²: fraction of 2D variance explained by the annotation label. For nuisance labels, lower is better.
- NMI: agreement between KMeans clusters in the 2D plot and the annotation labels. For nuisance labels, lower is better.

Dropped from the main slide workflow:
- Davies-Bouldin, because it overlaps with Silhouette and can explode for highly overlapping groups.
- Calinski-Harabasz, because it overlaps with other compactness/separation measures and is unbounded.
- ARI, because it overlaps with NMI and is less readable for imbalanced labels.

3FTx main slide recommendation:
Show one SP-in-embedding problem plot (PCA full-or-mature), then compare mature-only PCA, manual paired-delta ρPCA, and unified annotation ρPCA colored by major group. The manual paired-delta background is the 3FTx-specific best-case control; the unified annotation background is the generalizable ProtSpace method.

ToxProt main slide recommendation:
Show PCA and ρPCA colored by length and taxonomy order. The ρPCA background is built from length + order.
