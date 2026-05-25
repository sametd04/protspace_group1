# Functional Groups README (Swiss-Prot / UniProtKB)

This document defines practical functional group sets and shows how to test them on DR outputs.

Important:
- There is no single finite list of “all functional groups” in Swiss-Prot.
- Functional groups are derived from annotation systems (EC, GO, Pfam, InterPro, Keywords, pathways, etc.).
- For DR evaluation, you should define one or more label schemes explicitly.

---

## 1. Functional Group Families You Can Use

## 1.1 EC-based groups (enzyme-centric)
- Source: Enzyme Commission (EC) numbers.
- Label options:
  - EC top-level class (1..7): coarse and robust.
  - EC second level (`x.y`): more specific.
  - Full EC (`x.y.z.w`): very specific, often sparse.

Good for:
- clean, biologically interpretable function classes.

---

## 1.2 GO-based groups
- Source: Gene Ontology.
- Sub-ontologies:
  - Molecular Function (GO:MF)
  - Biological Process (GO:BP)
  - Cellular Component (GO:CC)

Label options:
- MF slim categories (recommended first).
- BP slim categories (harder, broader).
- Top GO term per protein (if you need one-label-per-protein).

Good for:
- broad functional biology beyond enzymes.

---

## 1.3 Domain/family-based groups
- Source: Pfam, InterPro, PROSITE.
- Label options:
  - dominant Pfam family per protein
  - InterPro family ID
  - domain architecture class (if multi-domain modeling is available)

Good for:
- sequence-function relations via conserved domains.

---

## 1.4 Swiss-Prot keyword groups
- Source: UniProt Keywords.
- Examples:
  - `Kinase`
  - `Transmembrane`
  - `DNA-binding`
  - `Signal peptide`
  - `ATP-binding`

Good for:
- quick coarse function grouping without heavy ontology parsing.

---

## 1.5 Pathway groups
- Source: UniProt pathway annotation / Reactome / KEGG links.
- Label options:
  - pathway family
  - curated high-level pathway bucket

Good for:
- system-level biological grouping.

---

## 1.6 Subcellular localization groups (semi-functional)
- Source: Subcellular location annotation.
- Examples:
  - nucleus
  - cytoplasm
  - membrane
  - mitochondrion
  - extracellular

Good for:
- complementary biological axis; often correlated with function.

---

## 2. Recommended Label Schemes for DR Benchmarking

Use at least two of these:

1. `EC_top_level` (single-label multiclass)
2. `GO_MF_slim` (single-label or multi-label reduced)
3. `Pfam_family_top` (single-label)

And keep confounders:
- `taxonomy_group` (e.g., phylum/class)
- `seq_length` (numeric)

---

## 3. How to Check Functional Group Signal

For each DR projection (`pca`, `tsne`, `umap`, `ppca`):

1. Function predictability:
- train classifier on 2D/3D coordinates
- report macro-F1 (cross-validated)
- higher = better functional separability

2. Functional cluster quality:
- silhouette score on function labels
- higher = tighter/separated function groups

3. kNN functional purity:
- for each point, fraction of neighbors sharing function label
- higher = better local function consistency

4. Confounder leakage (for your main hypothesis):
- `R²(seq_length ~ DR coordinates)` lower = better
- taxonomy predictability lower = better

Interpretation target:
- good contrastive DR should reduce confounder dominance while preserving or improving function metrics.

---

## 4. Example Evaluation Table (What to Report)

Per method:
- `length_r2`
- `taxonomy_macro_f1`
- `function_macro_f1` (for each scheme: EC, GO, Pfam)
- `function_silhouette`
- `knn_function_purity`

Example conclusion format:
- “`ppca` reduced length R² and taxonomy F1 versus PCA/UMAP/t-SNE, while increasing EC and Pfam macro-F1.”

---

## 5. How to Build Label Files

Create a TSV with one row per protein:

`protein_id  seq_length  taxonomy_group  ec_top  go_mf_slim  pfam_top`

Minimal required columns for your question:
- `protein_id`
- `seq_length`
- `taxonomy_group`
- at least one function column (`ec_top` or `go_mf_slim` or `pfam_top`)

If a protein has multiple labels:
- start with a deterministic single-label rule (e.g., highest-confidence or most-specific curated label),
- then extend to multi-label evaluation if needed.

---

## 6. Practical Pros/Cons by Grouping Type

EC:
- Pros: clear biochemical meaning.
- Cons: missing for many non-enzymes.

GO-MF:
- Pros: broad coverage.
- Cons: hierarchical, noisy granularity unless slimmed.

Pfam/InterPro:
- Pros: strong sequence-level signal.
- Cons: can reflect homology more than high-level biological role.

Keywords:
- Pros: easy to use.
- Cons: coarse and uneven specificity.

Pathway:
- Pros: biological systems view.
- Cons: sparse/incomplete depending on organism coverage.

---

## 7. Suggested Start (Fast, Robust)

Start with:
1. `ec_top` (if available)
2. `pfam_top`
3. `taxonomy_group` and `seq_length` as confounders

Then add:
4. `go_mf_slim`

This gives a robust first answer to:
- whether contrastive DR suppresses taxonomy/length axes
- whether function-relevant grouping becomes clearer.

