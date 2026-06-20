# Contrastive DR Evaluation README (ρPCA vs PCA/UMAP/t-SNE)

This guide shows how to answer:

`On the redundancy-reduced SwissProt set: can contrastive DR remove sequence length and taxonomy as dominant axes, revealing functional groupings instead?`

It covers:
- how to run `ppca`, `pca`, `umap`, `tsne`
- how to build target/background datasets
- how to evaluate confounder removal and functional structure
- pros/cons of each method

---

## 1. Core Concepts

- `Target` (foreground): dataset you want to understand.
- `Background`: dataset that contains unwanted variation you want to suppress in contrastive DR.
- In this project, contrastive DR is `ppca` (ρPCA).
- `pca`, `umap`, `tsne` are not contrastive (single-dataset DR).

Practical implication:
- For your research question, `ppca` should be run with a biologically meaningful background via `--ppca-background`.

---

## 2. What Are “Functional Groupings”?

Functional groupings are groups of proteins that share biological function, for example:
- EC class
- GO terms (BP/MF)
- InterPro/Pfam families
- curated functional categories from SwissProt annotations

For evaluation you need one function label per protein (or a reduced multiclass label set).

---

## 3. Recommended Dataset Design

### 3.1 Target (required)
- Redundancy-reduced SwissProt embedding set (your main analysis set).

### 3.2 Background (for `ppca`)
Use one of these strategies:
1. Broad SwissProt background with diverse taxonomy and length distribution.
2. Taxonomy-matched but function-broad background.
3. Length-matched but taxonomy-broad background.

This lets you test what confounder the contrastive setup removes best.

---

## 4. Run the 4 DR Methods

From repo root:

```bash
cd /home/ac/Documents/Prot/protspace_group1
```

Single run producing all four projections:

```bash
protspace project \
  -i data/target_reduced_swissprot.h5 \
  -m pca2 \
  -m tsne2:perplexity=30;learning_rate=200 \
  -m umap2:n_neighbors=25;min_dist=0.1 \
  -m ppca2:background_path=data/background_swissprot.h5;regularization_mu=1e-3;background_strategy=external \
  -o output/dr_compare
```

Notes:
- `background_path=...` in the `ppca2` method spec is supported by the current branch.
- You can also pass global `--ppca-background` instead of inline `background_path`.
- For `ppca`, keep `regularization_mu > 0` in high-dimensional embeddings.

---

## 5. Evaluation Plan for Your Question

You want to test two things simultaneously:
1. Are sequence length and taxonomy less dominant in the embedding?
2. Are function-based groups more visible/separable?

Use the same protein set across all methods, then compute:

### 5.1 Confounder dominance metrics
- `R²(length ~ DR1 + DR2)` (linear regression): lower is better.
- Taxonomy predictability from DR coordinates (e.g., logistic regression CV accuracy/F1): lower means taxonomy axis is reduced.

### 5.2 Functional structure metrics
- Function-label predictability from DR coordinates (logistic regression CV accuracy/F1): higher is better.
- Silhouette score using function labels: higher is better.
- Optional: kNN purity by function label in DR space.

### 5.3 Decision logic
- Best method for your hypothesis has:
  - low length/taxonomy predictability
  - high function separability/predictability

---

## 6. Minimal Evaluation Script (Template)

Save as `scripts/eval_confounders_vs_function.py`:

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, silhouette_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder


def load_projection_tables(out_dir: Path):
    meta = pd.read_parquet(out_dir / "projections_metadata.parquet")
    data = pd.read_parquet(out_dir / "projections_data.parquet")
    return meta, data


def extract_projection_xy(data_df: pd.DataFrame, projection_name: str):
    sub = data_df[data_df["projection_name"] == projection_name].copy()
    # Assumes columns x/y exist in your projection table; adapt names if needed.
    X = sub[["x", "y"]].to_numpy()
    ids = sub["protein_id"].astype(str).to_numpy()
    return ids, X


def cv_macro_f1(X, y, seed=42):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = LogisticRegression(max_iter=300, multi_class="auto")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, X, y_enc, cv=cv)
    return f1_score(y_enc, pred, average="macro")


def main():
    out_dir = Path("output/dr_compare")
    labels_path = Path("data/eval_labels.tsv")  # protein_id,seq_length,taxonomy,function

    meta, data = load_projection_tables(out_dir)
    labels = pd.read_csv(labels_path, sep="\t")

    rows = []
    for pname in meta["projection_name"].unique():
        ids, X = extract_projection_xy(data, pname)
        df = pd.DataFrame({"protein_id": ids})
        df = df.merge(labels, on="protein_id", how="inner")
        Xi = X[: len(df)]

        # Length R^2
        reg = LinearRegression().fit(Xi, df["seq_length"].to_numpy())
        length_r2 = r2_score(df["seq_length"].to_numpy(), reg.predict(Xi))

        # Taxonomy and function F1
        tax_f1 = cv_macro_f1(Xi, df["taxonomy"].astype(str).to_numpy())
        fun_f1 = cv_macro_f1(Xi, df["function"].astype(str).to_numpy())

        # Functional silhouette
        y_fun = LabelEncoder().fit_transform(df["function"].astype(str).to_numpy())
        sil = silhouette_score(Xi, y_fun) if len(np.unique(y_fun)) > 1 else np.nan

        rows.append(
            {
                "projection": pname,
                "length_r2": float(length_r2),
                "taxonomy_macro_f1": float(tax_f1),
                "function_macro_f1": float(fun_f1),
                "function_silhouette": float(sil),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        by=["function_macro_f1", "function_silhouette"], ascending=False
    )
    print(result.to_string(index=False))
    result.to_csv("output/dr_compare/eval_summary.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
```

Run:

```bash
python scripts/eval_confounders_vs_function.py
```

Adjust:
- projection coordinate column names (`x`, `y`) if your table uses different names
- label file path and column names

---

## 7. How to Generate Target/Background Subsets

If you already have two H5 embedding files, use them directly.

If not, create subsets from a master table:
- `target_ids.txt`: redundancy-reduced SwissProt IDs
- `background_ids.txt`: contrastive background IDs

Then filter/export embeddings into two H5 files (`target.h5`, `background.h5`) using your existing data-prep workflow.

Important rules:
- same embedding model/dimension in target and background
- avoid overlap leakage when possible
- verify both sets have enough samples

---

## 8. Pros and Cons of Each Method

### 8.1 `ppca` (ρPCA, contrastive)
Pros:
- directly optimizes target-vs-background variance ratio
- best fit for confounder suppression hypothesis
- can reveal biology hidden by dominant nuisance axes

Cons:
- requires meaningful background design
- more sensitive to covariance conditioning (`regularization_mu`)
- interpretation depends on contrast choice

### 8.2 `pca`
Pros:
- fast, deterministic, interpretable linear baseline
- good sanity check

Cons:
- not contrastive
- often dominated by global nuisance variation

### 8.3 `umap`
Pros:
- good local neighborhood visualization
- often strong visual cluster separation

Cons:
- not contrastive
- stochastic, hyperparameter-sensitive
- geometry less directly interpretable

### 8.4 `tsne`
Pros:
- strong local cluster visualization
- useful for qualitative pattern discovery

Cons:
- not contrastive
- weak global geometry preservation
- sensitive to perplexity/seed; can over-emphasize apparent clusters

---

## 9. Suggested Experimental Matrix

Run at least:
1. `PCA/UMAP/t-SNE` on target only.
2. `ppca` with background A (broad taxonomy+length).
3. `ppca` with background B (taxonomy-matched).
4. `ppca` with background C (length-matched).

This tells you whether improvements come from contrastive formulation or from a specific background choice.

---

## 10. Practical Interpretation Template

When reporting:
- “Compared with PCA/UMAP/t-SNE, `ppca` reduced sequence-length R² from X to Y and taxonomy macro-F1 from A to B.”
- “At the same time, function macro-F1 improved from C to D (silhouette E to F).”
- “This supports/does not support the hypothesis that contrastive DR suppresses nuisance axes and reveals functional grouping.”

---

## 11. Quick Checklist

- [ ] target/background embedding dimensionality identical
- [ ] `ppca` run with explicit background (`--ppca-background` or `background_path`)
- [ ] same proteins/labels used for method comparison
- [ ] confounder and function metrics both reported
- [ ] multiple seeds for UMAP/t-SNE robustness

