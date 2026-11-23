# Geometric Correction of IPIP-NEO-300 Item Embeddings

Can geometric corrections improve the recovery of expected negative relationships between reverse-keyed personality items in sentence-embedding space?

This project encodes the 300 IPIP-NEO-300 item statements with `all-mpnet-base-v2`, computes pairwise cosine similarities, and compares global corrections with methods that use the instrument's known Big Five domain and facet labels. The goal is to test whether the geometry of item-text embeddings can better reflect theoretically expected item relationships before any psychometric network estimation.

## Research question and rationale

Sentence embeddings may capture both item-specific semantics and shared features of questionnaire wording, such as register, syntax, and topic. A shared component can inflate similarity across items, including pairs whose scoring directions are reversed.

The project evaluates whether correcting this geometry improves the sign of similarity for reverse-keyed items within the same facet. This is a targeted evaluation of item-pair relationships, not a complete validation of an instrument or a replacement for response-based psychometrics.

## Current embedding-only results

The repository's current code documentation reports the following results on a systematic set of **696 within-facet reverse-keyed pairs**. Sign-accuracy is the proportion of pairs whose cosine similarity has the expected negative sign under the theoretical reverse-keying assumption.

| Condition | Uses domain/facet labels? | Reported sign-accuracy | Similarity-structure correlation vs. raw |
|---|---:|---:|---:|
| Raw embeddings | No | 0.0% | 1.00 |
| Global mean-centering | No | 27.4% | 0.78 |
| Drop-top / all-but-the-top | No | 26.7% | 0.68 |
| ZCA / full-rescale | No | 31.6% | 0.73 |
| Domain-centering (blanket) | Yes | 42.8% | 0.70 |
| Facet-centering (blanket) | Yes | 86.5% | 0.53 |
| Nested drop-top | Yes | 86.2% | 0.59 |
| **Nested full-rescale** | **Yes** | **90.5%** | **0.59** |

These are the results recorded in the project's source-code documentation; they have not been independently re-run as part of this repository review. The strongest reported sign-recovery result is the nested full-rescale condition. It should not be described as universally best on every metric: for example, domain-centering has a higher reported correlation with the raw similarity structure (0.70), but lower sign-accuracy (42.8%).

**Critical qualification:** this is still an embedding-only evaluation against a theoretical expectation. It does not establish that corrected similarities match real item-response correlations. Validation against response-derived correlations remains a separate, unfinished stage.

## Why there are two method families

The four content-blind conditions use only the embedding geometry. The four content-aware conditions additionally use the item's known domain or facet. These are different method classes and should be compared as such.

| Condition | Method family | Operation |
|---|---|---|
| 1. Raw | Baseline | Uncorrected sentence embeddings |
| 2. Centered | Content-blind | Subtract the global item-embedding mean |
| 3. Drop-top | Content-blind | Mean-center, then remove the leading principal-component direction(s) |
| 4. ZCA / full-rescale | Content-blind | Mean-center, then regularized variance rescaling in the PCA basis |
| 5. Domain blanket | Content-aware | Center each item on its Big Five domain mean |
| 6. Facet blanket | Content-aware | Center each item on its facet mean |
| 7. Nested drop-top | Content-aware | Select the global, domain-centred, or facet-centred similarity according to each pair's scope, using drop-top as the base |
| 8. Nested full-rescale | Content-aware | The same scope-matched procedure, using ZCA / full-rescale as the base |

The exact operations and evaluation logic are specified in [geometric-correction-methodology.md](geometric-correction-methodology.md).

## Evaluation and artifact control

- `reverse_keyed_pairs.csv` contains 696 systematically enumerated within-facet pairs: each plus-keyed item is paired with each minus-keyed item in the same facet.
- `same_keyed_control_pairs.csv` contains 654 same-facet pairs with the same keying direction. This is included because group-centering can mechanically shift within-group similarities, independent of semantic information.
- The content-aware script reports the reverse-keyed and same-keyed control means, along with their difference, for each condition. Interpret facet-informed results using this control rather than relying on raw negative-sign recovery alone.

The current implementation reports a group-size reference value of `-1/(k-1)` for a group of size `k` (for ten items, approximately -0.111). Treat this as a theoretical centering reference under the assumptions described in the code, not as a universal expected cosine similarity for arbitrary embedding data.

## Repository contents

| File | Purpose |
|---|---|
| `foundations.py` | Canonical 300-item pool, embedding generation, cosine similarity, correction functions, evaluation metrics, and optional empirical-correlation utilities |
| `content_blind_corrections.py` | Four content-blind conditions and their comparison |
| `content_aware_corrections.py` | Four content-aware conditions, full eight-condition comparison, and same-keyed artifact control |
| `item_metadata.csv` | Domain, facet, and keying-direction metadata for the 300 items |
| `reverse_keyed_pairs.csv` | The 696-pair reverse-keyed evaluation set |
| `same_keyed_control_pairs.csv` | The 654-pair same-keyed artifact-control set |
| `requirements.txt` | Python dependencies |
| `geometric-correction-methodology.md` | Method definitions, evaluation logic, and limitations |

## Setup

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

The first embedding run downloads the `all-mpnet-base-v2` model through Sentence Transformers and therefore requires network access. The model produces 768-dimensional sentence embeddings.

## Run the pipeline

Run these commands from the repository directory:

```bash
# Step 1: generate and cache raw embeddings and the raw similarity matrix
python foundations.py --mode embeddings

# Step 2: evaluate the four content-blind conditions
python content_blind_corrections.py --plot

# Step 3: evaluate all eight conditions and the same-keyed artifact control
python content_aware_corrections.py --plot
```

The foundations script creates its raw similarity heatmap by default. Use `--skip-plot` to suppress it. The two correction scripts accept `--plot` to save their comparison plots.

Generated CSVs and plots are written beside the scripts in the repository directory. Existing embeddings are reused when both `embeddings.csv` and the raw similarity matrix are present; pass `--force` to recompute the embeddings.

## Optional empirical validation

The empirical mode computes a Pearson item-response correlation matrix from a CSV or Parquet dataset containing columns `i1` through `i300`:

```bash
python foundations.py --mode empirical --input /path/to/IPIP300-SCORES.parquet
```

Parquet input requires the optional `pyarrow` package. The script retains respondents with at least 50% non-missing item responses, then computes pairwise Pearson correlations. It writes `empirical_correlation_matrix.csv` beside the scripts. Re-run the two correction scripts after that file is present; they will add comparisons against the empirical pairwise signs.

**Item order is essential.** The response columns `i1`–`i300` must correspond exactly to the canonical item order in `ITEM_STEMS` in `foundations.py`. Do not alphabetically sort the column names. The source response data and empirical matrix are not included in this repository; do not commit individual-level response data.

## Scope and limitations

- One sentence-transformer model (`all-mpnet-base-v2`) and one 300-item personality inventory are currently evaluated.
- The current headline sign-accuracy values are based on a theoretical expected sign for reverse-keyed item pairs, not empirical signs from respondents.
- The content-aware methods depend on known Big Five domain and facet labels; they are not purely text-only corrections.
- The nested conditions splice pairwise similarities from multiple representations. Positive semidefiniteness is not guaranteed or currently checked; inspect the matrix eigenvalues before using it with methods that require a valid Gram or correlation matrix.
- The reported results do not yet establish generalisation to other instruments, item pools, languages, or embedding models.
- The code documents Pearson correlations for the optional response-based validation. Polychoric correlations may be worth examining for ordinal Likert responses in a later sensitivity analysis.
- Model selection, parameter sensitivity, and independent validation across instruments remain necessary before making broad claims about psychometric validity.

## Status

- [x] 300-item IPIP-NEO-300 embedding pipeline
- [x] Four content-blind correction conditions
- [x] Four content-aware correction conditions
- [x] Systematic 696-pair reverse-keyed evaluation set
- [x] 654-pair same-keyed artifact-control set
- [x] Eight-condition embedding-only comparison
- [ ] Empirical validation against response-derived item correlations
- [ ] Sensitivity analysis across embedding models and correction parameters
- [ ] Independent validation on additional item pools
- [ ] Full research report
