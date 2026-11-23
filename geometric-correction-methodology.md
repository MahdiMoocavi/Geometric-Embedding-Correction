# Geometric correction of sentence embeddings: methodology

## 1. Research question

This project tests whether geometric transformations of sentence embeddings improve the recovery of expected negative relationships between reverse-keyed personality items. The item text is embedded without participant response data. The current evaluation is therefore an embedding-space proxy; it is not, by itself, evidence that the corrected matrix reproduces empirical psychometric relationships.

## 2. Item pool and representation

The implementation contains 300 IPIP-NEO-300 item stems in the canonical order stored in `ITEM_STEMS` in `foundations.py`. The default encoder is Sentence Transformers' `all-mpnet-base-v2`, which returns a 768-dimensional vector for each item.

Let (x_i \in \mathbb{R}^{768}) denote the embedding for item (i). The raw item-to-item similarity is cosine similarity:

\[
s_{ij}=\frac{x_i^\top x_j}{\|x_i\|\,\|x_j\|}.
\]

The project evaluates 696 within-facet pairs formed by pairing each plus-keyed item with each minus-keyed item in the same facet. The working theoretical expectation is that such reverse-keyed pairs should have negative similarity. The evaluation does not assume that this theoretical expectation is always correct in observed responses; the optional empirical stage is intended to test that distinction.

## 3. Content-blind conditions

The first four conditions use only the embedding matrix and do not use item-domain or facet labels.

### 1. Raw baseline

Use the original embeddings (x_i) without transformation.

### 2. Global mean-centering

Calculate the global item-embedding mean (\bar{x}), then:

\[
\delta_i=x_i-\bar{x}.
\]

The cosine similarity matrix is calculated from the centred vectors.

### 3. Drop-top / all-but-the-top

Compute the covariance matrix of the centred embeddings and its eigendecomposition:

\[
C=\frac{1}{n-1}\Delta^\top\Delta=V\Lambda V^\top.
\]

Remove the first (m) principal-component directions:

\[
g_i^{DT}=\delta_i-\sum_{\ell=1}^{m}(\delta_i^\top v_\ell)v_\ell.
\]

The implementation defaults to (m=1). The code comments recommend choosing (m) from the eigenvalue spectrum before inspecting evaluation scores rather than tuning it to maximize sign-accuracy on the same pair set.

### 4. ZCA / full-rescale

The implementation applies regularized variance rescaling in the PCA basis:

\[
g_i^{FR}=\sum_{\ell}
\frac{\delta_i^\top v_\ell}{(\lambda_\ell+\epsilon)^{\alpha/2}}v_\ell,
\qquad \epsilon=\text{eps\_frac}\times\lambda_1.
\]

Defaults are (\alpha=0.5) and `eps_frac=0.01). The implementation uses a numerical floor for very small denominators. (\alpha=1) corresponds to full ZCA whitening under this parameterisation; the default is partial rescaling.

## 4. Content-aware conditions

These conditions additionally use the instrument's known domain and facet metadata in `item_metadata.csv`. The metadata contain five domain labels and 30 facet codes.

### 5. Domain blanket-centering

For each item, subtract the mean embedding of its domain from the globally mean-centred embedding. The resulting domain-centred representation is used for every item pair, including pairs from different domains. This is a blanket transformation, not a pair-specific choice of correction.

### 6. Facet blanket-centering

The same operation is performed using each item's facet mean, and that representation is used for every item pair.

### 7. Nested drop-top

The correction is selected at the scope of each item pair:

- Between different domains: use the content-blind drop-top similarity.
- Within the same domain but different facets: use the similarity of domain-centred drop-top vectors.
- Within the same facet: use the similarity of facet-centred drop-top vectors.

The three resulting similarity matrices are spliced according to the pair's domain and facet membership.

### 8. Nested full-rescale

Use the same pair-scope selection as condition 7, but build the three representations on the ZCA / full-rescale base rather than the drop-top base.

The nested methods therefore produce a similarity matrix by combining entries from different representations; they are not equivalent to applying one single corrected embedding matrix to all items and then calculating cosine similarity. The pair-scope masks are symmetric, so the assembled matrix remains symmetric, but this construction does **not** guarantee positive semidefiniteness. The current scripts do not test this property. Before using a nested matrix with a downstream method that requires a positive-semidefinite Gram or correlation matrix, inspect its eigenvalues and decide whether a PSD adjustment is justified for that use case.

## 5. Evaluation metrics

### Reverse-keyed sign-accuracy

For each of the 696 predefined reverse-keyed pairs, the code checks whether the cosine similarity has a negative sign. Sign-accuracy is the proportion of pairs passing this check. A zero similarity is not counted as negative.

The current comparison values are recorded in the source-code documentation and README. They should be regenerated with the scripts before publication or use as final reported results.

### Similarity-structure correlation

The code calculates the Pearson correlation between the off-diagonal entries of a condition's similarity matrix and those of the raw similarity matrix. This describes how much the overall pairwise similarity pattern changes relative to the raw baseline. It is a structure-retention diagnostic, not a criterion that must be maximized independently of the target pairwise sign correction.

### Same-keyed artifact control

`same_keyed_control_pairs.csv` contains 654 same-facet pairs with the same keying direction. These pairs are intended to control for the mechanical effect of centering all items within a small group. The content-aware script reports their mean similarity alongside the reverse-keyed-pair mean and their difference for each condition.

The code includes the reference value (-1/(k-1)) for a group of (k) items (approximately -0.111 for (k=10)). This is a theoretical group-centering reference under the assumptions described in the code comments. It should not be treated as a universal exact expectation for observed cosine similarities.

## 6. Optional response-based validation

The optional empirical mode accepts a CSV or Parquet dataset containing item response columns `i1` through `i300` in exactly the canonical order used by `ITEM_STEMS`. It:

1. Retains respondents with at least 50% non-missing item responses.
2. Computes pairwise Pearson correlations between the 300 item-response columns.
3. Writes a 300-by-300 empirical correlation matrix.
4. Allows the correction scripts to evaluate the signs of reverse-keyed pairs against the empirical signs.

The content-aware script also calculates the correlation between each candidate similarity matrix and the empirical response-correlation matrix. The code uses Pearson correlations for ordinal item responses and explicitly notes that polychoric correlations may be a useful later sensitivity analysis.

The individual-level response data and the empirical correlation matrix are not included in this repository.

## 7. Interpretation and limitations

- The current results are specific to one sentence-embedding model and one 300-item personality inventory.
- The content-aware methods use known domain and facet labels as part of the correction. They are not unsupervised or purely text-only transformations.
- The current sign-accuracy evaluation uses a theoretical expected sign. Some reverse-keyed pairs may not be negatively correlated in actual responses because of item wording, response styles, or other substantive factors.
- A high negative-sign recovery rate on the predefined pair set does not establish recovery of the full psychometric structure.
- The nested conditions splice pairwise similarities from multiple representations. Positive semidefiniteness is not guaranteed and is not currently checked; this matters for downstream algorithms that require a valid Gram or correlation matrix.
- The current files do not provide cross-validation across independent item pools, embedding models, or samples.
- The optional empirical stage is therefore important before treating the correction as a validated psychometric procedure.

## 8. Reproduction order

From the repository directory:

```bash
pip install -r requirements.txt
python foundations.py --mode embeddings
python content_blind_corrections.py --plot
python content_aware_corrections.py --plot
```

The foundations script generates the embeddings and raw baseline. The two correction scripts reuse the saved embeddings and produce their respective comparison tables. The scripts write generated files next to the source files. See the README for the optional empirical-validation command and package requirements.
