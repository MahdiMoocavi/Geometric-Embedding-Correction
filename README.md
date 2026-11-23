# Geometric Correction of IPIP-NEO-300 Item Embeddings

Do purely geometric corrections to raw sentence embeddings — computed from
item text alone, no participant response data required — recover the
correct (negative) relationship between reverse-keyed personality items
that raw embeddings get wrong?

This repository holds the code and results. The scientific argument itself
(motivation, related work, full discussion) lives in the accompanying
report, not here — this file is for finding your way around the code and
reproducing the results.

## The problem, in one paragraph

Sentence embeddings blend two things: content specific to an item, and a
component shared by ~every item in the pool (register, topic, syntax). The
shared component inflates similarity between any two items regardless of
content, and it hits reverse-keyed item pairs hardest — e.g. "Get angry
easily." and "Rarely get irritated.", which should be strongly *negatively*
related, instead come out positive or neutral in raw embedding space. This
matters beyond the pair itself: any downstream network-estimation method
fed a similarity matrix like this has no way to know the sign is wrong.

## Key result so far

Sign-accuracy on a systematic 696-pair reverse-keyed evaluation set (every
plus-keyed item paired with every minus-keyed item within the same facet,
derived from the instrument's actual verified keying — not hand-picked):

| Condition | Uses facet/domain labels? | Sign-accuracy |
|---|---|---|
| Raw (uncorrected) | No | 0.0% |
| Mean-centered | No | 27.4% |
| Drop-top (all-but-the-top) | No | 26.7% |
| ZCA whitening | No | 31.6% |
| Domain-centering (blanket) | Yes | 42.8% |
| Facet-centering (blanket) | Yes | 86.5% |
| Nested, drop-top base | Yes | 86.2% |
| **Nested, full-rescale base** | **Yes** | **90.5%** |

Global correction alone caps out under a third correct. Correction that
uses the item's known facet — even minimally — jumps to the high 80s/low
90s. The best-performing condition also best preserves the item pool's
overall similarity structure relative to raw, not just its sign-accuracy.

**Important caveat, controlled for rather than glossed over:** centering
within a small group (10 items per facet) has a mechanical side effect —
it mathematically forces that group's residuals toward a floor of about
−0.11 pairwise similarity, independent of any real content. This was
caught via a random-noise stress test and is controlled for directly:
`same_keyed_control_pairs.csv` holds same-facet pairs that feel the
identical mechanical pull but should genuinely stay positive. The gap
between reverse-keyed pairs and this control set (+0.06 to +0.12 across
the facet-informed conditions) is the artifact-controlled evidence for
real signal — see the docstring in `content_aware_corrections.py` for the
full derivation and numbers.

**This is all still the theoretical/embedding-only result.** Validation
against real response data (145K+ participants, same 300 items) is the
next stage — see Status below.

## Repository contents

```
foundations.py                  Item pool, raw embeddings, the full
                                 correction + evaluation toolbox, and
                                 empirical-validation utilities.
                                 RUN THIS FIRST.

content_blind_corrections.py    Conditions that use only embedding
                                 geometry: raw, mean-centered, drop-top,
                                 ZCA. Imports its toolbox from
                                 foundations.py.

content_aware_corrections.py    Conditions that additionally use verified
                                 facet/domain labels: domain-/facet-
                                 blanket centering, and two nested
                                 (scope-matched) corrections. Imports
                                 shared logic from both files above,
                                 produces the full 1-8 comparison, and
                                 runs the artifact-control check.

item_metadata.csv               Domain, facet, and keying direction for
                                 all 300 items. Verified two independent
                                 ways against the actual IPIP-NEO-300
                                 scoring key (a GitHub data mirror and the
                                 five-factor-e PyPI package) -- zero
                                 keying mismatches on 299/300 items
                                 cross-checked.

reverse_keyed_pairs.csv         The 696-pair systematic evaluation set,
                                 derived directly from the verified keying
                                 above.

same_keyed_control_pairs.csv    654 same-facet, same-keying-direction
                                 pairs -- the artifact control for the
                                 facet-level conditions (see caveat above).

geometric-correction-methodology.md   Full formal specification of every
                                       correction (the notation the code's
                                       docstrings match against directly).

bibliography.md                 Full reference list.

requirements.txt                Exact packages needed to run this pipeline.
```

Running the pipeline also produces (not included until you run it):
`embeddings.csv`, `similarity_matrix_<condition>.csv` for each of the 8
conditions, `comparison_summary_*.csv`, and comparison plots.

## Setup

```bash
pip install -r requirements.txt
```

## Running it

```bash
python foundations.py --mode embeddings --plot
python content_blind_corrections.py --plot
python content_aware_corrections.py --plot
```

Each script prints its own results table and saves it as CSV. All three
read/write plain CSVs in the working directory — no hidden state, and
nothing is recomputed unnecessarily (`foundations.py` caches embeddings
and skips re-encoding on repeat runs).

### Adding the empirical validation

The response-data step is deliberately decoupled from everything above —
run it wherever the actual response data lives, since only the small
resulting correlation matrix needs to travel, not 145K rows of raw
responses:

```bash
python foundations.py --mode empirical --input IPIP300-SCORES.parquet
```

(reading `.parquet` needs `pyarrow` — `pip install pyarrow` if you don't
already have it; not needed anywhere else in this pipeline.)

Copy the resulting `empirical_correlation_matrix.csv` into the same
directory as everything else, then re-run the two correction scripts —
both automatically detect the file and add the empirical comparison to
their output.

## A note on the item pool

The 300 items are the standard, public-domain IPIP-NEO-300 inventory
(Goldberg / Johnson). This was confirmed, not assumed — see
`geometric-correction-methodology.md` and the docstring in
`content_aware_corrections.py` for how the facet/domain/keying metadata was
cross-verified against two independent public sources.

## Status

- [x] Content-blind corrections (raw, centered, drop-top, ZCA)
- [x] Content-aware corrections (domain/facet blanket, nested)
- [x] Systematic 696-pair reverse-keyed evaluation set
- [x] Full 1-8 theoretical comparison
- [ ] Empirical validation against real response data (in progress)
- [ ] Written report (in progress)
