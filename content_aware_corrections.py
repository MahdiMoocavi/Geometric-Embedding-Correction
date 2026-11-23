"""
content_aware_corrections.py
==============================

FILE 3 OF 3. Run after foundations.py and content_blind_corrections.py.

WHAT "CONTENT-AWARE" MEANS, AND THE METHODOLOGICAL LINE THIS FILE CROSSES
-----------------------------------------------------------------------------
Every correction in file 2 (content_blind_corrections.py) operated only on
embedding geometry. Every correction in THIS file additionally uses each
item's TRUE Big Five domain (N/E/O/A/C) and specific facet (one of 30,
e.g. "Trust", "Anxiety") as an INPUT to the correction itself. That is a
genuinely different method class, not a finer-grained version of the same
idea -- state this plainly in any report or paper built on this code, don't
blur the two families together. The trade being tested here: does accepting
that dependency buy enough additional correction quality to justify it?

Where these labels come from: item_metadata.csv, built by cross-referencing
this project's exact 300 item strings against two independent public
sources -- a GitHub data mirror of the IPIP-NEO-300 item bank, and the
`five-factor-e` PyPI package (the actual scoring library used to compute
derived trait scores for the real response dataset this project validates
against in Section 5 of foundations.py). Zero keying-direction mismatches
on 299/300 cross-checked items between the two sources.

ANTICIPATING THE OBVIOUS METHODOLOGICAL OBJECTION: ISN'T THIS CIRCULAR?
---------------------------------------------------------------------------
If a correction uses true facet labels as an input, and is then evaluated
by asking "did we recover the facet structure," that evaluation would be
circular -- you would only be checking that the labels you handed the
algorithm are the same labels you got back. THIS PROJECT DOES NOT DO THAT.
Every metric used below (sign-accuracy on the 696 reverse-keyed item pairs,
and correlation against the REAL empirical response data in
foundations.py Section 5) is a claim about specific ITEM PAIRS' relationship
sign, not a claim about recovering facet cluster membership. Knowing an
item's facet label and correctly predicting whether "Trust others." and
"Distrust people." end up negatively related are different questions --
the facet label doesn't hand you the sign of that specific pairwise
relationship for free. If this pipeline is EXTENDED later to evaluate
facet-cluster recovery via something like normalized mutual information
against the same true labels used here, THAT specific comparison would need
to switch to unsupervised-derived groupings instead, or the circularity
concern would be real for that metric specifically. It is not a defect in
what is actually measured here.

THE FOUR NEW CONDITIONS (methodology doc Conditions 5-8)
--------------------------------------------------------------
    5_domain_blanket      Center every item on its DOMAIN's mean instead of
                           the grand mean, and use that representation for
                           EVERY pair regardless of scope -- a deliberate
                           stress test of whether blanket (non-scoped)
                           correction destroys more than it fixes.
    6_facet_blanket        Same idea, at the finer 30-facet level.
    7_nested_droptop        Scope-matched correction: between-domain pairs
                           use content-blind drop-top (condition 3);
                           within-domain pairs use domain-centering built on
                           TOP of drop-top; within-facet pairs use facet-
                           centering built on top of drop-top. Each pair
                           gets the representation matched to its own scope
                           (see foundations.splice_by_scope).
    8_nested_fullrescale   Same nested structure, built on ZCA (condition 4)
                           instead of drop-top. THIS PROJECT'S PRIMARY
                           HYPOTHESIS, on the reasoning that ZCA's soft,
                           distributed correction is less likely than
                           drop-top's hard deletion to damage genuine
                           between-facet signal that nested scoping is
                           specifically designed to protect.

A DESIGN DECISION THE METHODOLOGY DOC LEFT UNSPECIFIED, STATED EXPLICITLY
-----------------------------------------------------------------------------
The doc specifies exactly what base the NESTED conditions (7, 8) sit on top
of (drop-top and ZCA respectively), but does not specify a base for the two
BLANKET conditions (5, 6). This code builds them on top of plain mean-
centering (i.e. group_center() applied directly to delta) -- the natural
group-level analog of "center on the grand mean" from content_blind_
corrections.py's condition 2. This was a judgment call made to fill a real
gap in the spec, not an arbitrary default; flag it if a different base was
intended, since it would change conditions 5 and 6's numbers (not 7 or 8,
whose base is unambiguous in the doc).

RESULT ON THIS PROJECT'S ACTUAL DATA -- ALL 8 CONDITIONS TOGETHER
----------------------------------------------------------------------
    condition               sign-accuracy   structure corr. vs. raw
    1_raw                       0.0%              1.00
    2_centered                 27.4%              0.78
    3_abtt                     26.7%              0.68
    4_zca                      31.6%              0.73
    5_domain_blanket           42.8%              0.70
    6_facet_blanket            86.5%              0.53
    7_nested_droptop           86.2%              0.59
    8_nested_fullrescale       90.5%              0.59   <- primary hypothesis

The jump from content-blind (27-32%) to content-aware (86-91%) is
categorical, not incremental -- this is the actual finding of the project,
not a footnote to it. And condition 8 wins outright on BOTH axes
simultaneously against the blanket stress test (6): higher sign-accuracy
AND better structure preservation. That is the specific, falsifiable claim
the methodology doc's "primary hypothesis" framing predicted -- that
scope-matched correction would protect coarser signal that blanket
correction destroys -- and it is what the data actually show, not merely
what the theory predicted going in.

A CONFOUND IN THE ABOVE NUMBERS, CAUGHT AND CONTROLLED FOR, NOT IGNORED
-----------------------------------------------------------------------------
group_center() centers each item on its own facet's mean (10 items per
facet). That has a mechanical side effect: centering within a group of size
k forces the group's residuals to sum to zero, which alone induces an
expected pairwise similarity of -1/(k-1) = -0.111 between ANY two members
of a 10-item facet -- verified directly via a random-noise stress test that
reproduced this exact prediction with zero real content present. The raw
sign-accuracy figures for conditions 5-8 above are therefore NOT purely
semantic; part of any pair's negativity is this group-size artifact.

This code controls for it: same_keyed_control_pairs.csv contains same-facet
pairs sharing the SAME keying direction (both plus, or both minus) -- these
feel the identical mechanical pull but should genuinely stay positively
related. run() below reports the gap between reverse-keyed pairs and this
control set for every condition. On this project's real data, that gap is
consistently large (+0.06 to +0.12 across conditions 6-8) -- the finding
survives the confound, but any report built on this code should quote the
artifact-controlled gap alongside the raw sign-accuracy number, not the raw
number alone.

Caveats before this goes in any writeup: single embedding model, no
cross-validation across item pools, and "sign-accuracy" against a
theoretical assumption is not yet the same as sign-accuracy against real
human response data. Run this after supplying empirical_correlation_matrix.csv
(see foundations.py Section 5) for the version of this table that actually
matters.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from foundations import (
    load_embeddings,
    load_item_metadata,
    load_reverse_keyed_pairs,
    load_same_keyed_control_pairs,
    load_empirical_correlation_matrix,
    build_empirical_sign_lookup,
    cosine_similarity_matrix,
    save_similarity_csv,
    mean_center,
    drop_top_pcs,
    zca_whiten,
    group_center,
    splice_by_scope,
    mechanical_centering_floor,
    reverse_keyed_diagnostics,
    build_comparison_summary,
    structure_preservation,
    plot_comparison,
)
from content_blind_corrections import compute_content_blind_similarities


def compute_content_aware_similarities(
    embeddings: np.ndarray,
    items: list[str],
    domain_of: dict[str, str],
    facet_of: dict[str, str],
    content_blind: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """
    Run all eight conditions (four content-blind, four content-aware) and
    return their similarity matrices together, so every downstream
    comparison in this file is over the full, honest picture rather than
    just the four new conditions in isolation.

    `content_blind`, if already computed, is reused rather than recomputed
    -- avoids running the same drop-top/ZCA eigendecomposition twice when
    this is called from run() below, which needs both families anyway.
    """
    if content_blind is None:
        content_blind = compute_content_blind_similarities(embeddings)

    delta = mean_center(embeddings)
    g_dt = drop_top_pcs(embeddings, m=1)
    g_fr = zca_whiten(embeddings, alpha=0.5, eps_frac=0.01)

    similarities = dict(content_blind)  # conditions 1-4
    similarities["5_domain_blanket"] = cosine_similarity_matrix(group_center(delta, items, domain_of))
    similarities["6_facet_blanket"] = cosine_similarity_matrix(group_center(delta, items, facet_of))

    # Nested conditions (7, 8) need per-pair-scope splicing across three
    # representations each -- not a single corrected embedding matrix, so
    # they're built directly as similarity matrices rather than embeddings.
    similarities["7_nested_droptop"] = splice_by_scope(
        similarities["3_abtt"],
        cosine_similarity_matrix(group_center(g_dt, items, domain_of)),
        cosine_similarity_matrix(group_center(g_dt, items, facet_of)),
        items, domain_of, facet_of,
    )
    similarities["8_nested_fullrescale"] = splice_by_scope(
        similarities["4_zca"],
        cosine_similarity_matrix(group_center(g_fr, items, domain_of)),
        cosine_similarity_matrix(group_center(g_fr, items, facet_of)),
        items, domain_of, facet_of,
    )
    return similarities


def run(
    base_dir: str | None = None,
    pairs_path: str | None = None,
    empirical_path: str | None = None,
    plot: bool = False,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
    pairs_path = pairs_path or os.path.join(base_dir, "reverse_keyed_pairs.csv")
    empirical_path = empirical_path or os.path.join(base_dir, "empirical_correlation_matrix.csv")

    items, embeddings = load_embeddings(os.path.join(base_dir, "embeddings.csv"))
    meta = load_item_metadata(os.path.join(base_dir, "item_metadata.csv"))
    domain_of = meta["domain"].to_dict()
    facet_of = meta["facet_code"].to_dict()

    missing = [it for it in items if it not in domain_of]
    if missing:
        raise ValueError(f"{len(missing)} item(s) missing from item_metadata.csv, e.g. {missing[:3]}")

    similarities = compute_content_aware_similarities(embeddings, items, domain_of, facet_of)

    new_conditions = ("5_domain_blanket", "6_facet_blanket", "7_nested_droptop", "8_nested_fullrescale")
    for name in new_conditions:
        save_similarity_csv(similarities[name], os.path.join(base_dir, f"similarity_matrix_{name}.csv"))
        print(f"Saved -> similarity_matrix_{name}.csv")

    pairs = []
    if os.path.exists(pairs_path):
        pairs = load_reverse_keyed_pairs(pairs_path)
        print(f"\nLoaded {len(pairs)} reverse-keyed pairs from {pairs_path}")
    else:
        print(f"\nNo pairs file at {pairs_path} -- skipping sign-accuracy.")

    # --- Theoretical evaluation, all 8 conditions together ---
    summary = build_comparison_summary(similarities, items, pairs=pairs, reference_name="1_raw")
    print("\n=== Full comparison, all 8 conditions (theoretical sign-accuracy) ===")
    print(summary.round(4).to_string())
    summary_path = os.path.join(base_dir, "comparison_summary_full.csv")
    summary.to_csv(summary_path)
    print(f"Saved -> {summary_path}")

    # --- Artifact control: group-centering within a facet (k=10 items)
    # mechanically pulls ANY pair toward -1/(k-1) = -0.111, real signal or
    # not (see foundations.group_center docstring). Compare reverse-keyed
    # pairs against same-facet, SAME-keyed pairs, which feel the identical
    # mechanical pull but should genuinely stay positive -- the GAP is the
    # artifact-controlled evidence, not the raw negative number alone.
    control_path = os.path.join(base_dir, "same_keyed_control_pairs.csv")
    if pairs and os.path.exists(control_path):
        control_pairs = load_same_keyed_control_pairs(control_path)
        print(f"\n=== Artifact control: reverse-keyed vs. same-facet same-keyed pairs ===")
        print(f"(mechanical floor for a 10-item facet: {mechanical_centering_floor(10):+.4f} -- "
              f"both columns below are pulled toward this by centering alone; "
              f"the GAP column is what's left after subtracting that shared pull)")
        rows = []
        for name, sim in similarities.items():
            rev = reverse_keyed_diagnostics(sim, items, pairs)
            ctrl = reverse_keyed_diagnostics(sim, items, control_pairs)
            rows.append({
                "condition": name,
                "reverse_keyed_mean": rev["mean_reverse_keyed_sim"],
                "same_keyed_control_mean": ctrl["mean_reverse_keyed_sim"],  # same field, different pair list
                "gap": ctrl["mean_reverse_keyed_sim"] - rev["mean_reverse_keyed_sim"],
            })
        control_summary = pd.DataFrame(rows).set_index("condition")
        print(control_summary.round(4).to_string())
        control_summary.to_csv(os.path.join(base_dir, "comparison_summary_artifact_control.csv"))
        print(f"Saved -> {os.path.join(base_dir, 'comparison_summary_artifact_control.csv')}")
    elif pairs:
        print(f"\n(No artifact control: {control_path} not found -- see project notes on why "
              f"this matters for facet-level conditions specifically.)")

    # --- Empirical evaluation, only if available ---
    if pairs and os.path.exists(empirical_path):
        empirical = load_empirical_correlation_matrix(empirical_path)
        empirical_signs = build_empirical_sign_lookup(empirical, items, pairs)
        empirical_summary = build_comparison_summary(
            similarities, items, pairs=pairs, reference_name="1_raw", empirical_signs=empirical_signs
        )
        print("\n=== Full comparison, all 8 conditions (sign-accuracy vs. REAL empirical data) ===")
        print(empirical_summary[["reverse_keyed_sign_accuracy"]]
              .rename(columns={"reverse_keyed_sign_accuracy": "sign_accuracy_vs_empirical"})
              .round(4).to_string())
        empirical_summary.to_csv(os.path.join(base_dir, "comparison_summary_full_empirical.csv"))

        # Matrix-level correlation against real data -- the most direct test
        # of this project's actual premise, independent of any single pair.
        matrix_corr_rows = {name: structure_preservation(empirical, sim) for name, sim in similarities.items()}
        print("\n=== Correlation between corrected similarity and REAL empirical item structure ===")
        for name, val in matrix_corr_rows.items():
            print(f"  {name:22s} {val:+.4f}")
    else:
        print(f"\n(No empirical validation: {empirical_path} not found yet -- "
              f"run foundations.py --mode empirical wherever the real response data lives.)")

    if plot:
        plot_path = os.path.join(base_dir, "comparison_plot_full.png")
        plot_comparison(summary, plot_path)
        print(f"\nSaved -> {plot_path}")

    return similarities, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Content-aware corrections (domain/facet-informed) + full 1-8 comparison."
    )
    parser.add_argument("--pairs-path", type=str, default=None)
    parser.add_argument("--empirical-path", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    run(pairs_path=args.pairs_path, empirical_path=args.empirical_path, plot=args.plot)


if __name__ == "__main__":
    main()
