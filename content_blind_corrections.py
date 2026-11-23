"""
content_blind_corrections.py
=============================

FILE 2 OF 3. Run after foundations.py (`--mode embeddings`).

WHAT "CONTENT-BLIND" MEANS, AND WHY IT'S THE RIGHT PLACE TO START
--------------------------------------------------------------------
Every correction in this file operates ONLY on the geometry of the
embeddings themselves -- no information about which Big Five domain or
which of the 30 facets an item is designed to measure ever enters the
calculation. This is a deliberate, conservative starting point: a
correction that works without ever looking at the item's trait label is a
stronger, more general result than one that requires it, and critically, it
cannot possibly leak the evaluation labels into the correction itself. If
content-blind correction were sufficient on its own, this would be a much
simpler and more broadly applicable finding than what content_aware_
corrections.py (file 3) goes on to test.

Spoiler, stated plainly rather than left for the reader to discover: it is
NOT sufficient on its own. That is not a null result -- it is the empirical
argument for why file 3 exists, and the numbers below are the evidence for
that argument, not just a rejected first attempt.

THE FOUR CONDITIONS (methodology doc Conditions 1-4; mean-centering
promoted to its own condition rather than being folded silently into the
others as just a preprocessing step)
------------------------------------------------------------------------
    1_raw       Uncorrected baseline. The control every other condition,
                in this file and file 3, is measured against.
    2_centered  Subtract the population mean from every item embedding.
                The simplest possible correction, and the natural one to
                beat before reaching for anything more complex.
    3_abtt      "All-but-the-top" / drop-top: after centering, delete the
                top m principal-component directions entirely. Competing
                hypothesis to (2): the nuisance signal is concentrated in
                a few directions, not just a constant offset.
    4_zca       Whitening / "full-rescale": after centering, softly rescale
                EVERY direction by its own variance. Competing hypothesis
                to (3): the nuisance signal is distributed across many
                directions, not concentrated in a few.

See foundations.py Section 3 for the exact mathematical specification of
each (mean_center, drop_top_pcs, zca_whiten) -- not duplicated here, so
there is exactly one implementation to audit.

RESULT ON THIS PROJECT'S ACTUAL DATA (696-pair systematic evaluation set)
----------------------------------------------------------------------------
    condition    sign-accuracy    structure corr. vs. raw
    1_raw            0.0%              1.00  (trivial)
    2_centered      27.4%              0.78
    3_abtt          26.7%              0.68
    4_zca           31.6%              0.73

All three corrections eliminate the population-wide positive offset
(mean_offdiag_sim drops from +0.244 to ~0), but none gets anywhere close to
solving the actual problem: at best, ZCA gets fewer than a third of
unambiguous reverse-keyed pairs correctly signed. The offset was never the
whole story -- see file 3.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from foundations import (
    load_embeddings,
    load_reverse_keyed_pairs,
    load_empirical_correlation_matrix,
    build_empirical_sign_lookup,
    cosine_similarity_matrix,
    save_similarity_csv,
    mean_center,
    drop_top_pcs,
    zca_whiten,
    build_comparison_summary,
    plot_comparison,
)


def compute_content_blind_similarities(
    embeddings: np.ndarray,
    drop_top_m: int = 1,
    zca_alpha: float = 0.5,
    zca_eps_frac: float = 0.01,
) -> dict[str, np.ndarray]:
    """
    Run all four content-blind conditions and return their similarity
    matrices. This function is imported directly by content_aware_
    corrections.py (file 3) so the two families are computed identically
    every time, rather than file 3 keeping its own copy of "how to compute
    the content-blind baseline" that could quietly drift out of sync with
    this file.

    Returns
    -------
    dict mapping condition name -> similarity matrix (each shape (300, 300))
    """
    conditions = {
        "1_raw": embeddings,
        "2_centered": mean_center(embeddings),
        "3_abtt": drop_top_pcs(embeddings, m=drop_top_m),
        "4_zca": zca_whiten(embeddings, alpha=zca_alpha, eps_frac=zca_eps_frac),
    }
    return {name: cosine_similarity_matrix(e) for name, e in conditions.items()}


def run(
    base_dir: str | None = None,
    drop_top_m: int = 1,
    zca_alpha: float = 0.5,
    zca_eps_frac: float = 0.01,
    pairs_path: str | None = None,
    empirical_path: str | None = None,
    plot: bool = False,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
    pairs_path = pairs_path or os.path.join(base_dir, "reverse_keyed_pairs.csv")
    empirical_path = empirical_path or os.path.join(base_dir, "empirical_correlation_matrix.csv")

    items, embeddings = load_embeddings(os.path.join(base_dir, "embeddings.csv"))
    print(f"Loaded {len(items)} items x {embeddings.shape[1]} dims\n")

    similarities = compute_content_blind_similarities(
        embeddings, drop_top_m=drop_top_m, zca_alpha=zca_alpha, zca_eps_frac=zca_eps_frac
    )
    for name, sim in similarities.items():
        save_similarity_csv(sim, os.path.join(base_dir, f"similarity_matrix_{name}.csv"))
        print(f"Saved -> similarity_matrix_{name}.csv")

    pairs = []
    if os.path.exists(pairs_path):
        pairs = load_reverse_keyed_pairs(pairs_path)
        print(f"\nLoaded {len(pairs)} reverse-keyed pairs from {pairs_path}")
    else:
        print(f"\nNo pairs file at {pairs_path} -- skipping sign-accuracy.")

    # --- Theoretical evaluation (assumes every reverse-keyed pair "should"
    # be negative) -- always available, no real response data required.
    summary = build_comparison_summary(similarities, items, pairs=pairs, reference_name="1_raw")
    print("\n=== Content-blind comparison (theoretical sign-accuracy) ===")
    print(summary.round(4).to_string())
    summary_path = os.path.join(base_dir, "comparison_summary_content_blind.csv")
    summary.to_csv(summary_path)
    print(f"Saved -> {summary_path}")

    # --- Empirical evaluation, only if the real correlation matrix has been
    # supplied (see foundations.py --mode empirical). Skipped gracefully
    # otherwise, rather than failing the whole run.
    if pairs and os.path.exists(empirical_path):
        empirical = load_empirical_correlation_matrix(empirical_path)
        empirical_signs = build_empirical_sign_lookup(empirical, items, pairs)
        empirical_summary = build_comparison_summary(
            similarities, items, pairs=pairs, reference_name="1_raw", empirical_signs=empirical_signs
        )
        print("\n=== Content-blind comparison (sign-accuracy vs. REAL empirical data) ===")
        print(empirical_summary[["reverse_keyed_sign_accuracy"]]
              .rename(columns={"reverse_keyed_sign_accuracy": "sign_accuracy_vs_empirical"})
              .round(4).to_string())
        empirical_summary.to_csv(os.path.join(base_dir, "comparison_summary_content_blind_empirical.csv"))
    else:
        print(f"\n(No empirical validation: {empirical_path} not found yet -- "
              f"run foundations.py --mode empirical wherever the real response data lives.)")

    if plot:
        plot_path = os.path.join(base_dir, "comparison_plot_content_blind.png")
        plot_comparison(summary, plot_path)
        print(f"Saved -> {plot_path}")

    return similarities, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Content-blind corrections (mean-centering, drop-top, ZCA) + comparison."
    )
    parser.add_argument("--drop-top-m", type=int, default=1,
                         help="Top PCs to remove for all-but-the-top (default: 1; see docstring for why).")
    parser.add_argument("--zca-alpha", type=float, default=0.5,
                         help="Rescale exponent for whitening (0.5=soft default, 1.0=full ZCA).")
    parser.add_argument("--zca-eps-frac", type=float, default=0.01,
                         help="Regularization epsilon as a fraction of lambda_1.")
    parser.add_argument("--pairs-path", type=str, default=None)
    parser.add_argument("--empirical-path", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    run(
        drop_top_m=args.drop_top_m,
        zca_alpha=args.zca_alpha,
        zca_eps_frac=args.zca_eps_frac,
        pairs_path=args.pairs_path,
        empirical_path=args.empirical_path,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()
