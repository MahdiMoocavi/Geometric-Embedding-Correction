"""
foundations.py
==============

FILE 1 OF 3 in this pipeline. Read this one first.

    foundations.py                     <- you are here
    content_blind_corrections.py       <- imports from this file
    content_aware_corrections.py       <- imports from both of the above

WHAT PROBLEM IS THIS PROJECT SOLVING?
--------------------------------------
Sentence embeddings (dense numeric vectors produced by a transformer
language model, here `all-mpnet-base-v2`) are increasingly used to estimate
the structure of a psychometric questionnaire directly from item TEXT --
no participant response data required. The appeal is obvious: item wording
is available before a single person has taken the survey.

The problem: raw embeddings carry a geometric distortion that is invisible
if you only look at a handful of items, but systematic once you look at all
of them together. Every embedding blends two things: (1) content specific
to that item, and (2) a shared component common to ~every sentence in the
set -- register, syntax, "this is a self-report personality item" -- which
has nothing to do with the trait being measured. Component (2) inflates
cosine similarity between ANY two items regardless of content. The clearest
symptom: REVERSE-KEYED item pairs (e.g. "Get angry easily" / "Rarely get
irritated"), which should be strongly NEGATIVELY related, instead come out
positively or near-neutrally related in raw embedding space. That is not a
minor numerical quirk -- it silently corrupts any similarity matrix used as
the input to network estimation (e.g. EGA/GENIE-style psychometric network
methods), because those methods have no way to know the sign is wrong.

This project asks a direct, falsifiable question: can purely geometric
correction of the embeddings -- applied BEFORE any network estimation step,
and requiring nothing beyond the embeddings themselves (or, in the second
half of this project, the item's known trait/facet label) -- recover the
correct negative relationship between reverse-keyed items?

WHAT'S IN THIS FILE
--------------------
    1. The 300-item pool (IPIP-NEO-300, public domain)
    2. Computing raw embeddings + cosine similarity
    3. THE CORRECTION TOOLBOX -- every mathematical operation used by both
       downstream files. One implementation per formula, used everywhere,
       so there is nothing to accidentally let drift out of sync.
    4. Evaluation metrics -- how we decide whether a correction worked
    5. Validation against REAL human response data -- the actual ground
       truth this whole project is ultimately answerable to

TWO WAYS THIS FILE GETS RUN
-----------------------------
This file has two independent jobs that happen to share the same toolbox,
so they live in the same file rather than forcing you to track which of
two near-identical files to look in:

    python foundations.py --mode embeddings
        Encodes the 300 items with all-mpnet-base-v2 and saves embeddings.csv
        + a raw-similarity baseline. Run this on a machine with
        sentence-transformers installed.

    python foundations.py --mode empirical --input RESPONSES.parquet
        Computes the empirical item-to-item correlation matrix from real
        survey response data (raw Likert answers, one column per item).
        Deliberately independent of the embeddings -- run this wherever the
        response data actually lives (it may be a different machine; only
        the small resulting correlation matrix needs to travel, not the raw
        response data). Needs only pandas -- sentence-transformers is
        imported lazily inside compute_embeddings() and is never touched by
        this code path.

ITEM ORDER -- READ THIS BEFORE WRITING ANY NEW CODE AGAINST THIS PIPELINE
---------------------------------------------------------------------------
embeddings.csv row order, item_metadata.csv row order, and the i1...i300
column order in the real response dataset (`ecorbari/IPIP300-SCORES` on
Hugging Face) are ALL the same canonical IPIP-NEO-300 item numbering
(id 1 = "Worry about things.", id 300 = "Often make last-minute plans.").
This alignment was verified two independent ways: against a static GitHub
data mirror, and against the `five-factor-e` PyPI package that was actually
used to score the real response dataset -- zero keying mismatches on
299/300 cross-checked items. Never re-sort any of these by anything other
than the order they were written in. `i1, i10, i100, ...` alphabetical sort
is the textbook silent way this kind of alignment breaks; every function
below that touches item order enforces it explicitly rather than trusting
file or column order.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pandas as pd


# =============================================================================
# SECTION 1 -- THE ITEM POOL
# =============================================================================
# The standard, public-domain IPIP-NEO-300 inventory (Goldberg / Johnson,
# 2014), canonical item order (id 1..300). See item_metadata.csv for the
# domain / facet / keying-direction annotation of every item below.

ITEM_STEMS: list[str] = [
    "Worry about things.",
    "Make friends easily.",
    "Have a vivid imagination.",
    "Trust others.",
    "Complete tasks successfully.",
    "Get angry easily.",
    "Love large parties.",
    "Believe in the importance of art.",
    "Would never cheat on my taxes.",
    "Like order.",
    "Often feel blue.",
    "Take charge.",
    "Experience my emotions intensely.",
    "Make people feel welcome.",
    "Try to follow the rules.",
    "Am easily intimidated.",
    "Am always busy.",
    "Prefer variety to routine.",
    "Am easy to satisfy.",
    "Go straight for the goal.",
    "Often eat too much.",
    "Love excitement.",
    "Like to solve complex problems.",
    "Dislike being the center of attention.",
    "Get chores done right away.",
    "Panic easily.",
    "Radiate joy.",
    "Tend to vote for liberal political candidates.",
    "Sympathize with the homeless.",
    "Avoid mistakes.",
    "Fear for the worst.",
    "Warm up quickly to others.",
    "Enjoy wild flights of fantasy.",
    "Believe that others have good intentions.",
    "Excel in what I do.",
    "Get irritated easily.",
    "Talk to a lot of different people at parties.",
    "Like music.",
    "Stick to the rules.",
    "Like to tidy up.",
    "Dislike myself.",
    "Try to lead others.",
    "Feel others' emotions.",
    "Anticipate the needs of others.",
    "Keep my promises.",
    "Am afraid that I will do the wrong thing.",
    "Am always on the go.",
    "Like to visit new places.",
    "Can't stand confrontations.",
    "Work hard.",
    "Don't know why I do some of the things I do.",
    "Seek adventure.",
    "Love to read challenging material.",
    "Dislike talking about myself.",
    "Am always prepared.",
    "Become overwhelmed by events.",
    "Have a lot of fun.",
    "Believe that there is no absolute right or wrong.",
    "Feel sympathy for those who are worse off than myself.",
    "Choose my words with care.",
    "Am afraid of many things.",
    "Feel comfortable around people.",
    "Love to daydream.",
    "Trust what people say.",
    "Handle tasks smoothly.",
    "Get upset easily.",
    "Enjoy being part of a group.",
    "See beauty in things that others might not notice.",
    "Use flattery to get ahead.",
    "Want everything to be \u201cjust right.\u201d",
    "Am often down in the dumps.",
    "Can talk others into doing things.",
    "Am passionate about causes.",
    "Love to help others.",
    "Pay my bills on time.",
    "Find it difficult to approach others.",
    "Do a lot in my spare time.",
    "Interested in many things.",
    "Hate to seem pushy.",
    "Turn plans into actions.",
    "Do things I later regret.",
    "Love action.",
    "Have a rich vocabulary.",
    "Consider myself an average person.",
    "Start tasks right away.",
    "Feel that I'm unable to deal with things.",
    "Express childlike joy.",
    "Believe that criminals should receive help rather than punishment.",
    "Value cooperation over competition.",
    "Stick to my chosen path.",
    "Get stressed out easily.",
    "Act comfortably with others.",
    "Like to get lost in thought.",
    "Believe that people are basically moral.",
    "Am sure of my ground.",
    "Am often in a bad mood.",
    "Involve others in what I am doing.",
    "Love flowers.",
    "Use others for my own ends.",
    "Love order and regularity.",
    "Have a low opinion of myself.",
    "Seek to influence others.",
    "Enjoy examining myself and my life.",
    "Am concerned about others.",
    "Tell the truth.",
    "Am afraid to draw attention to myself.",
    "Can manage many things at the same time.",
    "Like to begin new things.",
    "Have a sharp tongue.",
    "Plunge into tasks with all my heart.",
    "Go on binges.",
    "Enjoy being part of a loud crowd.",
    "Can handle a lot of information.",
    "Seldom toot my own horn.",
    "Get to work at once.",
    "Can't make up my mind.",
    "Laugh my way through life.",
    "Believe in one true religion.",
    "Suffer from others' sorrows.",
    "Jump into things without thinking.",
    "Get caught up in my problems.",
    "Cheer people up.",
    "Indulge in my fantasies.",
    "Believe in human goodness.",
    "Come up with good solutions.",
    "Lose my temper.",
    "Love surprise parties.",
    "Enjoy the beauty of nature.",
    "Know how to get around the rules.",
    "Do things according to a plan.",
    "Have frequent mood swings.",
    "Take control of things.",
    "Try to understand myself.",
    "Have a good word for everyone.",
    "Listen to my conscience.",
    "Only feel comfortable with friends.",
    "React quickly.",
    "Prefer to stick with things that I know.",
    "Contradict others.",
    "Do more than what's expected of me.",
    "Love to eat.",
    "Enjoy being reckless.",
    "Enjoy thinking about things.",
    "Believe that I am better than others.",
    "Carry out my plans.",
    "Get overwhelmed by emotions.",
    "Love life.",
    "Tend to vote for conservative political candidates.",
    "Am not interested in other people's problems.",
    "Make rash decisions.",
    "Am not easily bothered by things.",
    "Am hard to get to know.",
    "Spend time reflecting on things.",
    "Think that all will be well.",
    "Know how to get things done.",
    "Rarely get irritated.",
    "Prefer to be alone.",
    "Do not like art.",
    "Cheat to get ahead.",
    "Often forget to put things back in their proper place.",
    "Feel desperate.",
    "Wait for others to lead the way.",
    "Seldom get emotional.",
    "Look down on others.",
    "Break rules.",
    "Stumble over my words.",
    "Like to take it easy.",
    "Dislike changes.",
    "Love a good fight.",
    "Set high standards for myself and others.",
    "Rarely overindulge.",
    "Act wild and crazy.",
    "Am not interested in abstract ideas.",
    "Think highly of myself.",
    "Find it difficult to get down to work.",
    "Remain calm under pressure.",
    "Look at the bright side of life.",
    "Believe that too much tax money goes to support artists.",
    "Tend to dislike soft-hearted people.",
    "Like to act on a whim.",
    "Am relaxed most of the time.",
    "Often feel uncomfortable around others.",
    "Seldom daydream.",
    "Distrust people.",
    "Misjudge situations.",
    "Seldom get mad.",
    "Want to be left alone.",
    "Do not like poetry.",
    "Put people under pressure.",
    "Leave a mess in my room.",
    "Feel that my life lacks direction.",
    "Keep in the background.",
    "Am not easily affected by my emotions.",
    "Am indifferent to the feelings of others.",
    "Break my promises.",
    "Am not embarrassed easily.",
    "Like to take my time.",
    "Don't like the idea of change.",
    "Yell at people.",
    "Demand quality.",
    "Easily resist temptations.",
    "Willing to try anything once.",
    "Avoid philosophical discussions.",
    "Have a high opinion of myself.",
    "Waste my time.",
    "Can handle complex problems.",
    "Laugh aloud.",
    "Believe laws should be strictly enforced.",
    "Believe in an eye for an eye.",
    "Rush into things.",
    "Am not easily disturbed by events.",
    "Avoid contacts with others.",
    "Do not have a good imagination.",
    "Suspect hidden motives in others.",
    "Don't understand things.",
    "Am not easily annoyed.",
    "Don't like crowded events.",
    "Do not enjoy going to art museums.",
    "Pretend to be concerned for others.",
    "Leave my belongings around.",
    "Seldom feel blue.",
    "Have little to say.",
    "Rarely notice my emotional reactions.",
    "Make people feel uncomfortable.",
    "Get others to do my duties.",
    "Am comfortable in unfamiliar situations.",
    "Like a leisurely lifestyle.",
    "Am a creature of habit.",
    "Insult people.",
    "Am not highly motivated to succeed.",
    "Am able to control my cravings.",
    "Seek danger.",
    "Have difficulty understanding abstract ideas.",
    "Know the answers to many questions.",
    "Need a push to get started.",
    "Know how to cope.",
    "Amuse my friends.",
    "Believe that we coddle criminals too much.",
    "Try not to think about the needy.",
    "Do crazy things.",
    "Don't worry about things that have already happened.",
    "Am not really interested in others.",
    "Seldom get lost in thought.",
    "Am wary of others.",
    "Have little to contribute.",
    "Keep my cool.",
    "Avoid crowds.",
    "Do not like concerts.",
    "Take advantage of others.",
    "Am not bothered by messy people.",
    "Feel comfortable with myself.",
    "Don't like to draw attention to myself.",
    "Experience very few emotional highs and lows.",
    "Turn my back on others.",
    "Do the opposite of what is asked.",
    "Am not bothered by difficult social situations.",
    "Let things proceed at their own pace.",
    "Dislike new foods.",
    "Get back at others.",
    "Do just enough work to get by.",
    "Never spend more than I can afford.",
    "Would never go hang gliding or bungee jumping.",
    "Am not interested in theoretical discussions.",
    "Boast about my virtues.",
    "Have difficulty starting tasks.",
    "Readily overcome setbacks.",
    "Am not easily amused.",
    "Believe that we should be tough on crime.",
    "Believe people should fend for themselves.",
    "Act without thinking.",
    "Adapt easily to new situations.",
    "Keep others at a distance.",
    "Have difficulty imagining things.",
    "Believe that people are essentially evil.",
    "Don't see the consequences of things.",
    "Rarely complain.",
    "Seek quiet.",
    "Do not enjoy watching dance performances.",
    "Obstruct others' plans.",
    "Am not bothered by disorder.",
    "Am very pleased with myself.",
    "Hold back my opinions.",
    "Don't understand people who get emotional.",
    "Take no time for others.",
    "Misrepresent the facts.",
    "Am able to stand up for myself.",
    "React slowly.",
    "Am attached to conventional ways.",
    "Hold a grudge.",
    "Put little time and effort into my work.",
    "Never splurge.",
    "Dislike loud music.",
    "Avoid difficult reading material.",
    "Make myself the center of attention.",
    "Postpone decisions.",
    "Am calm even in tense situations.",
    "Seldom joke around.",
    "Like to stand during the national anthem.",
    "Can't stand weak people.",
    "Often make last-minute plans.",
]

assert len(ITEM_STEMS) == 300, f"Expected 300 IPIP-NEO items, found {len(ITEM_STEMS)}"


# =============================================================================
# SECTION 2 -- RAW EMBEDDINGS AND COSINE SIMILARITY
# =============================================================================

def compute_embeddings(texts: list[str], model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    """
    Encode item text into dense vectors with a pretrained sentence-transformer.

    For readers without an NLP background: a "sentence embedding" is a fixed-
    length list of numbers (here, 768 of them) that a neural network produces
    for a piece of text, trained so that semantically similar sentences end
    up as nearby vectors. It's a numeric proxy for meaning, not a set of
    a-priori interpretable dimensions.

    Import is deliberately LOCAL to this function, not at module level. This
    keeps sentence-transformers (a heavy dependency: torch + transformers)
    out of every other code path in this pipeline -- content_blind_
    corrections.py, content_aware_corrections.py, and the --mode empirical
    path below never need it, since they all operate on already-saved
    embeddings or on raw survey responses.

    Returns
    -------
    np.ndarray, shape (n_items, n_dims) -- 768 dims for all-mpnet-base-v2.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    All pairwise cosine similarities: cos(theta) = (u . v) / (|u| |v|).

    IMPORTANT, because it has already caused confusion once in this project:
    cosine similarity ranges over [-1, +1], NOT [0, 1]. The [0, 1] restriction
    only holds for vectors with exclusively non-negative components (e.g.
    raw word counts). Dense transformer embeddings are real-valued in every
    dimension -- roughly half of every embedding's components are negative
    for all-mpnet-base-v2 -- so negative similarity is mathematically normal,
    not a bug, and is in fact the target outcome for reverse-keyed pairs
    once a correction is working.

    Returns
    -------
    np.ndarray, shape (n_items, n_items), symmetric, diagonal == 1.0.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normalized = embeddings / norms
    return np.dot(normalized, normalized.T)


def save_embeddings_csv(item_stems: list[str], embeddings: np.ndarray, filename: str) -> None:
    header = ["item"] + [f"dim_{i}" for i in range(embeddings.shape[1])]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for item, emb in zip(item_stems, embeddings):
            writer.writerow([item] + emb.tolist())


def save_similarity_csv(similarity: np.ndarray, filename: str) -> None:
    """Bare matrix, no header/index -- item labels live in embeddings.csv /
    item_metadata.csv and are matched back on by row position, per the item
    order guarantee documented at the top of this file."""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in similarity:
            writer.writerow(row.tolist())


def load_embeddings(embeddings_path: str) -> tuple[list[str], np.ndarray]:
    """
    Load a previously saved embeddings.csv. Every downstream script calls
    this instead of compute_embeddings() -- embeddings are computed exactly
    once (`python foundations.py --mode embeddings`) and reused from disk
    everywhere else in the pipeline.
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"{embeddings_path} not found. Run `python foundations.py --mode embeddings` "
            f"first -- every later step reads the saved baseline; none of them recompute it."
        )
    df = pd.read_csv(embeddings_path)
    items = df["item"].tolist()
    embeddings = df.drop(columns=["item"]).to_numpy()
    return items, embeddings


def load_item_metadata(path: str) -> pd.DataFrame:
    """domain (N/E/O/A/C), facet_code (e.g. 'N1'), keyed ('plus'/'minus')
    per item. See item_metadata.csv -- verified two independent ways against
    the actual IPIP-NEO-300 scoring key, zero mismatches on 299/300 checked."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- required for content-aware corrections.")
    return pd.read_csv(path).set_index("item")


def load_reverse_keyed_pairs(path: str) -> list[tuple[str, str]]:
    """696 systematic within-facet plus/minus item pairs. Every plus-keyed
    item paired against every minus-keyed item in the same one of the 30
    facets, derived directly from the verified keying above -- not hand-
    picked, and not the same 18-item pilot set used earlier in this project."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- required for sign-accuracy evaluation.")
    df = pd.read_csv(path)
    return list(zip(df["item_a"], df["item_b"]))


def load_same_keyed_control_pairs(path: str) -> list[tuple[str, str]]:
    """
    Same-facet pairs that share the SAME keying direction (both plus, or
    both minus) -- the artifact-control counterpart to the reverse-keyed
    pairs above. These experience the identical mechanical group-centering
    pull described in group_center()'s docstring, but should genuinely stay
    POSITIVELY related, since they measure the same facet in the same
    direction. Compare this set's results against the reverse-keyed set's
    results, not either one in isolation -- see content_aware_corrections.py.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- required for artifact-controlled evaluation.")
    df = pd.read_csv(path)
    return list(zip(df["item_a"], df["item_b"]))


def check_similarity_matrix(similarity: np.ndarray, label: str = "") -> None:
    """
    Cheap sanity check run after computing every condition in this project.
    Catches the most common way a similarity matrix silently goes wrong:
    wrong shape, broken symmetry (a bug in how a correction was applied), or
    NaNs (typically a division by a near-zero norm). Does not raise on the
    diagonal-equals-1 property, because that CAN legitimately fail for a
    group-centered condition if some item's embedding happens to exactly
    equal its own group mean (a zero vector) -- vanishingly unlikely with
    continuous embeddings, but a hard assert would be the wrong failure mode
    for a rare, non-catastrophic edge case; a warning is printed instead.
    """
    n_rows, n_cols = similarity.shape
    tag = f"[{label}] " if label else ""
    assert n_rows == n_cols, f"{tag}similarity matrix is not square: {similarity.shape}"
    assert np.allclose(similarity, similarity.T, atol=1e-6), f"{tag}similarity matrix is not symmetric"
    assert not np.isnan(similarity).any(), f"{tag}similarity matrix contains NaNs"
    diag = np.diag(similarity)
    if not np.allclose(diag, 1.0, atol=1e-4):
        off = np.where(~np.isclose(diag, 1.0, atol=1e-4))[0]
        print(f"WARNING: {tag}{len(off)} item(s) have self-similarity != 1.0 "
              f"(likely a near-zero-norm vector after correction), indices: {off[:5]}...")


def visualize_similarity_matrix(embeddings_path: str, similarity_path: str, output_path: str) -> str:
    """Render a similarity matrix as a heatmap for visual sanity-checking.
    Fixed color scale (vmin=-1, vmax=1) across every condition in this
    project so heatmaps are visually comparable to each other."""
    import matplotlib.pyplot as plt

    items = pd.read_csv(embeddings_path, usecols=["item"])["item"].tolist()
    similarity_df = pd.read_csv(similarity_path, header=None)
    similarity_df.index = items
    similarity_df.columns = items

    plt.figure(figsize=(24, 24))
    plt.imshow(similarity_df.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Cosine similarity")
    plt.xticks(range(len(items)), items, rotation=90, fontsize=6)
    plt.yticks(range(len(items)), items, fontsize=6)
    plt.title(f"Item-to-item cosine similarity\n({os.path.basename(similarity_path)})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


# =============================================================================
# SECTION 3 -- THE CORRECTION TOOLBOX
# =============================================================================
# Every correction is specified in geometric-correction-methodology.md using
# this notation:  x_i = raw embedding of item i;  x_bar = mean of all x_i;
# delta_i = x_i - x_bar;  C = covariance of {delta_i} = V Lambda V^T.
# The functions below are the literal implementation of that notation --
# variable names match the doc on purpose, so the two can be read side by
# side.

def mean_center(embeddings: np.ndarray) -> np.ndarray:
    """
    Global mean-centering:  delta_i = x_i - x_bar

    Psychometric intuition
    -----------------------
    Every item embedding blends item-specific content with a component
    shared by ~every item in the pool -- generic questionnaire-item
    register, topic, syntax. That shared component is exactly what inflates
    similarity between any two items regardless of their actual content, and
    is the single most direct explanation for why reverse-keyed pairs come
    out positively related in raw embedding space. Subtracting the
    population mean from every item removes precisely this shared
    component, by construction -- x_bar IS the average of everything every
    item has in common.

    This is also the FIRST STEP of every other correction below (drop-top
    and ZCA both start from delta, never from raw x_i), and stands on its
    own as the simplest possible correction -- the natural baseline to beat.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_items, n_dims)

    Returns
    -------
    np.ndarray, shape (n_items, n_dims) -- delta, the mean-centered embeddings
    """
    return embeddings - embeddings.mean(axis=0, keepdims=True)


def eigendecompose(centered_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Covariance eigendecomposition of already-mean-centered embeddings:
        C = (1/(n-1)) * delta^T @ delta = V Lambda V^T

    Eigenvalues returned in DESCENDING order (lambda_1 >= lambda_2 >= ...),
    matching the methodology doc's notation, so eigenvectors[:, 0] is the
    direction of greatest variance in the (already-centered) embedding
    cloud -- "PC1" in the usual PCA sense.

    Numerical note: with n=300 items and d=768 embedding dimensions, this
    covariance matrix has rank <= 299 (n-1, since it's built from
    mean-centered data). Eigenvalues beyond the true rank are numerically
    ~0, not exactly 0, due to floating point -- this matters for
    zca_whiten() below, which divides by these eigenvalues and must
    regularize accordingly.

    Returns
    -------
    (eigenvalues, eigenvectors) : (np.ndarray shape (d,), np.ndarray shape (d, d))
    """
    n = centered_embeddings.shape[0]
    cov = (centered_embeddings.T @ centered_embeddings) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def drop_top_pcs(embeddings: np.ndarray, m: int = 1) -> np.ndarray:
    """
    All-but-the-top / "drop-top" (hard, global correction):
        g_i^DT = delta_i - sum_{l=1}^{m} (delta_i . v_l) v_l

    Psychometric intuition
    -----------------------
    A competing hypothesis to plain mean-centering: perhaps the shared
    "nuisance" signal isn't just a constant additive offset (the population
    mean), but is concentrated along a small number of specific DIRECTIONS
    of high variance in the embedding cloud. If so, deleting those top
    principal-component directions entirely -- rather than just recentering
    -- should remove more of the distortion. This is a real, competing
    hypothesis about where the distortion lives, not a strictly-stronger
    version of mean-centering.

    m (how many top components to remove) is the one free parameter, and
    should be chosen from the eigenvalue spectrum's elbow BEFORE looking at
    any evaluation metric -- picking m to maximize sign-accuracy on the
    evaluation set would be fitting the correction to the test.

    Empirical note from this project's actual data: the eigenvalue spectrum
    has no sharp elbow (PC1 explains only ~6% of variance), which is why
    drop-top and plain mean-centering perform almost identically here --
    the distortion in THIS dataset behaves like a mean-offset, not a
    concentrated top-direction anisotropy. m=1 remains the theory-motivated
    default (a single constant offset implies a single shared direction to
    remove); this file's __main__ can sweep m for sensitivity, but m should
    never be chosen post-hoc against the evaluation metric.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_items, n_dims)
    m : int, number of top principal components to remove entirely

    Returns
    -------
    np.ndarray, shape (n_items, n_dims) -- g^DT
    """
    delta = mean_center(embeddings)
    _, eigenvectors = eigendecompose(delta)
    top_v = eigenvectors[:, :m]              # (n_dims, m)
    projections = delta @ top_v              # (n_items, m)
    removed = projections @ top_v.T          # (n_items, n_dims)
    return delta - removed


def zca_whiten(embeddings: np.ndarray, alpha: float = 0.5, eps_frac: float = 0.01) -> np.ndarray:
    """
    Whitening / "full-rescale" (soft, global correction):
        g_i^FR = sum_l [(delta_i . v_l) / (lambda_l + eps)^(alpha/2)] v_l,
        eps = eps_frac * lambda_1

    Psychometric intuition
    -----------------------
    The competing hypothesis to drop-top: rather than the nuisance signal
    being concentrated in a few directions (which you'd delete), it's
    DISTRIBUTED across many directions with unequal variance (which you'd
    rescale). Every direction gets divided by its own (regularized)
    standard deviation, so no single direction of variation is allowed to
    dominate the similarity calculation -- a soft correction rather than
    drop-top's hard deletion.

    alpha=0.5 (this project's default) is a partial shrink; alpha=1.0
    recovers textbook ZCA whitening (every retained direction rescaled to
    unit variance). alpha is the parameter to sweep for sensitivity.

    Numerical note (the practically important part of this function):
    with n=300 items < d=768 dims, the covariance is rank-deficient
    (rank <= 299 -- see eigendecompose() above), so most eigenvalues are
    numerically ~0. Dividing by an unregularized near-zero eigenvalue
    would amplify pure sampling noise in that direction, not signal --
    `eps` puts a floor under every denominator, sized relative to the
    largest eigenvalue rather than as an arbitrary absolute constant.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n_items, n_dims)
    alpha : float, rescaling exponent (0.5 default; 1.0 = full ZCA)
    eps_frac : float, regularization floor as a fraction of lambda_1

    Returns
    -------
    np.ndarray, shape (n_items, n_dims) -- g^FR
    """
    delta = mean_center(embeddings)
    eigenvalues, eigenvectors = eigendecompose(delta)
    eps = eps_frac * eigenvalues[0]
    scale = (eigenvalues + eps) ** (alpha / 2)
    scale = np.where(scale < 1e-12, 1e-12, scale)   # belt-and-suspenders floor
    coords = delta @ eigenvectors                    # coordinates in the eigenbasis
    rescaled = coords / scale                        # broadcast over the d eigen-directions
    return rescaled @ eigenvectors.T                 # back to the original embedding basis


def group_center(vectors: np.ndarray, items: list[str], group_of: dict[str, str]) -> np.ndarray:
    """
    Hierarchical (group) centering:
        h_i = g_i - mean_{j : group(j) = group(i)}(g_j)

    Psychometric intuition
    -----------------------
    This is where the second half of the project (content_aware_
    corrections.py) departs from everything above: instead of subtracting
    the GRAND mean across all 300 items (mean_center, above), subtract the
    mean of just the item's own group -- its Big Five domain, or its
    specific 30-way facet. This targets bias that is local to a construct
    rather than global to the whole item pool, at the cost of requiring the
    item's true label as an input to the correction itself. That is a
    meaningful methodological shift, not a free upgrade -- see
    content_aware_corrections.py's docstring for the full discussion of
    what this trades away (content-blindness) for what it buys (much
    higher sign-accuracy, empirically).

    A MECHANICAL EFFECT THIS FUNCTION INTRODUCES, AND WHY IT MATTERS
    --------------------------------------------------------------------
    Centering within a group of size k forces that group's residuals to
    sum to exactly zero -- pure arithmetic. For iid vectors, this
    constraint alone induces an expected pairwise cosine similarity of
    -1/(k-1) between ANY two members of the group, independent of content.
    For facets (k=10 items each), that floor is -1/9 = -0.111 -- verified
    directly in this project via a random-noise stress test that reproduced
    -0.111 almost exactly with zero real signal present. See
    mechanical_centering_floor() below.

    This means facet-level sign-accuracy numbers are NOT purely semantic:
    part of any pair's negativity comes from group size alone. The
    evaluation in content_aware_corrections.py controls for this directly
    by comparing reverse-keyed pairs against a same-facet, SAME-keyed
    control set that experiences the identical mechanical pull but should
    genuinely stay positive -- the GAP between the two, not the raw
    negative number alone, is the artifact-controlled evidence for real
    signal. Report both, not just the raw sign-accuracy figure.

    `vectors` is whatever representation is being centered -- delta for the
    "blanket" conditions in this project, or g^DT / g^FR for the "nested"
    conditions -- the grouping logic is identical either way, only the
    input differs.

    Parameters
    ----------
    vectors : np.ndarray, shape (n_items, n_dims)
    items : list[str], item text in the SAME row order as `vectors`
    group_of : dict[str, str], item text -> group label (domain or facet code)

    Returns
    -------
    np.ndarray, shape (n_items, n_dims) -- h, group-centered
    """
    df = pd.DataFrame(vectors, index=items)
    groups = pd.Series([group_of[it] for it in items], index=items)
    group_means = df.groupby(groups).transform("mean")
    return (df - group_means).to_numpy()


def mechanical_centering_floor(group_size: int) -> float:
    """
    The expected pairwise cosine similarity that group-centering induces
    between ANY two members of a group of this size, from the sum-to-zero
    constraint alone -- zero real signal required. Derivation: for iid
    x_1..x_k with mean x_bar, Cov(x_i - x_bar, x_j - x_bar) = -sigma^2/k for
    i != j, giving correlation -1/(k-1) after normalizing by
    Var(x_i - x_bar) = sigma^2 (1 - 1/k). Verified empirically in this
    project via a random-noise stress test (see project history) that
    reproduced this exact prediction on facet-sized groups (k=10) to three
    decimal places.

    Use this as a reference point when reading any result produced by
    group_center(), never as a thing to report on its own -- it's the floor
    a result needs to be understood RELATIVE TO, not a result itself.
    """
    if group_size <= 1:
        raise ValueError("group_size must be >= 2 for a pairwise correlation to be defined")
    return -1.0 / (group_size - 1)


def splice_by_scope(
    sim_global: np.ndarray,
    sim_domain: np.ndarray,
    sim_facet: np.ndarray,
    items: list[str],
    domain_of: dict[str, str],
    facet_of: dict[str, str],
) -> np.ndarray:
    """
    Combine three similarity matrices into one final matrix by choosing,
    per item PAIR, the representation matched to that pair's scope:
        between-domain (different Big Five domains)   -> sim_global
        within-domain, between-facet                  -> sim_domain
        within-facet (the finest scope)                -> sim_facet

    Psychometric intuition
    -----------------------
    This is the mechanism behind this project's "nested" conditions, and
    the reason they outperform blanket group-centering empirically, not
    just theoretically: each level of centering should remove exactly the
    shared bias visible AT that level, and no finer. Applying facet-level
    centering to a pair of items from entirely different domains would
    remove information those two items never shared bias on in the first
    place -- using facet-centering only where two items are actually in the
    same facet avoids that overcorrection, while still applying it exactly
    where the residual bias this project set out to fix (facet-local
    collapse of reverse-keyed pairs) actually lives.

    Returns
    -------
    np.ndarray, shape (n_items, n_items) -- the spliced similarity matrix
    """
    domains = np.array([domain_of[it] for it in items])
    facets = np.array([facet_of[it] for it in items])

    same_domain = domains[:, None] == domains[None, :]
    same_facet = facets[:, None] == facets[None, :]

    spliced = sim_global.copy()
    spliced[same_domain & ~same_facet] = sim_domain[same_domain & ~same_facet]
    spliced[same_facet] = sim_facet[same_facet]
    return spliced


# =============================================================================
# SECTION 4 -- MEASURING SUCCESS
# =============================================================================

def mean_offdiag_similarity(similarity_matrix: np.ndarray) -> float:
    """
    Average cosine similarity across every item pair (excluding each item
    with itself). This is a coarse, easy-to-misread number -- it captures
    whether the shared POSITIVE baseline has been removed, and nothing more.
    A correction can drive this to exactly zero while still getting every
    single reverse-keyed pair's sign wrong; see reverse_keyed_diagnostics()
    below for the metric that actually answers this project's question.
    """
    n = similarity_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(similarity_matrix[mask].mean())


def structure_preservation(reference_similarity: np.ndarray, corrected_similarity: np.ndarray) -> float:
    """
    Pearson correlation between two similarity matrices (off-diagonal
    entries only) -- a check against a specific failure mode: a correction
    that "solves" reverse-keyed pairs by scrambling the ENTIRE similarity
    structure isn't a real win. High correlation with the reference (usually
    raw) = gentle reshaping; low = substantial reorganization. Neither value
    is automatically good or bad -- the nested conditions in this project
    deliberately trade some of this for sign-accuracy, and the interesting
    empirical question is exactly how much.
    """
    n = reference_similarity.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.corrcoef(reference_similarity[mask], corrected_similarity[mask])[0, 1])


def reverse_keyed_diagnostics(
    similarity_matrix: np.ndarray,
    items: list[str],
    pairs: list[tuple[str, str]],
    empirical_signs: dict[tuple[str, str], float] | None = None,
) -> dict:
    """
    THE central metric of this project: for a list of reverse-keyed item
    pairs, does the corrected similarity actually come out with the
    expected sign?

    By default, "expected sign" is negative -- the theoretical assumption
    that a reverse-keyed pair should be strongly inversely related. If
    `empirical_signs` is supplied (built from real response data -- see
    Section 5 below), sign-accuracy is checked against the REAL sign of
    that pair's correlation instead. This distinction matters: some
    reverse-keyed pairs empirically correlate only weakly negative, or even
    slightly positive (imperfect item wording, acquiescence bias) --
    theoretical sign-accuracy can't tell "the correction got this pair's
    sign wrong" apart from "the theoretical assumption about this pair was
    wrong to begin with." Empirical sign-accuracy can.

    Also returns the mean similarity across every OTHER off-diagonal pair,
    as a control: this is what separates "the correction specifically fixed
    the pairs that matter" from "the correction just uniformly deflated
    every similarity in the matrix," which mean_offdiag_similarity() alone
    cannot distinguish.

    Returns
    -------
    dict with keys: n_pairs, mean_reverse_keyed_sim, reverse_keyed_sign_accuracy,
    mean_all_other_sim, missing_pairs
    """
    index = {item: i for i, item in enumerate(items)}
    pair_sims, expected_signs, missing = [], [], []
    for a, b in pairs:
        if a not in index or b not in index:
            missing.append((a, b))
            continue
        pair_sims.append(similarity_matrix[index[a], index[b]])
        if empirical_signs is not None:
            expected_signs.append(empirical_signs.get((a, b), empirical_signs.get((b, a), -1.0)))
        else:
            expected_signs.append(-1.0)
    pair_sims = np.array(pair_sims)
    expected_signs = np.array(expected_signs)

    n = similarity_matrix.shape[0]
    all_sims = similarity_matrix[~np.eye(n, dtype=bool)]
    matches = float((np.sign(pair_sims) == expected_signs).mean()) if len(pair_sims) else float("nan")

    return {
        "n_pairs": len(pair_sims),
        "mean_reverse_keyed_sim": float(pair_sims.mean()) if len(pair_sims) else float("nan"),
        "reverse_keyed_sign_accuracy": matches,
        "mean_all_other_sim": float(all_sims.mean()),
        "missing_pairs": missing,
    }


def build_comparison_summary(
    similarities: dict[str, np.ndarray],
    items: list[str],
    pairs: list[tuple[str, str]] | None = None,
    reference_name: str | None = None,
    empirical_signs: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    """
    Shared comparison-table builder used by both content_blind_corrections.py
    and content_aware_corrections.py, so the metric definitions and table
    format are identical regardless of which conditions are being compared.
    """
    reference_name = reference_name or next(iter(similarities))
    reference_sim = similarities[reference_name]

    rows = []
    for name, sim in similarities.items():
        check_similarity_matrix(sim, label=name)
        row = {
            "condition": name,
            "mean_offdiag_sim": mean_offdiag_similarity(sim),
            "structure_corr_vs_reference": 1.0 if name == reference_name else structure_preservation(reference_sim, sim),
        }
        if pairs:
            diag = reverse_keyed_diagnostics(sim, items, pairs, empirical_signs=empirical_signs)
            row["mean_reverse_keyed_sim"] = diag["mean_reverse_keyed_sim"]
            row["reverse_keyed_sign_accuracy"] = diag["reverse_keyed_sign_accuracy"]
            row["mean_all_other_sim"] = diag["mean_all_other_sim"]
        rows.append(row)

    return pd.DataFrame(rows).set_index("condition")


def plot_comparison(summary: pd.DataFrame, output_path: str) -> str:
    import matplotlib.pyplot as plt

    conditions = summary.index.tolist()
    has_pairs = "mean_reverse_keyed_sim" in summary.columns

    fig, axes = plt.subplots(1, 2 if has_pairs else 1, figsize=(12 if has_pairs else 5.5, 5))
    if not has_pairs:
        axes = [axes]

    axes[0].bar(conditions, summary["mean_offdiag_sim"], color="steelblue")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Mean off-diagonal cosine similarity\n(all item pairs)")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_xticks(range(len(conditions)))
    axes[0].set_xticklabels(conditions, rotation=45, ha="right", fontsize=8)

    if has_pairs:
        width = 0.38
        x = np.arange(len(conditions))
        axes[1].bar(x - width / 2, summary["mean_reverse_keyed_sim"], width,
                    label="reverse-keyed pairs", color="indianred")
        axes[1].bar(x + width / 2, summary["mean_all_other_sim"], width,
                    label="all other pairs", color="lightgray")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(conditions, rotation=45, ha="right", fontsize=8)
        axes[1].set_title("Reverse-keyed pairs vs. everything else")
        axes[1].set_ylabel("Cosine similarity")
        axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


# =============================================================================
# SECTION 5 -- VALIDATION AGAINST REAL HUMAN RESPONSE DATA
# =============================================================================
# Everything above evaluates corrections against a THEORETICAL assumption
# (reverse-keyed pairs should be negatively related). This section computes
# the actual ground truth: real correlations between real people's real
# answers to these exact 300 items, from ecorbari/IPIP300-SCORES on Hugging
# Face (145K+ participants). Embedding-only diagnostics are a proxy; this is
# the thing the proxy is a proxy FOR.

ITEM_RESPONSE_COLUMNS: list[str] = [f"i{i}" for i in range(1, 301)]  # canonical order, do not sort


def compute_empirical_correlation_matrix(item_responses: pd.DataFrame, min_valid_frac: float = 0.5) -> tuple[np.ndarray, int]:
    """
    Pearson correlation matrix over real Likert item responses.

    Why Pearson and not polychoric: polychoric correlation is the more
    textbook-correct choice for 5-point ordinal Likert data, but there is no
    mainstream, fast Python implementation for a 300x300 matrix (this is
    routine in R via `psych`/`polycor`). Pearson on 5-point Likert data is a
    standard, well-precedented simplification in practice; flagged here
    explicitly rather than silently -- a technical reviewer should be able
    to see this decision was made deliberately, not by default.

    Parameters
    ----------
    item_responses : pd.DataFrame, columns i1..i300 IN CANONICAL ORDER
    min_valid_frac : float, drop respondents missing more than this
        fraction of their 300 item responses before computing correlations

    Returns
    -------
    (correlation_matrix, n_respondents_used) : (np.ndarray (300, 300), int)
    """
    before = len(item_responses)
    item_responses = item_responses.dropna(thresh=int(min_valid_frac * item_responses.shape[1]))
    dropped = before - len(item_responses)
    if dropped:
        print(f"Dropped {dropped} respondent(s) with too many missing item responses.")

    print(f"Computing Pearson correlation over {len(item_responses)} respondents "
          f"x {item_responses.shape[1]} items...")
    corr = item_responses.corr(method="pearson").to_numpy()
    return corr, len(item_responses)


def load_empirical_correlation_matrix(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python foundations.py --mode empirical --input <file>` "
            f"wherever the real response data lives, then copy the resulting "
            f"empirical_correlation_matrix.csv into this directory."
        )
    return pd.read_csv(path, header=None).to_numpy()


def empirical_pair_diagnostics(empirical_matrix: np.ndarray, items: list[str], pairs: list[tuple[str, str]]) -> dict:
    """
    What does the ground truth itself say about the reverse-keyed pairs,
    independent of any embedding correction? Important context before
    trusting any sign-accuracy number: if a meaningful fraction of
    "reverse-keyed" pairs are NOT actually negatively correlated in real
    data, that is informative about the theoretical assumption itself, not
    only about how well a correction recovers it.
    """
    index = {item: i for i, item in enumerate(items)}
    vals = np.array([empirical_matrix[index[a], index[b]] for a, b in pairs if a in index and b in index])
    n = empirical_matrix.shape[0]
    all_vals = empirical_matrix[~np.eye(n, dtype=bool)]
    return {
        "n_pairs": len(vals),
        "mean_empirical_reverse_keyed_corr": float(vals.mean()),
        "frac_reverse_keyed_actually_negative": float((vals < 0).mean()),
        "mean_empirical_all_other_corr": float(all_vals.mean()),
    }


def build_empirical_sign_lookup(
    empirical_matrix: np.ndarray, items: list[str], pairs: list[tuple[str, str]]
) -> dict[tuple[str, str], float]:
    """Real sign per pair, for use as `empirical_signs` in
    reverse_keyed_diagnostics() -- replaces the blanket "-1" theoretical
    assumption with the actual empirical ground truth."""
    index = {item: i for i, item in enumerate(items)}
    signs: dict[tuple[str, str], float] = {}
    for a, b in pairs:
        if a not in index or b not in index:
            continue
        val = empirical_matrix[index[a], index[b]]
        signs[(a, b)] = float(np.sign(val)) if val != 0 else -1.0
    return signs


# =============================================================================
# ENTRY POINT
# =============================================================================

def _run_embeddings_mode(base_dir: str, force: bool, plot: bool) -> None:
    embeddings_path = os.path.join(base_dir, "embeddings.csv")
    similarity_path = os.path.join(base_dir, "similarity_matrix_1_raw.csv")

    if not force and os.path.exists(embeddings_path) and os.path.exists(similarity_path):
        print(f"Already exists: {embeddings_path}, {similarity_path} (pass --force to recompute).")
    else:
        embeddings = compute_embeddings(ITEM_STEMS)
        similarity = cosine_similarity_matrix(embeddings)
        check_similarity_matrix(similarity, label="1_raw")
        save_embeddings_csv(ITEM_STEMS, embeddings, embeddings_path)
        save_similarity_csv(similarity, similarity_path)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Saved -> {embeddings_path}")
        print(f"Saved -> {similarity_path}")

    if plot:
        heatmap_path = os.path.join(base_dir, "heatmap_1_raw.png")
        visualize_similarity_matrix(embeddings_path, similarity_path, heatmap_path)
        print(f"Saved -> {heatmap_path}")


def _run_empirical_mode(base_dir: str, input_path: str, output_name: str) -> None:
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path, columns=ITEM_RESPONSE_COLUMNS)
    else:
        df = pd.read_csv(input_path, usecols=ITEM_RESPONSE_COLUMNS)
    missing = [c for c in ITEM_RESPONSE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} expected item column(s), e.g. {missing[:5]}.")
    df = df[ITEM_RESPONSE_COLUMNS]  # enforce canonical numeric order explicitly

    corr, n_used = compute_empirical_correlation_matrix(df)
    output_path = os.path.join(base_dir, output_name)
    pd.DataFrame(corr).to_csv(output_path, index=False, header=False)

    print(f"\nSaved {corr.shape[0]}x{corr.shape[1]} empirical correlation matrix -> {output_path}")
    print(f"(based on N={n_used} respondents)")
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    print(f"Mean off-diagonal empirical correlation: {off_diag.mean():+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Foundations: raw embeddings, correction toolbox, and empirical validation."
    )
    parser.add_argument("--mode", choices=["embeddings", "empirical"], default="embeddings")
    parser.add_argument("--force", action="store_true", help="[embeddings mode] recompute even if cached.")
    parser.add_argument("--skip-plot", action="store_true", help="[embeddings mode] skip the heatmap.")
    parser.add_argument("--input", type=str, default=None,
                         help="[empirical mode] path to response data (.parquet or .csv) with i1..i300 columns.")
    parser.add_argument("--output", type=str, default="empirical_correlation_matrix.csv",
                         help="[empirical mode] output filename.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.mode == "embeddings":
        _run_embeddings_mode(base_dir, force=args.force, plot=not args.skip_plot)
    else:
        if not args.input:
            parser.error("--mode empirical requires --input <path to response data>")
        _run_empirical_mode(base_dir, args.input, args.output)


if __name__ == "__main__":
    main()
