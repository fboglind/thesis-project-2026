"""fas_scorer.py

Scoring for Phonemic Verbal Fluency (FAS) test responses.

This module exposes two complementary public APIs:

- :func:`score_fas_counts` — the count-based scorer (Tallberg et al.
  2008, Borkowski et al. 1967): tracks per-letter totals, proper-noun
  flags, repetitions, and total FAS score. Originally the only scorer
  in the project; kept under this name to avoid breaking older callers.

- :func:`cluster_letter` and :func:`score_fas` — Phase 4c clustering
  scorer (Troyer 1997, chain method per Pakhomov 2016) using the
  pre-registered Swedish-orthographic rules in
  :mod:`thesis_project.scoring.phonemic_rules`. The clustering scorer
  expects pre-filtered word lists (proper-noun markers stripped,
  repetitions removed) and is responsible for the per-letter and
  aggregated cluster metrics plus the Linz MWF feature.

The two APIs are kept separate so the cluster scorer can be tested in
isolation; the FAS pipeline calls both.
"""

from __future__ import annotations

import math

import numpy as np

from .phonemic_rules import linked

# Re-export the original count-based scorer name so legacy callers
# still resolve. The new clustering ``score_fas`` is defined below.
# (The two share the module but operate on different inputs.)


# ──────────────────────────────────────────────────────
# Edit distance helper (no external dependency)
# ──────────────────────────────────────────────────────

def _levenshtein(s1: str, s2: str) -> int:
    """Pure-Python Levenshtein edit distance."""
    if s1 == s2:
        return 0
    len1, len2 = len(s1), len(s2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    prev = list(range(len2 + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[len2]


def _normalized_edit_distance(s1: str, s2: str) -> float:
    """Levenshtein distance normalized by max string length.

    Returns a value in [0, 1]: 0 = identical, 1 = maximally different.
    """
    if not s1 and not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    return _levenshtein(s1, s2) / max_len


# ──────────────────────────────────────────────────────
# Per-letter scoring
# ──────────────────────────────────────────────────────

def _score_letter(responses: list[str], flagged_words: set[str]) -> dict:
    """Compute metrics for a single letter's response sequence.

    Args:
        responses: All response slots for this letter, including empty strings
            for time points where no word was produced.
        flagged_words: Set of lowercased words that were angle-bracket flagged
            (proper nouns or rule violations).

    Returns:
        dict with keys: total_words, proper_nouns, repetitions, valid_words,
        mean_edit_distance.
    """
    # Only count non-empty slots
    produced = [w for w in responses if w]

    total = len(produced)
    proper_nouns = sum(1 for w in produced if w in flagged_words)
    repetitions = total - len(set(produced))
    valid = max(0, total - proper_nouns - repetitions)

    if len(produced) >= 2:
        dists = [
            _normalized_edit_distance(produced[i], produced[i + 1])
            for i in range(len(produced) - 1)
        ]
        mean_edit_dist: float = float(np.mean(dists))
    else:
        mean_edit_dist = float("nan")

    return {
        "total_words": total,
        "proper_nouns": proper_nouns,
        "repetitions": repetitions,
        "valid_words": valid,
        "mean_edit_distance": mean_edit_dist,
    }


# ──────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────

def score_fas_counts(
    responses_f: list[str],
    responses_a: list[str],
    responses_s: list[str],
    flagged_errors: list[str] | None = None,
) -> dict:
    """Compute count-based FAS metrics for a single participant.

    This is the original scorer (Tallberg/Borkowski lineage). It does
    its own proper-noun flagging and repetition counting. For the
    Phase 4c clustering metrics, see :func:`score_fas` below — that
    function expects already-filtered word lists.

    Args:
        responses_f: Response slots for letter F (empty string = no word at
            that time point, not the same as the participant stopping).
        responses_a: Response slots for letter A.
        responses_s: Response slots for letter S.
        flagged_errors: List of original angle-bracket-flagged forms from the
            loader, e.g. ['<Anna>', '<Stockholm>']. These are counted as
            proper nouns and excluded from the valid-word count.

    Returns:
        dict with keys:

        Per-letter counts:
            total_f/a/s, valid_f/a/s, proper_nouns_f/a/s, repetitions_f/a/s

        Aggregate:
            total_fas_score    — sum of valid words across F + A + S
            proper_nouns_count — total flagged errors across all letters
            repetitions_count  — total repetitions across all letters
            letter_asymmetry   — std of [valid_f, valid_a, valid_s]

        Exploratory phonemic:
            mean_edit_distance_f/a/s — mean normalised Levenshtein distance
                between consecutive responses within each letter
    """
    if flagged_errors is None:
        flagged_errors = []

    # Build lookup set from angle-bracket forms, e.g. '<Anna>' → 'anna'
    flagged_words: set[str] = {
        w[1:-1].lower()
        for w in flagged_errors
        if w.startswith("<") and w.endswith(">")
    }

    f = _score_letter(responses_f, flagged_words)
    a = _score_letter(responses_a, flagged_words)
    s = _score_letter(responses_s, flagged_words)

    total_fas_score = f["valid_words"] + a["valid_words"] + s["valid_words"]
    letter_asymmetry = float(np.std([f["valid_words"], a["valid_words"], s["valid_words"]]))

    return {
        # Per-letter
        "total_f": f["total_words"],
        "total_a": a["total_words"],
        "total_s": s["total_words"],
        "valid_f": f["valid_words"],
        "valid_a": a["valid_words"],
        "valid_s": s["valid_words"],
        "proper_nouns_f": f["proper_nouns"],
        "proper_nouns_a": a["proper_nouns"],
        "proper_nouns_s": s["proper_nouns"],
        "repetitions_f": f["repetitions"],
        "repetitions_a": a["repetitions"],
        "repetitions_s": s["repetitions"],
        # Aggregate
        "total_fas_score": total_fas_score,
        "proper_nouns_count": f["proper_nouns"] + a["proper_nouns"] + s["proper_nouns"],
        "repetitions_count": f["repetitions"] + a["repetitions"] + s["repetitions"],
        "letter_asymmetry": letter_asymmetry,
        # Exploratory phonemic similarity
        "mean_edit_distance_f": f["mean_edit_distance"],
        "mean_edit_distance_a": a["mean_edit_distance"],
        "mean_edit_distance_s": s["mean_edit_distance"],
    }


# ──────────────────────────────────────────────────────
# Phase 4c clustering scorer (Troyer/Pakhomov chain method)
# ──────────────────────────────────────────────────────

def cluster_letter(words: list[str]) -> dict:
    """Chain-method clustering for one letter's response sequence.

    Two consecutive words form a cluster link when
    :func:`thesis_project.scoring.phonemic_rules.linked` returns True.
    Singletons count as clusters of size 1 (Pakhomov 2016 convention),
    matching the Phase 3 SVF scorer.

    Args:
        words: ordered list of valid words (already lowercased, with
            proper-noun markers stripped, repetitions removed).

    Returns:
        dict with keys:
            clusters          — list[list[str]]
            cluster_sizes     — list[int]
            cluster_count     — int
            switch_count      — int (= max(0, cluster_count - 1))
            mean_cluster_size — float (NaN if cluster_count == 0)
            max_cluster_size  — int

    Edge cases:
        - words == []          : count=0, switch=0, mean=NaN, max=0
        - len(words) == 1      : count=1, switch=0, mean=1.0, max=1
    """
    if not words:
        return {
            "clusters": [],
            "cluster_sizes": [],
            "cluster_count": 0,
            "switch_count": 0,
            "mean_cluster_size": float("nan"),
            "max_cluster_size": 0,
        }

    if len(words) == 1:
        return {
            "clusters": [list(words)],
            "cluster_sizes": [1],
            "cluster_count": 1,
            "switch_count": 0,
            "mean_cluster_size": 1.0,
            "max_cluster_size": 1,
        }

    clusters: list[list[str]] = []
    current: list[str] = [words[0]]
    for prev, curr in zip(words[:-1], words[1:]):
        if linked(prev, curr):
            current.append(curr)
        else:
            clusters.append(current)
            current = [curr]
    clusters.append(current)

    sizes = [len(c) for c in clusters]
    return {
        "clusters": clusters,
        "cluster_sizes": sizes,
        "cluster_count": len(clusters),
        "switch_count": max(0, len(clusters) - 1),
        "mean_cluster_size": float(sum(sizes) / len(sizes)),
        "max_cluster_size": max(sizes),
    }


def score_fas(
    f_words: list[str],
    a_words: list[str],
    s_words: list[str],
    word_freq_provider=None,
) -> dict:
    """Compute per-letter and aggregated FAS clustering metrics.

    Inputs are already filtered (proper nouns stripped, repetitions
    removed) by the FAS pipeline. This function does NOT re-filter.

    Args:
        f_words: filtered word list for letter F.
        a_words: filtered word list for letter A.
        s_words: filtered word list for letter S.
        word_freq_provider: optional ``WordFrequencyProvider``. When
            supplied, the returned dict's ``mean_word_frequency`` field
            is the mean Zipf frequency across the pooled F+A+S words;
            NaN if no provider is given or the pooled list is empty.

    Returns a flat dict with keys:

        # Per-letter
        cluster_count_f, mean_cluster_size_f, switch_count_f
        cluster_count_a, mean_cluster_size_a, switch_count_a
        cluster_count_s, mean_cluster_size_s, switch_count_s

        # Aggregates (Linz feature inputs)
        cluster_count_total      — sum of per-letter cluster counts
        switch_count_total       — sum of per-letter switch counts
        mean_cluster_size        — grand mean over pooled cluster sizes
                                   (each cluster contributes once)

        # Lexical
        mean_word_frequency      — mean Zipf frequency over F+A+S pooled
                                   words; NaN if no provider or empty.
    """
    f = cluster_letter(f_words)
    a = cluster_letter(a_words)
    s = cluster_letter(s_words)

    pooled_sizes = f["cluster_sizes"] + a["cluster_sizes"] + s["cluster_sizes"]
    if pooled_sizes:
        mean_cluster_size = float(sum(pooled_sizes) / len(pooled_sizes))
    else:
        mean_cluster_size = float("nan")

    pooled_words = list(f_words) + list(a_words) + list(s_words)
    if word_freq_provider is None or not pooled_words:
        mwf: float = float("nan")
    else:
        mwf = float(word_freq_provider.mean_word_frequency(pooled_words))
        if mwf is None or (isinstance(mwf, float) and math.isnan(mwf)):
            mwf = float("nan")

    return {
        # Per-letter cluster metrics
        "cluster_count_f": f["cluster_count"],
        "mean_cluster_size_f": f["mean_cluster_size"],
        "switch_count_f": f["switch_count"],
        "cluster_count_a": a["cluster_count"],
        "mean_cluster_size_a": a["mean_cluster_size"],
        "switch_count_a": a["switch_count"],
        "cluster_count_s": s["cluster_count"],
        "mean_cluster_size_s": s["mean_cluster_size"],
        "switch_count_s": s["switch_count"],
        # Aggregates
        "cluster_count_total": f["cluster_count"] + a["cluster_count"] + s["cluster_count"],
        "switch_count_total": f["switch_count"] + a["switch_count"] + s["switch_count"],
        "mean_cluster_size": mean_cluster_size,
        # Lexical
        "mean_word_frequency": mwf,
    }
