"""fas_scorer.py

Scoring for Phonemic Verbal Fluency (FAS) test responses.
Primarily count-based following Tallberg et al. (2008) and
Borkowski et al. (1967).
"""

import numpy as np


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

def score_fas(
    responses_f: list[str],
    responses_a: list[str],
    responses_s: list[str],
    flagged_errors: list[str] | None = None,
) -> dict:
    """Compute FAS metrics for a single participant.

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
