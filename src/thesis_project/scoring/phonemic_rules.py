"""phonemic_rules.py

Pre-registered Swedish-orthographic phonemic-similarity rules for FAS
clustering, adapted from Troyer et al. (1997).

Two consecutive valid words form a cluster link if any of R1–R3 holds:

- **R1 (shared first two letters):** ``w1[:2] == w2[:2]``. Direct port
  of Troyer's first rule.
- **R2 (shared rime):** the substring of each word from its last
  vowel onwards is identical and is at least ``min_rime_length`` (=2)
  characters long. The minimum-length guard prevents trivial matches
  on words ending in a single vowel (e.g. ``-a`` infinitives, definite
  suffixes), which are extremely common in Swedish.
- **R3 (vowel-only single-character substitution):** the words have the
  same length and differ at exactly one position, and both characters
  at that position are Swedish vowels.

Troyer's fourth rule (homonyms) is **dropped** for the Swedish
adaptation. Troyer's homonym rule presupposes a pronunciation
dictionary or a participant verbal cue ("some vs. sum") which is not
available for text-only synthetic data, and Swedish has substantially
fewer orthographic homophones than English. Documented as a
methodological limitation in the thesis.

References:
    Troyer, A. K., Moscovitch, M., & Winocur, G. (1997). Clustering and
    switching as two components of verbal fluency: evidence from
    younger and older healthy adults. *Neuropsychology*, 11(1).
"""

from __future__ import annotations

SWEDISH_VOWELS: frozenset[str] = frozenset("aeiouyåäö")


def shared_first_two_letters(w1: str, w2: str) -> bool:
    """R1: True iff ``w1`` and ``w2`` share their first two characters.

    Both words are assumed lowercased and stripped. Returns False if
    either word has fewer than 2 characters.
    """
    if len(w1) < 2 or len(w2) < 2:
        return False
    return w1[:2] == w2[:2]


def _rime(word: str) -> str | None:
    """Return the rime (last-vowel-onwards substring) or None if no vowel."""
    for i in range(len(word) - 1, -1, -1):
        if word[i] in SWEDISH_VOWELS:
            return word[i:]
    return None


def shared_rime(w1: str, w2: str, min_rime_length: int = 2) -> bool:
    """R2: True iff the rime of ``w1`` and ``w2`` are identical and at
    least ``min_rime_length`` characters long.

    The rime is the substring from the last vowel (in
    :data:`SWEDISH_VOWELS`) to the end of the word. If a word has no
    vowels, returns False.
    """
    r1 = _rime(w1)
    r2 = _rime(w2)
    if r1 is None or r2 is None:
        return False
    if len(r1) < min_rime_length or len(r2) < min_rime_length:
        return False
    return r1 == r2


def vowel_only_substitution(w1: str, w2: str) -> bool:
    """R3: True iff ``w1`` and ``w2`` differ by exactly one substitution
    and both substituted characters are Swedish vowels.

    Insertions and deletions do not count: words of different length
    always return False. Identical words return False (the substitution
    requirement is exactly one differing position).
    """
    if len(w1) != len(w2):
        return False
    diff_idx = -1
    for i, (c1, c2) in enumerate(zip(w1, w2)):
        if c1 != c2:
            if diff_idx >= 0:
                return False  # second difference — fails
            diff_idx = i
    if diff_idx < 0:
        return False  # words are identical → no substitution
    c1, c2 = w1[diff_idx], w2[diff_idx]
    return c1 in SWEDISH_VOWELS and c2 in SWEDISH_VOWELS


def linked(w1: str, w2: str) -> bool:
    """The cluster-link predicate: True iff any of R1, R2, R3 holds."""
    return (
        shared_first_two_letters(w1, w2)
        or shared_rime(w1, w2)
        or vowel_only_substitution(w1, w2)
    )
