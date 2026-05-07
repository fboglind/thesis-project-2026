"""Tests for thesis_project.scoring.phonemic_rules.

Covers each of R1, R2, R3 plus the union ``linked`` predicate, with
Swedish-specific edge cases (åäö, single-vowel rime guard).
"""

from thesis_project.scoring.phonemic_rules import (
    SWEDISH_VOWELS,
    linked,
    shared_first_two_letters,
    shared_rime,
    vowel_only_substitution,
)


# ──────────────────────────────────────────────────────
# R1 — shared first two letters
# ──────────────────────────────────────────────────────

def test_r1_basic():
    assert shared_first_two_letters("frukt", "frid")
    assert shared_first_two_letters("art", "arm")
    assert not shared_first_two_letters("frukt", "fisk")


def test_r1_short_words():
    assert not shared_first_two_letters("a", "an")
    assert not shared_first_two_letters("", "")
    assert not shared_first_two_letters("an", "a")


def test_r1_swedish_chars():
    assert shared_first_two_letters("åska", "åska")
    assert shared_first_two_letters("öken", "öken")
    assert not shared_first_two_letters("ål", "öl")  # different second-position char doesn't apply, len 2 each
    # Actually: "ål" and "öl" have len 2 each; w1[:2]='ål', w2[:2]='öl', not equal.


# ──────────────────────────────────────────────────────
# R2 — shared rime
# ──────────────────────────────────────────────────────

def test_r2_basic():
    assert shared_rime("sand", "stand")        # rime "and"
    assert shared_rime("fågel", "tagel")       # rime "el"
    assert not shared_rime("sand", "salt")     # "and" vs "alt"


def test_r2_min_length_guard():
    """Words ending in single vowel have rime length 1; must NOT match."""
    assert not shared_rime("saga", "vaga")     # rime "a", length 1 each
    assert not shared_rime("krypa", "skapa")   # rime "a", length 1 each


def test_r2_no_vowels():
    """Synthetic edge case: word with no vowels has no rime.

    Note: 'y' is in SWEDISH_VOWELS per the spec, so a true vowel-free
    word in this rule's eyes uses only consonants from {bcdfghjklmnpqrstvwxz}.
    """
    assert not shared_rime("bcd", "bcd")
    assert not shared_rime("ksk", "ksk")


def test_r2_swedish_vowels_in_rime():
    """å, ä, ö all count as vowels for rime extraction."""
    # 'höst' rime = 'öst' (len 3); 'köst' rime = 'öst'
    assert shared_rime("höst", "köst")


def test_r2_different_lengths_same_rime():
    # 'pant' rime='ant', 'plant' rime='ant'  → match
    assert shared_rime("pant", "plant")


# ──────────────────────────────────────────────────────
# R3 — vowel-only substitution
# ──────────────────────────────────────────────────────

def test_r3_basic():
    assert vowel_only_substitution("fil", "fal")    # i → a
    assert vowel_only_substitution("fas", "fos")    # a → o


def test_r3_different_lengths():
    assert not vowel_only_substitution("fil", "fjäl")
    assert not vowel_only_substitution("fil", "fila")


def test_r3_consonant_substitution_rejected():
    assert not vowel_only_substitution("fil", "fis")   # l → s, both consonants
    assert not vowel_only_substitution("hund", "rund")  # h → r, both consonants


def test_r3_mixed_substitution_rejected():
    """One vowel, one consonant — rule requires both to be vowels."""
    # Edge case: "pak" vs "pek" both vowels (a/e) — should pass
    assert vowel_only_substitution("pak", "pek")
    # "pak" vs "pa" different lengths — fails
    assert not vowel_only_substitution("pak", "pa")


def test_r3_identical_rejected():
    """Identical words have zero differences; substitution requires exactly one."""
    assert not vowel_only_substitution("fil", "fil")


def test_r3_two_differences_rejected():
    assert not vowel_only_substitution("fil", "fos")  # i→o, l→s — two diffs


def test_r3_swedish_vowel_swap():
    assert vowel_only_substitution("åska", "öska")
    assert vowel_only_substitution("häst", "höst")


# ──────────────────────────────────────────────────────
# linked (union)
# ──────────────────────────────────────────────────────

def test_linked_any_rule():
    assert linked("frukt", "frid")        # R1
    assert linked("sand", "stand")        # R2
    assert linked("fil", "fal")           # R3
    assert not linked("hund", "elefant")  # nothing


def test_linked_short_words():
    """Two unrelated short consonant words satisfy no rule."""
    assert not linked("b", "k")
    # Single-char vowel pairs DO satisfy R3 by construction (both vowels,
    # equal length, exactly one differing position) — exercised here for
    # documentation purposes.
    assert linked("a", "i")


# ──────────────────────────────────────────────────────
# Vowel-set hygiene
# ──────────────────────────────────────────────────────

def test_swedish_vowels_set():
    for v in "aeiouyåäö":
        assert v in SWEDISH_VOWELS
    for c in "bcdfghjklmnpqrstvwxz":
        assert c not in SWEDISH_VOWELS
