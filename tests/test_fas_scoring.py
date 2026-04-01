"""test_fas_scoring.py

Unit tests for the FAS scorer: parsing logic, flagged-error handling,
count metrics, and aggregate scores.
"""

import math

from thesis_project.scoring.fas_scorer import _levenshtein, score_fas


# ──────────────────────────────────────────────────────
# score_fas: basic counts
# ──────────────────────────────────────────────────────

def test_total_and_valid_counts():
    # 3 F-words, no proper nouns, no repetitions
    result = score_fas(
        responses_f=["fisk", "fågel", "fjäril"],
        responses_a=["anka", "abborre"],
        responses_s=["snö", "sol", "sten"],
    )
    assert result["total_f"] == 3
    assert result["total_a"] == 2
    assert result["total_s"] == 3
    assert result["valid_f"] == 3
    assert result["valid_a"] == 2
    assert result["valid_s"] == 3
    assert result["total_fas_score"] == 8


def test_total_fas_score():
    result = score_fas(
        responses_f=["fisk", "fisk"],  # 1 repetition
        responses_a=["anka"],
        responses_s=["snö"],
    )
    assert result["total_f"] == 2
    assert result["repetitions_f"] == 1
    assert result["valid_f"] == 1
    assert result["total_fas_score"] == 3  # 1 + 1 + 1


def test_proper_noun_flagged():
    result = score_fas(
        responses_f=["fisk", "fredrik"],
        responses_a=["anka"],
        responses_s=["snö"],
        flagged_errors=["<Fredrik>"],
    )
    assert result["proper_nouns_f"] == 1
    assert result["valid_f"] == 1  # fisk only; fredrik is flagged
    assert result["proper_nouns_count"] == 1


def test_repetitions_counted():
    result = score_fas(
        responses_f=["fisk", "fisk", "fisk"],
        responses_a=[],
        responses_s=[],
    )
    assert result["repetitions_f"] == 2
    assert result["valid_f"] == 1
    assert result["repetitions_count"] == 2


def test_empty_slots_not_counted():
    # Empty strings mean no word produced — should not inflate total count
    result = score_fas(
        responses_f=["fisk", "", "fågel"],
        responses_a=["", "anka", ""],
        responses_s=["snö"],
    )
    assert result["total_f"] == 2
    assert result["total_a"] == 1
    assert result["total_s"] == 1


def test_letter_asymmetry_equal():
    result = score_fas(
        responses_f=["fisk"],
        responses_a=["anka"],
        responses_s=["snö"],
    )
    assert result["letter_asymmetry"] == 0.0


def test_letter_asymmetry_unequal():
    result = score_fas(
        responses_f=["fisk", "fågel", "fjäril"],
        responses_a=[],
        responses_s=[],
    )
    assert result["letter_asymmetry"] > 0.0


def test_all_empty():
    result = score_fas(responses_f=[], responses_a=[], responses_s=[])
    assert result["total_fas_score"] == 0
    assert result["letter_asymmetry"] == 0.0


def test_mean_edit_distance_present():
    result = score_fas(
        responses_f=["fisk", "fågel"],
        responses_a=["anka"],
        responses_s=[],
    )
    assert math.isfinite(result["mean_edit_distance_f"])
    assert math.isnan(result["mean_edit_distance_a"])  # only 1 word, no pairs


# ──────────────────────────────────────────────────────
# Levenshtein helper
# ──────────────────────────────────────────────────────

def test_levenshtein_identical():
    assert _levenshtein("fisk", "fisk") == 0


def test_levenshtein_empty():
    assert _levenshtein("", "fisk") == 4
    assert _levenshtein("fisk", "") == 4


def test_levenshtein_one_edit():
    assert _levenshtein("fisk", "disk") == 1


def test_levenshtein_completely_different():
    assert _levenshtein("abc", "xyz") == 3
