"""Tests for the new FAS clustering scorer (Phase 4c).

The count-based scorer is exercised in tests/test_fas_scoring.py
under its preserved name ``score_fas_counts``; this file covers the
new ``cluster_letter`` and ``score_fas`` APIs.
"""

import math

import pytest

from thesis_project.scoring.fas_scorer import cluster_letter, score_fas


# ──────────────────────────────────────────────────────
# cluster_letter
# ──────────────────────────────────────────────────────

def test_cluster_letter_empty():
    r = cluster_letter([])
    assert r["cluster_count"] == 0
    assert r["switch_count"] == 0
    assert math.isnan(r["mean_cluster_size"])
    assert r["max_cluster_size"] == 0
    assert r["clusters"] == []


def test_cluster_letter_single():
    r = cluster_letter(["fil"])
    assert r["cluster_count"] == 1
    assert r["switch_count"] == 0
    assert r["mean_cluster_size"] == 1.0
    assert r["max_cluster_size"] == 1
    assert r["clusters"] == [["fil"]]


def test_cluster_letter_all_linked_r1():
    """All adjacent pairs share first two letters: one big cluster."""
    r = cluster_letter(["frukt", "frid", "frost"])
    assert r["cluster_count"] == 1
    assert r["mean_cluster_size"] == 3.0
    assert r["switch_count"] == 0
    assert r["max_cluster_size"] == 3


def test_cluster_letter_no_links():
    """Three F-words with no link in any rule → three singletons."""
    # fil-frukt: R1 'fi' vs 'fr' ✗; R2 'il' vs 'ukt' ✗; R3 len differ ✗ → no link
    # frukt-färg: R1 'fr' vs 'fä' ✗; R2 'ukt' vs 'ärg' ✗; R3 len differ ✗ → no link
    r = cluster_letter(["fil", "frukt", "färg"])
    assert r["cluster_count"] == 3
    assert r["switch_count"] == 2
    assert r["mean_cluster_size"] == 1.0
    assert r["max_cluster_size"] == 1


def test_cluster_letter_mixed():
    """frukt-frid linked (R1), frid-färg unlinked → 2 clusters."""
    r = cluster_letter(["frukt", "frid", "färg"])
    assert r["cluster_count"] == 2
    assert r["cluster_sizes"] == [2, 1]
    assert r["switch_count"] == 1
    assert r["mean_cluster_size"] == 1.5
    assert r["max_cluster_size"] == 2


def test_cluster_letter_r2_link():
    """Words linked only via shared rime."""
    # 'sand' rime 'and', 'stand' rime 'and' — R2 link
    r = cluster_letter(["sand", "stand"])
    assert r["cluster_count"] == 1
    assert r["mean_cluster_size"] == 2.0


def test_cluster_letter_r3_link():
    """Words linked only via vowel-only substitution."""
    r = cluster_letter(["fil", "fal"])
    assert r["cluster_count"] == 1


# ──────────────────────────────────────────────────────
# score_fas
# ──────────────────────────────────────────────────────

def test_score_fas_aggregates():
    f = ["frukt", "frid", "färg"]    # 2 clusters: [frukt, frid], [färg]
    a = ["arm", "art", "arv"]        # all R1-linked: 1 cluster of 3
    s = ["sand", "stand"]            # R2-linked: 1 cluster of 2
    r = score_fas(f, a, s, word_freq_provider=None)

    assert r["cluster_count_total"] == 2 + 1 + 1
    assert r["switch_count_total"] == 1 + 0 + 0
    # pooled cluster sizes: [2, 1, 3, 2] → mean 2.0
    assert r["mean_cluster_size"] == pytest.approx(2.0)
    assert math.isnan(r["mean_word_frequency"])

    # Per-letter checks
    assert r["cluster_count_f"] == 2
    assert r["switch_count_f"] == 1
    assert r["mean_cluster_size_f"] == 1.5
    assert r["cluster_count_a"] == 1
    assert r["switch_count_a"] == 0
    assert r["mean_cluster_size_a"] == 3.0
    assert r["cluster_count_s"] == 1
    assert r["switch_count_s"] == 0
    assert r["mean_cluster_size_s"] == 2.0


def test_score_fas_all_empty():
    r = score_fas([], [], [], word_freq_provider=None)
    assert r["cluster_count_total"] == 0
    assert r["switch_count_total"] == 0
    assert math.isnan(r["mean_cluster_size"])
    assert math.isnan(r["mean_word_frequency"])


def test_score_fas_one_letter_only():
    """Only F has words; A and S empty. Pair linked via R3."""
    r = score_fas(["fil", "fal"], [], [], word_freq_provider=None)
    # fil vs fal: R3 single vowel substitution i→a → linked, 1 cluster of 2.
    assert r["cluster_count_f"] == 1
    assert r["cluster_count_a"] == 0
    assert r["cluster_count_s"] == 0
    assert r["cluster_count_total"] == 1


def test_score_fas_with_frequency():
    """With a real provider, MWF is finite."""
    from thesis_project.lexical.word_frequency import WordFrequencyProvider
    provider = WordFrequencyProvider(source="wordfreq")
    r = score_fas(["fil"], ["arm"], ["sand"], word_freq_provider=provider)
    assert isinstance(r["mean_word_frequency"], float)
    assert not math.isnan(r["mean_word_frequency"])


def test_score_fas_with_stub_provider():
    """The provider is called with the pooled F+A+S list."""
    captured = {}

    class StubProvider:
        def mean_word_frequency(self, words):
            captured["words"] = list(words)
            return 4.2

    r = score_fas(["fil", "fan"], ["arm"], ["sand"], word_freq_provider=StubProvider())
    assert r["mean_word_frequency"] == 4.2
    assert captured["words"] == ["fil", "fan", "arm", "sand"]


def test_score_fas_per_letter_keys_present():
    """All nine per-letter keys are present in the output schema."""
    r = score_fas([], [], [], word_freq_provider=None)
    expected_keys = {
        "cluster_count_f", "mean_cluster_size_f", "switch_count_f",
        "cluster_count_a", "mean_cluster_size_a", "switch_count_a",
        "cluster_count_s", "mean_cluster_size_s", "switch_count_s",
        "cluster_count_total", "switch_count_total", "mean_cluster_size",
        "mean_word_frequency",
    }
    assert expected_keys.issubset(r.keys())
