"""Tests for ``thesis_project.evaluation.sq3_divergence``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from thesis_project.evaluation.sq3_divergence import compute_divergence_catalog

FIXTURES = Path(__file__).parent / "fixtures"


class _NullSaldoGraph:
    """Minimal SaldoGraph that reports everything OOV."""

    def lookup(self, written_form: str) -> list[str]:
        return []

    def path_length(self, s1: str, s2: str):
        return None

    def wu_palmer(self, s1: str, s2: str):
        return None


class _MockSaldoGraph:
    def __init__(self, senses, distances, primaries=None):
        self._senses = senses
        self._dist = {}
        for (a, b), d in distances.items():
            self._dist[(a, b)] = d
            self._dist[(b, a)] = d
        self._primary = primaries or {}

    def lookup(self, w):
        return list(self._senses.get(w, []))

    def path_length(self, s1, s2):
        if s1 == s2:
            return 0
        return self._dist.get((s1, s2))

    def wu_palmer(self, s1, s2):
        if s1 == s2:
            return 1.0
        d = self._dist.get((s1, s2))
        return None if d is None else 1.0 / (1.0 + d / 2.0)

    def primary(self, sense):
        return self._primary.get(sense)


def _build_inputs():
    ratings = pd.read_csv(FIXTURES / "sq3_mock_ratings_rater1.csv")
    sampled = pd.read_csv(FIXTURES / "sq3_mock_sampled.csv")
    cos = sampled[["pair_id", "cosine_sim"]].copy()
    return ratings, sampled, cos


def test_threshold_is_strictly_90th_percentile():
    ratings, sampled, cos = _build_inputs()
    catalog = compute_divergence_catalog(
        ratings, sampled, cos, _NullSaldoGraph(), threshold_percentile=90.0
    )
    assert "is_divergence_case" in catalog.columns
    threshold = catalog.attrs["threshold_value"]
    expected = np.percentile(catalog["disagreement"], 90.0)
    assert threshold == float(expected)
    # Definition: strictly greater than the threshold.
    assert (catalog["is_divergence_case"] == (catalog["disagreement"] > threshold)).all()


def test_catalog_is_sorted_descending_by_disagreement():
    ratings, sampled, cos = _build_inputs()
    catalog = compute_divergence_catalog(
        ratings, sampled, cos, _NullSaldoGraph(), threshold_percentile=90.0
    )
    diffs = catalog["disagreement"].to_numpy()
    assert (diffs[:-1] >= diffs[1:]).all()


def test_required_columns_present():
    ratings, sampled, cos = _build_inputs()
    catalog = compute_divergence_catalog(
        ratings, sampled, cos, _NullSaldoGraph(), threshold_percentile=90.0
    )
    expected = {
        "pair_id",
        "target",
        "response",
        "rater_mean_rating",
        "cosine_sim_primary_model",
        "disagreement",
        "rater_category",
        "is_compound",
        "saldo_relation_summary",
        "is_divergence_case",
    }
    assert set(catalog.columns) == expected


def test_saldo_summary_distinguishes_oov_and_relation():
    ratings = pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p3", "p4"],
            "rating": [3, 2, 0, 0],
            "category": ["coordinate", "hypernym", "unrelated", "unrelated"],
            "is_compound": [False, False, False, False],
        }
    )
    sampled = pd.DataFrame(
        {
            "pair_id": ["p1", "p2", "p3", "p4"],
            "target": ["hund", "hund", "kamel", "katt"],
            "response": ["djur", "katt", "cykel_oov", "räv"],
        }
    )
    cos = pd.DataFrame(
        {"pair_id": ["p1", "p2", "p3", "p4"], "cosine_sim": [0.9, 0.7, 0.05, 0.5]}
    )
    senses = {
        "hund": ["hund..1"],
        "djur": ["djur..1"],
        "katt": ["katt..1"],
        "kamel": ["kamel..1"],
        "räv": ["räv..1"],
    }
    distances = {
        ("hund..1", "djur..1"): 1,
        ("hund..1", "katt..1"): 2,
        ("katt..1", "räv..1"): 2,
    }
    primaries = {"hund..1": "djur..1", "katt..1": "djur..1"}
    g = _MockSaldoGraph(senses, distances, primaries)
    catalog = compute_divergence_catalog(ratings, sampled, cos, g)
    by_pair = {row["pair_id"]: row["saldo_relation_summary"] for _, row in catalog.iterrows()}
    assert by_pair["p1"] == "mother(t→r)"  # hund's primary is djur
    assert by_pair["p2"] == "m-sibling"    # hund and katt share djur as primary
    assert by_pair["p3"] == "oov(r)"        # cykel_oov absent
    assert by_pair["p4"] == "m-sibling"    # katt and räv reachable via 2 hops
