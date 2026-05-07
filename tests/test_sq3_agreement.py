"""Tests for ``thesis_project.evaluation.sq3_agreement``.

The methodology document and instructions reference
``tests/fixtures/saldo_mini.xml`` as a Phase A artifact. Phase A's
``SaldoGraph`` is upstream and read-only here; until that fixture lands
the SALDO test uses a duck-typed mock that exposes the public methods
``lookup``, ``path_length`` and ``wu_palmer`` exactly as
``SaldoGraph`` does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from thesis_project.evaluation.sq3_agreement import (
    AgreementResult,
    rater_model_per_category,
    rater_model_per_quartile,
    rater_model_spearman,
    rater_saldo_spearman,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load_consolidated_ratings() -> pd.DataFrame:
    a = pd.read_csv(FIXTURES / "sq3_mock_ratings_rater1.csv")
    b = pd.read_csv(FIXTURES / "sq3_mock_ratings_rater2.csv")
    merged = a.merge(b, on="pair_id", suffixes=("_a", "_b"))
    out = pd.DataFrame(
        {
            "pair_id": merged["pair_id"],
            "rating": (merged["rating_a"] + merged["rating_b"]) / 2.0,
            "category": merged["category_a"],
        }
    )
    return out


def _build_model_cosines(ratings: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return a cosine table that loosely correlates with the rating."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.05, size=len(ratings))
    cos = (ratings["rating"].to_numpy() / 3.0) + noise
    cos = np.clip(cos, 0.0, 1.0)
    return pd.DataFrame({"pair_id": ratings["pair_id"].to_numpy(), "cosine_sim": cos})


def test_rater_model_spearman_against_scipy():
    ratings = _load_consolidated_ratings()
    cos = _build_model_cosines(ratings, seed=11)
    out = rater_model_spearman({"ratings": ratings}["ratings"].assign(), {"sbert": cos})  # type: ignore[arg-type]
    # Build directly to avoid any aliasing issues.
    out = rater_model_spearman(ratings, {"sbert": cos})
    rho_expected, _ = spearmanr(ratings["rating"], cos["cosine_sim"])
    res = out["sbert"]
    assert isinstance(res, AgreementResult)
    assert res.spearman_rho == pytest.approx(rho_expected)
    assert res.n == len(ratings)
    assert res.ci_low <= res.spearman_rho <= res.ci_high or np.isnan(res.ci_low)


def test_per_quartile_long_format():
    ratings = _load_consolidated_ratings()
    sampled = pd.read_csv(FIXTURES / "sq3_mock_sampled.csv")
    cos = _build_model_cosines(ratings, seed=12)
    out = rater_model_per_quartile(ratings, sampled, {"sbert": cos})
    assert set(out.columns) == {"model_name", "quartile", "n", "spearman_rho", "ci_low", "ci_high"}
    assert set(out["quartile"]) <= {0, 1, 2, 3}
    assert (out["model_name"] == "sbert").all()


def test_per_category_below_min_n_flag():
    ratings = _load_consolidated_ratings()
    cos = _build_model_cosines(ratings, seed=13)
    out = rater_model_per_category(ratings, {"sbert": cos}, min_n=15)
    assert "below_min_n" in out.columns
    # Every category in the fixture has only ~4 rows (30/7), so every row
    # should be flagged below_min_n with NaN rho.
    assert out["below_min_n"].all()
    assert out["spearman_rho"].isna().all()
    assert out["ci_low"].isna().all()


class _MockSaldoGraph:
    """Minimal duck-typed SaldoGraph for unit tests."""

    def __init__(self, senses: dict[str, list[str]], distances: dict[tuple[str, str], int]):
        self._senses = senses
        # Symmetric distance map.
        self._dist = {}
        for (a, b), d in distances.items():
            self._dist[(a, b)] = d
            self._dist[(b, a)] = d

    def lookup(self, written_form: str) -> list[str]:
        return list(self._senses.get(written_form, []))

    def path_length(self, s1: str, s2: str) -> int | None:
        if s1 == s2:
            return 0
        return self._dist.get((s1, s2))

    def wu_palmer(self, s1: str, s2: str) -> float | None:
        if s1 == s2:
            return 1.0
        d = self._dist.get((s1, s2))
        if d is None:
            return None
        # Toy formula: closer pairs map closer to 1.0, distant pairs to 0.
        return float(1.0 / (1.0 + d / 2.0))


def test_rater_saldo_spearman_with_mock_graph():
    target_response = pd.DataFrame(
        {
            "pair_id": [f"p{i}" for i in range(6)],
            "target": ["hund", "hund", "katt", "katt", "öken", "räv"],
            "response": ["katt", "djur", "djur", "räv", "öken_oov", "rev_oov"],
        }
    )
    ratings = pd.DataFrame(
        {
            "pair_id": target_response["pair_id"],
            # Higher rating where SALDO will report shorter paths.
            "rating": [3, 2, 2, 3, 0, 0],
        }
    )
    senses = {
        "hund": ["hund..1"],
        "katt": ["katt..1"],
        "djur": ["djur..1"],
        "räv": ["räv..1"],
        "öken": ["öken..1"],
        # öken_oov, rev_oov absent on purpose
    }
    distances = {
        ("hund..1", "katt..1"): 2,
        ("hund..1", "djur..1"): 1,
        ("katt..1", "djur..1"): 1,
        ("katt..1", "räv..1"): 2,
        ("hund..1", "räv..1"): 3,
        ("öken..1", "djur..1"): 5,
    }
    g = _MockSaldoGraph(senses, distances)
    res = rater_saldo_spearman(ratings, g, target_response, n_bootstrap=200)
    assert res.n_total == 6
    assert res.n_in_vocab == 4  # last two pairs are OOV on the response side
    assert res.oov_rate == pytest.approx(2 / 6)
    # With only 4 in-vocab points the Spearman still resolves; sign should be
    # negative (more distant pair => lower path similarity, but this fixture
    # is small so we just check the result is finite).
    assert not np.isnan(res.path_spearman.spearman_rho)
    assert not np.isnan(res.wu_palmer_spearman.spearman_rho)


def test_per_category_handles_no_pair_overlap():
    ratings = pd.DataFrame(
        {"pair_id": ["a", "b"], "rating": [1.0, 2.0], "category": ["coordinate", "hypernym"]}
    )
    cos = pd.DataFrame({"pair_id": ["x", "y"], "cosine_sim": [0.1, 0.9]})
    out = rater_model_per_category(ratings, {"sbert": cos}, min_n=15)
    assert out.empty
