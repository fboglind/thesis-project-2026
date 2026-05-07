"""Tests for ``thesis_project.evaluation.sq3_reliability``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from thesis_project.evaluation.sq3_reliability import (
    VALID_CATEGORIES,
    compute_reliability,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_branch_multi_against_sklearn_reference():
    files = [
        FIXTURES / "sq3_mock_ratings_rater1.csv",
        FIXTURES / "sq3_mock_ratings_rater2.csv",
    ]
    report = compute_reliability(files, branch="multi")
    a = pd.read_csv(files[0]).set_index("pair_id").sort_index()
    b = pd.read_csv(files[1]).set_index("pair_id").sort_index()
    expected_kappa = cohen_kappa_score(
        a["rating"], b["rating"], weights="quadratic", labels=[0, 1, 2, 3]
    )
    expected_rho, _ = spearmanr(a["rating"], b["rating"])
    expected_cat = cohen_kappa_score(a["category"], b["category"])

    key = ("rater1", "rater2")
    assert report.weighted_kappa[key] == pytest.approx(expected_kappa)
    assert report.spearman_rho[key] == pytest.approx(expected_rho)
    assert report.category_kappa[key] == pytest.approx(expected_cat)
    assert report.n_pairs == 30
    assert report.branch == "multi"
    assert report.rater_ids == ["rater1", "rater2"]


def test_branch_sole_test_retest_orders_files_correctly():
    files = [
        FIXTURES / "sq3_mock_ratings_rater_round1.csv",
        FIXTURES / "sq3_mock_ratings_rater_round2.csv",
    ]
    report = compute_reliability(files, branch="sole_test_retest")
    assert report.branch == "sole_test_retest"
    # The pairwise key ordering follows the file list order.
    keys = list(report.weighted_kappa.keys())
    assert len(keys) == 1
    assert keys[0] == ("rater_round1", "rater_round2")
    # Round-2 file is a permutation of round-1 row order, so reliability code
    # must align by pair_id rather than by row position.
    assert 0.0 < report.weighted_kappa[keys[0]] < 1.0


def test_interpretability_flag_thresholds():
    files = [
        FIXTURES / "sq3_mock_ratings_rater1.csv",
        FIXTURES / "sq3_mock_ratings_rater2.csv",
    ]
    report = compute_reliability(files, branch="multi")
    mean_kappa = float(np.mean(list(report.weighted_kappa.values())))
    if mean_kappa >= 0.6:
        assert report.interpretability_flag == "primary"
    elif mean_kappa >= 0.4:
        assert report.interpretability_flag == "cautioned"
    else:
        assert report.interpretability_flag == "exploratory"


def test_validation_rejects_out_of_range_rating(tmp_path):
    df = pd.read_csv(FIXTURES / "sq3_mock_ratings_rater1.csv")
    df.loc[0, "rating"] = 5
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    files = [bad, FIXTURES / "sq3_mock_ratings_rater2.csv"]
    with pytest.raises(ValueError, match="outside"):
        compute_reliability(files, branch="multi")


def test_validation_rejects_unknown_category(tmp_path):
    df = pd.read_csv(FIXTURES / "sq3_mock_ratings_rater1.csv")
    df.loc[0, "category"] = "made_up_category"
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    files = [bad, FIXTURES / "sq3_mock_ratings_rater2.csv"]
    with pytest.raises(ValueError, match="closed list"):
        compute_reliability(files, branch="multi")


def test_branch_argument_validation():
    files = [FIXTURES / "sq3_mock_ratings_rater1.csv"]
    with pytest.raises(ValueError, match="2 or 3"):
        compute_reliability(files, branch="multi")
    with pytest.raises(ValueError, match="exactly 2"):
        compute_reliability(files, branch="sole_test_retest")
    with pytest.raises(ValueError, match="Unknown branch"):
        compute_reliability(files * 2, branch="weird")  # type: ignore[arg-type]


def test_categories_in_fixture_cover_closed_list():
    a = pd.read_csv(FIXTURES / "sq3_mock_ratings_rater1.csv")
    assert set(a["category"]) <= VALID_CATEGORIES
    assert set(a["category"]) == VALID_CATEGORIES.intersection(set(a["category"]))
    # at least one example of each category
    assert set(a["category"]) >= {
        "coordinate",
        "hypernym",
        "hyponym",
        "circumlocution",
        "phonological",
        "unrelated",
        "other",
    }
