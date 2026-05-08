"""Tests for ``thesis_project.evaluation.sq3_sampling``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from thesis_project.evaluation.sq3_sampling import (
    PRIMARY_SEED,
    RATER_COLUMNS,
    SENSITIVE_COLUMNS,
    TRAINING_SEED,
    assign_quartile,
    compute_quartile_cutpoints,
    draw_stratified_sample,
    draw_training_sample,
    load_eligible_pairs,
    make_rater_csv,
)


def _synthetic_bnt_csv(tmp_path, n: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "participant_id": [f"User-{i % 100}" for i in range(n)],
            "gender": rng.choice(["F", "M"], size=n),
            "age": rng.uniform(60, 90, size=n).round(1),
            "diagnosis": rng.choice(["HC", "MCI", "non-AD", "AD"], size=n),
            "mmse": rng.integers(15, 30, size=n),
            "gold": [f"target_{i % 30}" for i in range(n)],
            "raw_response": [f"resp_{i}" for i in range(n)],
            "normalized": [f"resp_{i}" for i in range(n)],
            "is_exact_match": rng.random(n) < 0.2,
            "is_non_response": rng.random(n) < 0.1,
            "cosine_sim": rng.uniform(0.0, 1.0, size=n),
            "binary_score": 0,
        }
    )
    path = tmp_path / "bnt.csv"
    df.to_csv(path, index=False)
    return path, df


def test_load_eligible_pairs_filters_correct_rows(tmp_path):
    path, df = _synthetic_bnt_csv(tmp_path)
    eligible = load_eligible_pairs(path)
    assert len(eligible) == int(((~df["is_non_response"]) & (~df["is_exact_match"])).sum())
    assert not eligible["is_non_response"].any()
    assert not eligible["is_exact_match"].any()
    assert list(eligible.columns) == list(df.columns)
    assert eligible.index.tolist() == list(range(len(eligible)))


def test_load_eligible_pairs_missing_column(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_eligible_pairs(p)


def test_quartile_cutpoints_match_pandas(tmp_path):
    _, df = _synthetic_bnt_csv(tmp_path)
    eligible = df[~df["is_non_response"] & ~df["is_exact_match"]].reset_index(drop=True)
    q25, q50, q75 = compute_quartile_cutpoints(eligible)
    expected = eligible["cosine_sim"].quantile([0.25, 0.5, 0.75]).tolist()
    assert (q25, q50, q75) == tuple(float(v) for v in expected)


def test_assign_quartile_returns_int_labels(tmp_path):
    _, df = _synthetic_bnt_csv(tmp_path)
    eligible = df[~df["is_non_response"] & ~df["is_exact_match"]].reset_index(drop=True)
    out = assign_quartile(eligible)
    assert "cosine_quartile" in out.columns
    assert set(out["cosine_quartile"].dropna().unique()) <= {0, 1, 2, 3}
    expected = pd.qcut(eligible["cosine_sim"], q=4, labels=False, duplicates="drop")
    assert (out["cosine_quartile"].astype("Int64").values == expected.astype("Int64").values).all()


def test_draw_stratified_sample_balanced_and_deterministic(tmp_path):
    _, df = _synthetic_bnt_csv(tmp_path, n=2400, seed=1)
    eligible = df[~df["is_non_response"] & ~df["is_exact_match"]].reset_index(drop=True)
    eligible = assign_quartile(eligible)
    s1 = draw_stratified_sample(eligible, per_quartile=50, seed=PRIMARY_SEED)
    s2 = draw_stratified_sample(eligible, per_quartile=50, seed=PRIMARY_SEED)
    pd.testing.assert_frame_equal(s1, s2)
    assert len(s1) == 200
    counts = s1["cosine_quartile"].value_counts().sort_index().tolist()
    assert counts == [50, 50, 50, 50]
    # pair_id uniqueness
    assert s1["pair_id"].is_unique
    # output sorted by pair_id, not by quartile (otherwise quartile order would
    # be monotone)
    quartiles = s1["cosine_quartile"].tolist()
    assert quartiles != sorted(quartiles)


def test_make_rater_csv_strips_sensitive_columns(tmp_path):
    _, df = _synthetic_bnt_csv(tmp_path, n=2400, seed=2)
    eligible = df[~df["is_non_response"] & ~df["is_exact_match"]].reset_index(drop=True)
    eligible = assign_quartile(eligible)
    sample = draw_stratified_sample(eligible, per_quartile=50, seed=PRIMARY_SEED)
    out = tmp_path / "rater.csv"
    make_rater_csv(sample, rater_seed=PRIMARY_SEED, output_path=out)
    raw = out.read_text(encoding="utf-8")
    written = pd.read_csv(out)
    assert set(written.columns) == set(RATER_COLUMNS)
    for col in SENSITIVE_COLUMNS:
        assert col not in written.columns
        assert col not in raw.splitlines()[0]
    assert len(written) == len(sample)
    # rating/category/is_compound/notes are empty in the written CSV
    for col in ("rating", "category", "is_compound", "notes"):
        assert written[col].isna().all() or (written[col].astype(str) == "").all()


def test_rater_seed_changes_row_order_but_not_pair_set(tmp_path):
    _, df = _synthetic_bnt_csv(tmp_path, n=2400, seed=3)
    eligible = df[~df["is_non_response"] & ~df["is_exact_match"]].reset_index(drop=True)
    eligible = assign_quartile(eligible)
    sample = draw_stratified_sample(eligible, per_quartile=50, seed=PRIMARY_SEED)
    p1 = tmp_path / "fb.csv"
    p2 = tmp_path / "co.csv"
    make_rater_csv(sample, rater_seed=PRIMARY_SEED + 0, output_path=p1)
    make_rater_csv(sample, rater_seed=PRIMARY_SEED + 1, output_path=p2)
    a = pd.read_csv(p1)
    b = pd.read_csv(p2)
    assert set(a["pair_id"]) == set(b["pair_id"])
    assert a["pair_id"].tolist() != b["pair_id"].tolist()


def test_dedup_at_gold_normalized_is_deterministic(tmp_path):
    """Same input → same dedup output, regardless of input row order.

    Constructs a small CSV with deliberate duplicates at the
    ``(gold, normalized)`` level across multiple participants, then
    asserts the dedup is deterministic and order-invariant.
    """
    base = pd.DataFrame(
        {
            "participant_id": [
                "User-2", "User-1", "User-3",  # all share (träd, träd)
                "User-1", "User-2",            # share (bil, fordon)
                "User-1",                      # unique (hund, vovve)
                "User-2",                      # unique (katt, kissemissen)
            ],
            "diagnosis": ["HC", "MCI", "AD", "HC", "MCI", "HC", "MCI"],
            "gold":       ["träd", "träd", "träd", "bil", "bil", "hund", "katt"],
            "normalized": ["träd", "träd", "träd", "fordon", "fordon", "vovve", "kissemissen"],
            "is_exact_match":  [False] * 7,
            "is_non_response": [False] * 7,
            "cosine_sim": [0.9, 0.9, 0.9, 0.6, 0.6, 0.4, 0.7],
        }
    )
    p1 = tmp_path / "bnt_a.csv"
    base.to_csv(p1, index=False)
    out_a = load_eligible_pairs(p1)

    permuted = base.iloc[[3, 0, 5, 6, 1, 4, 2]].reset_index(drop=True)
    p2 = tmp_path / "bnt_b.csv"
    permuted.to_csv(p2, index=False)
    out_b = load_eligible_pairs(p2)

    # 4 unique (gold, normalized) pairs survive.
    assert len(out_a) == 4
    assert len(out_b) == 4
    pairs_a = list(zip(out_a["gold"], out_a["normalized"]))
    pairs_b = list(zip(out_b["gold"], out_b["normalized"]))
    assert pairs_a == pairs_b

    # Order-invariance: the kept participant_id within each (gold, normalized)
    # group is the same regardless of input row order, because the sort
    # tiebreaker is participant_id.
    a_kept = dict(zip(pairs_a, out_a["participant_id"]))
    b_kept = dict(zip(pairs_b, out_b["participant_id"]))
    assert a_kept == b_kept
    # For (träd, träd), User-1 should be kept (lexicographically smallest).
    assert a_kept[("träd", "träd")] == "User-1"


def test_load_eligible_pairs_return_counts(tmp_path):
    base = pd.DataFrame(
        {
            "participant_id": ["U1", "U2", "U3", "U4"],
            "gold":            ["a",  "a",  "b",  "c"],
            "normalized":      ["x",  "x",  "y",  "z"],
            "is_exact_match":  [False, False, True,  False],
            "is_non_response": [False, False, False, False],
            "cosine_sim":      [0.5, 0.5, 0.9, 0.3],
        }
    )
    p = tmp_path / "bnt.csv"
    base.to_csv(p, index=False)
    eligible, counts = load_eligible_pairs(p, return_counts=True)
    assert counts["n_total"] == 4
    assert counts["n_exact_match"] == 1
    assert counts["n_after_filter"] == 3   # row b removed by exact-match
    assert counts["n_after_dedup"] == 2    # one (a,x) duplicate dropped
    assert counts["n_duplicates_dropped"] == 1
    assert len(eligible) == 2


def test_training_sample_disjoint_from_live(tmp_path):
    _, df = _synthetic_bnt_csv(tmp_path, n=2400, seed=4)
    eligible = df[~df["is_non_response"] & ~df["is_exact_match"]].reset_index(drop=True)
    eligible = assign_quartile(eligible)
    sample = draw_stratified_sample(eligible, per_quartile=50, seed=PRIMARY_SEED)
    keys = ["gold", "raw_response", "cosine_sim"]
    sample_tuples = set(map(tuple, sample[keys].itertuples(index=False, name=None)))
    elig_tuples = list(map(tuple, eligible[keys].itertuples(index=False, name=None)))
    live_idx = pd.Index([i for i, t in enumerate(elig_tuples) if t in sample_tuples])
    train = draw_training_sample(eligible, live_idx, n=20, seed=TRAINING_SEED)
    assert len(train) == 20
    train_keys = set(zip(train["gold"], train["raw_response"], train["cosine_sim"]))
    sample_keys = set(zip(sample["gold"], sample["raw_response"], sample["cosine_sim"]))
    assert train_keys.isdisjoint(sample_keys)
