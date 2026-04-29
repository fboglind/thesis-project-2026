"""test_data_loaders.py

Integration tests for the BNT, SVF, and FAS data loaders against the
real XLSX files. Verifies structure, types, MMSE extraction, label-based
metadata detection, and the per-test-metadata-only invariant.
"""

from pathlib import Path

import pandas as pd
import pytest

from thesis_project.preprocessing.data_loader import (
    DATA_DIR,
    BNT_PATH,
    FAS_PATH,
    SVF_PATH,
    _extract_metadata_rows,
    load_bnt_data,
    load_fas_data,
    load_svf_data,
    load_svf_user_meta,
    load_fas_user_meta,
)


V3_BNT_PATH = BNT_PATH
V3_SVF_PATH = SVF_PATH
V3_FAS_PATH = FAS_PATH
V2_BNT_PATH = DATA_DIR / "BNT-syntheticData_v2.xlsx"


# ──────────────────────────────────────────────────────
# BNT loader (v3 + v2)
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bnt_v3():
    return load_bnt_data(V3_BNT_PATH)


def test_load_bnt_returns_tuple(bnt_v3):
    items, meta = bnt_v3
    assert isinstance(items, pd.DataFrame)
    assert isinstance(meta, pd.DataFrame)


def test_load_bnt_user_meta_columns(bnt_v3):
    _, meta = bnt_v3
    assert list(meta.columns) == ["user", "gender", "age", "diagnosis", "mmse"]


def test_load_bnt_v3_includes_mmse(bnt_v3):
    """v3 BNT data: user_meta has an 'mmse' column with numeric values."""
    _, user_meta = bnt_v3
    assert "mmse" in user_meta.columns
    assert pd.api.types.is_numeric_dtype(user_meta["mmse"])
    assert user_meta["mmse"].notna().sum() > 0
    assert user_meta["mmse"].dropna().between(0, 30).all()


def test_load_bnt_diagnosis_values(bnt_v3):
    _, meta = bnt_v3
    valid = {"HC", "MCI", "non-AD", "AD"}
    bad = set(meta["diagnosis"].dropna().unique()) - valid
    assert not bad, f"Unexpected diagnosis values: {bad}"


def test_load_bnt_gold_lowercased(bnt_v3):
    items, _ = bnt_v3
    for g in items["gold"]:
        assert g == g.lower(), f"Gold word not lowercased: {g!r}"


@pytest.mark.skipif(
    not V2_BNT_PATH.exists(), reason="v2 BNT file not present in this checkout"
)
def test_load_v2_data_returns_nan_mmse():
    """Legacy v2 data without MMSE row: 'mmse' column is all NaN."""
    _, user_meta = load_bnt_data(V2_BNT_PATH)
    assert "mmse" in user_meta.columns
    assert user_meta["mmse"].isna().all()


# ──────────────────────────────────────────────────────
# SVF loader (v3) — list-of-dicts contract preserved
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def svf_participants():
    return load_svf_data(V3_SVF_PATH)


def test_svf_participant_count(svf_participants):
    assert len(svf_participants) == 100


def test_svf_record_keys(svf_participants):
    required = {
        "participant_id",
        "responses",
        "gender",
        "age",
        "diagnosis",
        "mmse",
    }
    for p in svf_participants:
        assert required == set(p.keys()), f"Missing keys in {p['participant_id']}"


def test_svf_participant_ids(svf_participants):
    ids = [p["participant_id"] for p in svf_participants]
    assert "User-1" in ids
    assert "User-100" in ids


def test_svf_responses_are_strings(svf_participants):
    for p in svf_participants:
        for r in p["responses"]:
            assert isinstance(r, str), (
                f"Non-string response in {p['participant_id']}: {r!r}"
            )
            assert r  # no empty strings


def test_svf_responses_lowercased(svf_participants):
    for p in svf_participants:
        for r in p["responses"]:
            assert r == r.lower(), f"Response not lowercased: {r!r}"


def test_svf_diagnosis_values(svf_participants):
    valid_diagnoses = {"HC", "MCI", "non-AD", "AD"}
    for p in svf_participants:
        assert p["diagnosis"] in valid_diagnoses, (
            f"Unexpected diagnosis '{p['diagnosis']}' for {p['participant_id']}"
        )


def test_svf_age_is_float_or_none(svf_participants):
    for p in svf_participants:
        assert p["age"] is None or isinstance(p["age"], float)


def test_svf_no_none_in_responses(svf_participants):
    for p in svf_participants:
        assert None not in p["responses"]


def test_load_svf_v3_includes_mmse(svf_participants):
    """Phase 4a sanity check: SVF loader exposes mmse for each participant.

    SVF/FAS loaders return list-of-dicts (back-compat with the existing
    pipelines); the per-test user_meta DataFrame is available via
    ``load_svf_user_meta`` for callers that need it.
    """
    for p in svf_participants:
        assert "mmse" in p
    user_meta = load_svf_user_meta(V3_SVF_PATH)
    assert "mmse" in user_meta.columns
    assert pd.api.types.is_numeric_dtype(user_meta["mmse"])
    assert user_meta["mmse"].notna().sum() > 0


# ──────────────────────────────────────────────────────
# FAS loader (v3) — list-of-dicts contract preserved
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fas_participants():
    return load_fas_data(V3_FAS_PATH)


def test_fas_participant_count(fas_participants):
    assert len(fas_participants) == 100


def test_fas_record_keys(fas_participants):
    required = {
        "participant_id",
        "responses_f",
        "responses_a",
        "responses_s",
        "flagged_errors",
        "gender",
        "age",
        "diagnosis",
        "mmse",
    }
    for p in fas_participants:
        assert required == set(p.keys()), f"Missing keys in {p['participant_id']}"


def test_fas_response_lists_same_length(fas_participants):
    for p in fas_participants:
        n_f = len(p["responses_f"])
        n_a = len(p["responses_a"])
        n_s = len(p["responses_s"])
        assert n_f == n_a == n_s, (
            f"{p['participant_id']}: F/A/S lists have different lengths "
            f"({n_f}, {n_a}, {n_s})"
        )


def test_fas_responses_lowercased(fas_participants):
    for p in fas_participants:
        for key in ("responses_f", "responses_a", "responses_s"):
            for word in p[key]:
                if word:
                    assert word == word.lower(), f"Not lowercased: {word!r}"


def test_fas_no_angle_brackets_in_responses(fas_participants):
    for p in fas_participants:
        for key in ("responses_f", "responses_a", "responses_s"):
            for word in p[key]:
                assert "<" not in word and ">" not in word, (
                    f"Angle brackets not stripped in {p['participant_id']}: {word!r}"
                )


def test_fas_flagged_errors_format(fas_participants):
    for p in fas_participants:
        for err in p["flagged_errors"]:
            assert err.startswith("<") and err.endswith(">"), (
                f"Flagged error not in <...> form: {err!r}"
            )


def test_fas_some_flagged_errors_exist(fas_participants):
    total_flagged = sum(len(p["flagged_errors"]) for p in fas_participants)
    assert total_flagged > 0, "Expected at least some flagged errors in the data"


def test_fas_diagnosis_values(fas_participants):
    valid_diagnoses = {"HC", "MCI", "non-AD", "AD"}
    for p in fas_participants:
        assert p["diagnosis"] in valid_diagnoses


def test_load_fas_v3_includes_mmse(fas_participants):
    """Phase 4a sanity check: FAS loader exposes mmse for each participant."""
    for p in fas_participants:
        assert "mmse" in p
    user_meta = load_fas_user_meta(V3_FAS_PATH)
    assert "mmse" in user_meta.columns
    assert pd.api.types.is_numeric_dtype(user_meta["mmse"])
    assert user_meta["mmse"].notna().sum() > 0


# ──────────────────────────────────────────────────────
# Label-based metadata detection
# ──────────────────────────────────────────────────────

def test_metadata_row_detection_label_based(tmp_path):
    """Metadata rows are detected by label, not by hardcoded index.

    Build a small synthetic spreadsheet whose metadata rows live at
    non-default positions and verify ``_extract_metadata_rows`` picks
    them up. The label-based extractor must also accept ``Kategori`` /
    ``Category`` interchangeably.
    """
    raw = pd.DataFrame(
        {
            "Gold": ["alpha", "beta", None, "Gender:", "Age:", "Kategori:", "MMSE:"],
            "User-1": ["a1", "b1", None, "F", 70, "HC", 28],
            "User-2": ["a2", "b2", None, "M", 65, "MCI", 24],
        }
    )
    meta = _extract_metadata_rows(raw)
    assert set(meta) == {"Gender", "Age", "Category", "MMSE"}
    assert list(meta["Gender"].values) == ["F", "M"]
    assert list(meta["MMSE"].values) == [28, 24]


def test_metadata_row_detection_handles_category_alias(tmp_path):
    """Both 'Kategori' (Swedish) and 'Category' (English) normalize to 'Category'."""
    raw = pd.DataFrame(
        {
            "Gold": ["Category:"],
            "User-1": ["HC"],
            "User-2": ["AD"],
        }
    )
    meta_en = _extract_metadata_rows(raw)
    raw2 = pd.DataFrame(
        {
            "Gold": ["Kategori:"],
            "User-1": ["HC"],
            "User-2": ["AD"],
        }
    )
    meta_sv = _extract_metadata_rows(raw2)
    assert "Category" in meta_en and "Category" in meta_sv
    assert list(meta_en["Category"].values) == list(meta_sv["Category"].values)


# ──────────────────────────────────────────────────────
# Per-test metadata independence (harmonization warning)
# ──────────────────────────────────────────────────────

def test_per_test_metadata_independence():
    """SVF and BNT meta for the same User-N may differ (data is unharmonized).

    This test exists to document that no cross-file invariant should be
    expected; it does not assert metadata equality. A non-zero mismatch
    count is the empirically-observed state per
    data/processed/harmonization_check_v3.csv.
    """
    _, bnt_meta = load_bnt_data(V3_BNT_PATH)
    svf_meta = load_svf_user_meta(V3_SVF_PATH)
    bnt_indexed = bnt_meta.set_index("user")
    svf_indexed = svf_meta.set_index("user")
    common = bnt_indexed.index.intersection(svf_indexed.index)
    assert len(common) > 0, "BNT and SVF should share at least some User-N labels"
    n_mismatched = (
        (bnt_indexed.loc[common, "diagnosis"]
         != svf_indexed.loc[common, "diagnosis"])
        .sum()
    )
    print(f"Diagnosis mismatches across BNT/SVF: {n_mismatched} / {len(common)}")
