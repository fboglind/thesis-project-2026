"""test_data_loaders.py

Integration tests for SVF and FAS data loaders against the real XLSX files.
These tests verify that the loaders produce the expected structure and types.
"""

import pytest

from thesis_project.preprocessing.data_loader import (
    FAS_PATH,
    SVF_PATH,
    load_fas_data,
    load_svf_data,
)


# ──────────────────────────────────────────────────────
# SVF loader
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def svf_participants():
    return load_svf_data(SVF_PATH)


def test_svf_participant_count(svf_participants):
    assert len(svf_participants) == 100


def test_svf_record_keys(svf_participants):
    required = {"participant_id", "responses", "gender", "age", "diagnosis"}
    for p in svf_participants:
        assert required == set(p.keys()), f"Missing keys in {p['participant_id']}"


def test_svf_participant_ids(svf_participants):
    ids = [p["participant_id"] for p in svf_participants]
    assert "User-1" in ids
    assert "User-100" in ids


def test_svf_responses_are_strings(svf_participants):
    for p in svf_participants:
        for r in p["responses"]:
            assert isinstance(r, str), f"Non-string response in {p['participant_id']}: {r!r}"
            assert r  # no empty strings (Nones should have been stripped)


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


# ──────────────────────────────────────────────────────
# FAS loader
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fas_participants():
    return load_fas_data(FAS_PATH)


def test_fas_participant_count(fas_participants):
    assert len(fas_participants) == 100


def test_fas_record_keys(fas_participants):
    required = {
        "participant_id", "responses_f", "responses_a", "responses_s",
        "flagged_errors", "gender", "age", "diagnosis",
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
                if word:  # skip empty strings
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


def test_fas_empty_strings_preserved(fas_participants):
    # At least some participants should have empty-string slots (not all responses
    # are non-empty across all three letters at every time point)
    has_empty = any(
        "" in p["responses_f"] or "" in p["responses_a"] or "" in p["responses_s"]
        for p in fas_participants
    )
    assert has_empty, "Expected some empty-string slots in FAS responses"


def test_fas_diagnosis_values(fas_participants):
    valid_diagnoses = {"HC", "MCI", "non-AD", "AD"}
    for p in fas_participants:
        assert p["diagnosis"] in valid_diagnoses
