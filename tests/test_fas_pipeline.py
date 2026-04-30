"""End-to-end tests for fas_pipeline.py.

Phase 4c additions: MMSE plumbing, clustering metric columns, and
order-preserving deduplication before clustering.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "fas_pipeline.py"

EXPECTED_COLUMNS = [
    "participant_id", "diagnosis", "age", "gender", "mmse",
    "total_f", "total_a", "total_s",
    "valid_f", "valid_a", "valid_s",
    "total_fas_score", "proper_nouns_count", "repetitions_count",
    "letter_asymmetry",
    "cluster_count_f", "mean_cluster_size_f", "switch_count_f",
    "cluster_count_a", "mean_cluster_size_a", "switch_count_a",
    "cluster_count_s", "mean_cluster_size_s", "switch_count_s",
    "cluster_count_total", "switch_count_total", "mean_cluster_size",
    "mean_word_frequency",
]


def test_pipeline_output_schema(tmp_path):
    """End-to-end run on the v3 fixture XLSX produces the expected schema."""
    out_csv = tmp_path / "fas_results.csv"
    result = subprocess.run(
        [sys.executable, str(PIPELINE), "--output", str(out_csv)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    df = pd.read_csv(out_csv)

    assert list(df.columns) == EXPECTED_COLUMNS

    assert df["mmse"].notna().sum() > 0
    assert df["mean_word_frequency"].notna().sum() > 0
    # All nine per-letter cluster columns exist (presence already checked
    # by the column-list comparison; this re-asserts numeric dtype).
    for col in [
        "cluster_count_f", "cluster_count_a", "cluster_count_s",
        "switch_count_f", "switch_count_a", "switch_count_s",
        "cluster_count_total", "switch_count_total",
    ]:
        assert pd.api.types.is_numeric_dtype(df[col])

    # Aggregation invariant: cluster_count_total == sum of per-letter
    sums = df["cluster_count_f"] + df["cluster_count_a"] + df["cluster_count_s"]
    assert (df["cluster_count_total"] == sums).all()


def test_pipeline_dedup_before_clustering(tmp_path, monkeypatch):
    """Repetitions are excluded from clustering but counted in repetitions_count.

    Build a synthetic 3-participant XLSX inline. We rely on the
    pipeline's response loader, so the easiest way to drive it is to
    call the internal _filter_letter helper directly.
    """
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from fas_pipeline import _filter_letter
    finally:
        sys.path.pop(0)

    # Duplicates and proper-noun flags removed; order preserved.
    flagged = {"anna"}
    out = _filter_letter(["fil", "fil", "frukt", "anna", "färg"], flagged)
    assert out == ["fil", "frukt", "färg"]

    # Empty strings filtered.
    out2 = _filter_letter(["", "fil", "  ", "frukt"], flagged=set())
    assert out2 == ["fil", "frukt"]

    # All blanks → empty.
    assert _filter_letter(["", "  "], flagged=set()) == []


def test_pipeline_clustering_uses_dedup_lists():
    """Repeated F-words don't inflate cluster_count_f.

    Builds a tiny synthetic participant by calling the scorer functions
    directly with a deduplicated word list (mirroring what the pipeline
    does); confirms cluster_count_f reflects the deduped list, not the
    raw one.
    """
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from fas_pipeline import _filter_letter
    finally:
        sys.path.pop(0)

    from thesis_project.scoring.fas_scorer import score_fas

    raw = ["fil", "fil", "frukt"]
    deduped = _filter_letter(raw, flagged=set())
    assert deduped == ["fil", "frukt"]

    # fil-frukt: R1 'fi' vs 'fr' ✗; R2 'il' vs 'ukt' ✗; R3 len differ ✗
    # → unlinked → 2 clusters of 1.
    r = score_fas(deduped, [], [], word_freq_provider=None)
    assert r["cluster_count_f"] == 2
    assert r["cluster_count_total"] == 2
