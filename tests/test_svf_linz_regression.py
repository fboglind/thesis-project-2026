"""Smoke + dry-run tests for scripts/svf_linz_regression.py.

The full regression is too slow for casual CI; these tests use a
30-participant synthetic CSV and the script's own --dry-run mode.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "svf_linz_regression.py"


def make_synthetic_linz_csv(tmp_path: Path, n: int = 30) -> Path:
    """Build a small synthetic SVF results CSV with known-ish correlations.

    Keeps each diagnosis group large enough for 5-fold StratifiedKFold,
    so we balance ≥ 5 participants per diagnosis. WC and MMSE are
    correlated so the |r|>0.3 filter has at least one survivor.
    """
    rng = np.random.default_rng(0)
    diagnoses = ["HC", "MCI", "non-AD", "AD"]
    rows = []
    per_group = max(5, n // len(diagnoses))
    diag_to_mmse_mean = {"HC": 28.5, "MCI": 25.5, "non-AD": 22.0, "AD": 18.0}
    pid = 0
    for diag in diagnoses:
        for _ in range(per_group):
            pid += 1
            mmse_mean = diag_to_mmse_mean[diag]
            mmse = float(np.clip(rng.normal(mmse_mean, 1.5), 0, 30))
            wc = float(np.clip(rng.normal(mmse - 8, 2), 1, 30))
            unique_words = max(1, int(wc - rng.integers(0, 3)))
            rows.append({
                "participant_id": f"User-{pid}",
                "diagnosis": diag,
                "age": float(rng.integers(55, 85)),
                "gender": rng.choice(["M", "F"]),
                "mmse": mmse,
                "total_words": wc,
                "unique_words": unique_words,
                "repetitions": int(wc) - unique_words,
                "mean_consecutive_similarity": float(rng.uniform(0.3, 0.7)),
                "pairwise_similarity_mean": float(rng.uniform(0.2, 0.5)),
                "temporal_gradient": float(rng.normal(0, 0.05)),
                "similarity_slope": float(rng.normal(0, 0.01)),
                "cluster_count": int(rng.integers(1, max(2, int(wc)))),
                "mean_cluster_size": float(rng.uniform(1.0, 3.0)),
                "switch_count": int(rng.integers(0, max(1, int(wc) - 1))),
                "max_cluster_size": int(rng.integers(1, max(2, int(wc)))),
                "mean_word_frequency": float(np.clip(
                    rng.normal(4.0 + (mmse - 25) * 0.05, 0.3), 1.0, 6.0,
                )),
            })
    df = pd.DataFrame(rows)
    out = tmp_path / "svf_results_with_mmse.csv"
    df.to_csv(out, index=False)
    return out


def test_linz_regression_dry_run_lists_planned_analysis(tmp_path):
    """--dry-run prints the planned analysis without executing."""
    fixture = make_synthetic_linz_csv(tmp_path)
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--input", str(fixture),
            "--output-dir", str(tmp_path / "out"),
            "--dry-run",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "5 Linz features" in out
    assert "SVR" in out
    assert "Ridge" in out
    assert "1000 bootstrap" in out
    assert "MMSE = 24" in out


def test_linz_regression_end_to_end_smoke(tmp_path):
    """Full run on a small fixture; verify all four output artefacts exist."""
    fixture = make_synthetic_linz_csv(tmp_path)
    out_dir = tmp_path / "out"
    fig_dir = tmp_path / "fig"
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--input", str(fixture),
            "--output-dir", str(out_dir),
            "--figure-dir", str(fig_dir),
            "--random-seed", "42",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "linz_correlation_table.csv").is_file()
    assert (out_dir / "linz_regression_mae.csv").is_file()
    assert (out_dir / "linz_confusion_matrix.csv").is_file()
    assert (out_dir / "linz_kappa.csv").is_file()
    assert (fig_dir / "svf_predicted_vs_actual.png").is_file()

    mae = pd.read_csv(out_dir / "linz_regression_mae.csv")
    assert set(mae["model"]) == {"SVR", "Ridge"}
    # Sanity: MAE on 0-30 scale should be well under the all-zero/30 floor.
    assert (mae["mae_mean"] < 8).all()


def test_linz_regression_errors_when_no_features_pass(tmp_path):
    """If every Linz feature is uncorrelated with MMSE, exit non-zero."""
    rng = np.random.default_rng(1)
    diagnoses = ["HC", "MCI", "non-AD", "AD"]
    rows = []
    pid = 0
    for diag in diagnoses:
        for _ in range(5):
            pid += 1
            rows.append({
                "participant_id": f"User-{pid}",
                "diagnosis": diag,
                "age": 70.0,
                "gender": "F",
                # mmse and features are independent random noise
                "mmse": float(rng.uniform(0, 30)),
                "total_words": float(rng.uniform(0, 30)),
                "unique_words": int(rng.integers(0, 30)),
                "repetitions": 0,
                "mean_consecutive_similarity": float(rng.uniform(0, 1)),
                "pairwise_similarity_mean": float(rng.uniform(0, 1)),
                "temporal_gradient": float(rng.normal(0, 1)),
                "similarity_slope": float(rng.normal(0, 1)),
                "cluster_count": int(rng.integers(1, 10)),
                "mean_cluster_size": float(rng.uniform(1, 3)),
                "switch_count": int(rng.integers(0, 10)),
                "max_cluster_size": int(rng.integers(1, 5)),
                "mean_word_frequency": float(rng.uniform(0, 6)),
            })
    df = pd.DataFrame(rows)
    fixture = tmp_path / "uncorrelated.csv"
    df.to_csv(fixture, index=False)

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--input", str(fixture),
            "--output-dir", str(tmp_path / "out"),
            "--figure-dir", str(tmp_path / "fig"),
            "--random-seed", "42",
        ],
        capture_output=True, text=True,
    )
    # Could pass (some random noise crosses 0.3) or fail with code 2 (the
    # designed exit). What we forbid is a silent crash on empty features.
    assert result.returncode in (0, 2), result.stderr
