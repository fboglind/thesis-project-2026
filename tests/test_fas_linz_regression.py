"""Smoke + dry-run tests for scripts/fas_linz_regression.py.

Mirrors tests/test_svf_linz_regression.py but uses the four-feature
FAS schema and FAS output paths.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "fas_linz_regression.py"


def make_synthetic_fas_csv(tmp_path: Path) -> Path:
    """Build a small synthetic FAS results CSV with one survivable feature.

    Five participants per diagnosis (HC, MCI, non-AD, AD) so the 5-fold
    StratifiedKFold by diagnosis is feasible. total_fas_score and MMSE
    are correlated so the |r|>0.3 filter has at least one survivor.
    """
    rng = np.random.default_rng(0)
    diagnoses = ["HC", "MCI", "non-AD", "AD"]
    diag_to_mmse_mean = {"HC": 28.5, "MCI": 25.5, "non-AD": 22.0, "AD": 18.0}
    rows = []
    pid = 0
    per_group = 5
    for diag in diagnoses:
        for _ in range(per_group):
            pid += 1
            mmse_mean = diag_to_mmse_mean[diag]
            mmse = float(np.clip(rng.normal(mmse_mean, 1.5), 0, 30))
            wc = float(np.clip(rng.normal(mmse - 6, 2), 1, 50))
            switches = int(np.clip(rng.normal(wc * 0.7, 2), 0, wc))
            rows.append({
                "participant_id": f"User-{pid}",
                "diagnosis": diag,
                "age": float(rng.integers(55, 85)),
                "gender": rng.choice(["M", "F"]),
                "mmse": mmse,
                # Pipeline-only columns we don't use here, but the
                # validator only checks the four feature columns + diag + mmse.
                "total_fas_score": wc,
                "mean_cluster_size": float(rng.uniform(1.0, 1.4)),
                "switch_count_total": switches,
                "mean_word_frequency": float(np.clip(
                    rng.normal(4.0, 0.2), 1.0, 6.0,
                )),
            })
    out = tmp_path / "fas_results_with_mmse.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def test_fas_linz_regression_dry_run_lists_planned_analysis(tmp_path):
    """--dry-run prints the planned analysis without executing."""
    fixture = make_synthetic_fas_csv(tmp_path)
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
    assert "4 FAS Linz features" in out
    assert "SVR" in out
    assert "Ridge" in out
    assert "1000 bootstrap" in out
    assert "MMSE = 24" in out


def test_fas_linz_regression_end_to_end_smoke(tmp_path):
    """Full run on a small fixture; verify all four output artefacts exist."""
    fixture = make_synthetic_fas_csv(tmp_path)
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
    assert (out_dir / "linz_fas_correlation_table.csv").is_file()
    assert (out_dir / "linz_fas_regression_mae.csv").is_file()
    assert (out_dir / "linz_fas_confusion_matrix.csv").is_file()
    assert (out_dir / "linz_fas_kappa.csv").is_file()
    assert (fig_dir / "fas_predicted_vs_actual.png").is_file()

    mae = pd.read_csv(out_dir / "linz_fas_regression_mae.csv")
    assert set(mae["model"]) == {"SVR", "Ridge"}
    # Sanity: MAE on 0-30 scale should be well under the all-zero/30 floor.
    assert (mae["mae_mean"] < 8).all()


def test_fas_linz_regression_errors_when_no_features_pass(tmp_path):
    """Uncorrelated features → either skip cleanly (exit 2) or run if
    random noise happened to cross the threshold; never crash."""
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
                "mmse": float(rng.uniform(0, 30)),
                "total_fas_score": float(rng.uniform(0, 30)),
                "mean_cluster_size": float(rng.uniform(1.0, 1.4)),
                "switch_count_total": float(rng.uniform(0, 30)),
                "mean_word_frequency": float(rng.uniform(0, 6)),
            })
    fixture = tmp_path / "uncorrelated.csv"
    pd.DataFrame(rows).to_csv(fixture, index=False)

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
    assert result.returncode in (0, 2), result.stderr
