"""fas_linz_regression.py

Linz-style SQ4 regression analysis on FAS features against MMSE.

Mirrors :mod:`scripts.svf_linz_regression` (Phase 4b-Linz) but operates
on the FAS output CSV and uses **four** features instead of five — the
SD (semantic distance) feature has no defensible analog for phonemic
fluency and is dropped by pre-registration.

Pre-registered methodological decisions (do not change post-hoc):

1. **Phonemic clustering rules pinned.** R1 (shared first two letters)
   ∪ R2 (shared rime, min length 2) ∪ R3 (vowel-only single-character
   substitution). Troyer's R4 (homonyms) is dropped for the Swedish
   adaptation. See Phase 4c-FAS-Troyer instructions and
   ``src/thesis_project/scoring/phonemic_rules.py``.

2. **Chain method (Pakhomov 2016), not Troyer all-pairwise.**
   Singletons count as size-1 clusters. Aggregates across F/A/S are
   ``cluster_count_total`` (sum), ``switch_count_total`` (sum), and
   ``mean_cluster_size`` (pooled grand mean over per-letter cluster
   sizes — each cluster contributes once).

3. **Feature selection rule: |Spearman r| > 0.3 with MMSE.** Mirrors
   Linz et al. (2017) Table III. The FAS Linz feature set is four
   features (WC, MCS, NOS, MWF); SD is intentionally absent.

4. **Bootstrap MAE: error-based, 1000 iterations.** 5-fold
   StratifiedKFold over diagnosis (not MMSE — MMSE is continuous);
   collect held-out predictions for every participant, then bootstrap
   over (predicted, actual) pairs.

Why Spearman, not Pearson: MMSE has known ceiling effects and a
heavily skewed distribution; Pearson is misleading.

Outputs (default to data/processed/linz_fas/ and figures/linz_fas/):
- linz_fas_correlation_table.csv   Spearman r, p, n, passes_threshold
- linz_fas_regression_mae.csv      per-model MAE with bootstrap CI
- linz_fas_confusion_matrix.csv    2×2 confusion at MMSE=24, stacked
- linz_fas_kappa.csv               Cohen's κ at MMSE=24 per model
- fas_predicted_vs_actual.png      2-panel scatter, diagnosis-coloured

Usage:
    python scripts/fas_linz_regression.py
    python scripts/fas_linz_regression.py --random-seed 42
    python scripts/fas_linz_regression.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

logger = logging.getLogger("fas_linz_regression")


# ──────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────

LINZ_FEATURES: dict[str, str] = {
    # Linz abbreviation → column name in fas_results_with_mmse.csv
    "WC": "total_fas_score",
    "MCS": "mean_cluster_size",
    "NOS": "switch_count_total",
    "MWF": "mean_word_frequency",
    # SD intentionally absent — no defensible phonemic-distance analog.
}

CORRELATION_THRESHOLD = 0.3
MMSE_CUTOFF = 24
BOOTSTRAP_ITERATIONS = 1000
CV_FOLDS = 5

DEFAULT_INPUT = Path("data/processed/fas_results_with_mmse.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/linz_fas")
DEFAULT_FIGURE_DIR = Path("figures/linz_fas")

DIAGNOSIS_COLOURS = {
    "HC": "#2ca02c",
    "MCI": "#ff7f0e",
    "non-AD": "#9467bd",
    "AD": "#d62728",
}


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Linz-style FAS→MMSE regression (SQ4)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned analysis without running it.",
    )
    return parser


# ──────────────────────────────────────────────────────
# Analysis steps
# ──────────────────────────────────────────────────────

def load_and_validate(path: Path) -> pd.DataFrame:
    """Read the input CSV; drop NaN-MMSE rows; validate column presence."""
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)

    required = {"mmse", "diagnosis", *LINZ_FEATURES.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {sorted(missing)}. "
            f"Path: {path}"
        )

    n_total = len(df)
    df_clean = df[df["mmse"].notna()].copy()
    n_dropped = n_total - len(df_clean)
    if n_dropped > 0:
        logger.warning(
            "Dropped %d/%d rows with NaN MMSE.", n_dropped, n_total,
        )
    if n_total > 0 and n_dropped / n_total > 0.10:
        raise RuntimeError(
            f"More than 10% of rows ({n_dropped}/{n_total}) have NaN MMSE; "
            "refusing to run the regression on heavily-truncated data."
        )

    logger.info("Loaded %d participants (after MMSE filtering).", len(df_clean))
    return df_clean


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman r, p, and the |r|>0.3 pass flag for each FAS Linz feature."""
    rows = []
    for label, col in LINZ_FEATURES.items():
        x = df[col].astype(float)
        y = df["mmse"].astype(float)
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            r, p = float("nan"), float("nan")
        else:
            r, p = spearmanr(x[mask], y[mask])
            r = float(r)
            p = float(p)
        passes = bool(np.isfinite(r) and abs(r) > CORRELATION_THRESHOLD)
        rows.append(
            {
                "feature": label,
                "column": col,
                "spearman_r": r,
                "p_value": p,
                "n": int(mask.sum()),
                "passes_threshold": passes,
            }
        )
    table = pd.DataFrame(rows)
    for _, row in table.iterrows():
        logger.info(
            "  %s (%s): r=%.3f, p=%.4f, n=%d, passes=%s",
            row["feature"], row["column"],
            row["spearman_r"], row["p_value"],
            row["n"], row["passes_threshold"],
        )
    return table


def select_features(corr_table: pd.DataFrame) -> list[str]:
    surviving = corr_table[corr_table["passes_threshold"]]["column"].tolist()
    dropped = corr_table[~corr_table["passes_threshold"]]["column"].tolist()
    logger.info("Surviving features (|r|>%.1f): %s",
                CORRELATION_THRESHOLD, surviving)
    logger.info("Dropped features: %s", dropped)
    return surviving


def cv_predict(
    X: np.ndarray,
    y: np.ndarray,
    diagnosis: np.ndarray,
    estimator,
    param_grid: dict,
    seed: int,
    model_label: str,
) -> tuple[np.ndarray, dict]:
    """5-fold StratifiedKFold by diagnosis with inner GridSearchCV.

    StandardScaler is fitted inside each pipeline so feature scaling
    does not leak across folds.
    """
    outer = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    preds = np.full_like(y, np.nan, dtype=float)
    best_params_count: dict = {}

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
    inner_grid = {f"model__{k}": v for k, v in param_grid.items()}

    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X, diagnosis)):
        n_train = len(train_idx)
        inner_splits = max(2, min(3, n_train // 2))
        gs = GridSearchCV(
            pipe,
            inner_grid,
            cv=inner_splits,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )
        gs.fit(X[train_idx], y[train_idx])
        preds[test_idx] = gs.predict(X[test_idx])
        key = tuple(sorted(gs.best_params_.items()))
        best_params_count[key] = best_params_count.get(key, 0) + 1
        logger.info(
            "  [%s] fold %d/%d: best=%s",
            model_label, fold_idx + 1, CV_FOLDS, gs.best_params_,
        )

    most_common = max(best_params_count.items(), key=lambda kv: kv[1])[0]
    summary = dict(most_common)
    logger.info("  [%s] most-common best params: %s", model_label, summary)
    return preds, summary


def bootstrap_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    maes = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        maes[i] = mean_absolute_error(y_true[idx], y_pred[idx])
    return float(maes.mean()), float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))


def run_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    estimator,
    param_grid: dict,
    model_label: str,
    seed: int,
) -> dict:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["mmse"].to_numpy(dtype=float)
    diagnosis = df["diagnosis"].astype(str).to_numpy()

    preds, best_params = cv_predict(
        X, y, diagnosis, estimator, param_grid, seed, model_label,
    )
    mae_mean, mae_lo, mae_hi = bootstrap_mae(y, preds, BOOTSTRAP_ITERATIONS, seed)
    logger.info(
        "  [%s] bootstrap MAE = %.3f  (95%% CI: %.3f – %.3f)",
        model_label, mae_mean, mae_lo, mae_hi,
    )

    true_class = (y >= MMSE_CUTOFF).astype(int)
    pred_class = (preds >= MMSE_CUTOFF).astype(int)
    cm = confusion_matrix(true_class, pred_class, labels=[0, 1])
    kappa = float(cohen_kappa_score(true_class, pred_class))
    logger.info("  [%s] confusion matrix at MMSE=%d:\n%s",
                model_label, MMSE_CUTOFF, cm)
    logger.info("  [%s] Cohen's κ = %.3f", model_label, kappa)

    return {
        "label": model_label,
        "predictions": preds,
        "y_true": y,
        "diagnosis": diagnosis,
        "mae_mean": mae_mean,
        "mae_ci_low": mae_lo,
        "mae_ci_high": mae_hi,
        "best_params": best_params,
        "confusion_matrix": cm,
        "kappa": kappa,
        "n_features": len(feature_cols),
        "features_used": feature_cols,
    }


def write_outputs(
    corr_table: pd.DataFrame,
    results: list[dict],
    output_dir: Path,
    figure_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["correlation"] = output_dir / "linz_fas_correlation_table.csv"
    corr_table.to_csv(paths["correlation"], index=False)

    paths["mae"] = output_dir / "linz_fas_regression_mae.csv"
    pd.DataFrame([
        {
            "model": r["label"],
            "mae_mean": r["mae_mean"],
            "mae_ci_low": r["mae_ci_low"],
            "mae_ci_high": r["mae_ci_high"],
            "n_features": r["n_features"],
            "features_used": ",".join(r["features_used"]),
        }
        for r in results
    ]).to_csv(paths["mae"], index=False)

    cm_rows = []
    for r in results:
        cm = r["confusion_matrix"]
        cm_rows.append({
            "model": r["label"],
            "true_below_pred_below": int(cm[0, 0]),
            "true_below_pred_above": int(cm[0, 1]),
            "true_above_pred_below": int(cm[1, 0]),
            "true_above_pred_above": int(cm[1, 1]),
            "cutoff": MMSE_CUTOFF,
        })
    paths["confusion"] = output_dir / "linz_fas_confusion_matrix.csv"
    pd.DataFrame(cm_rows).to_csv(paths["confusion"], index=False)

    paths["kappa"] = output_dir / "linz_fas_kappa.csv"
    pd.DataFrame([
        {"model": r["label"], "kappa": r["kappa"], "cutoff": MMSE_CUTOFF}
        for r in results
    ]).to_csv(paths["kappa"], index=False)

    paths["scatter"] = figure_dir / "fas_predicted_vs_actual.png"
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5),
                             squeeze=False)
    for ax, r in zip(axes[0], results):
        for diag, colour in DIAGNOSIS_COLOURS.items():
            mask = r["diagnosis"] == diag
            if mask.any():
                ax.scatter(
                    r["y_true"][mask], r["predictions"][mask],
                    c=colour, label=diag, alpha=0.75, edgecolor="black",
                    linewidth=0.4, s=40,
                )
        lo = float(min(r["y_true"].min(), r["predictions"].min()))
        hi = float(max(r["y_true"].max(), r["predictions"].max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6)
        # MMSE = 24 cutoff lines (clinical convention)
        ax.axvline(MMSE_CUTOFF, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axhline(MMSE_CUTOFF, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Actual MMSE")
        ax.set_ylabel("Predicted MMSE")
        ax.set_title(
            f"{r['label']}  (MAE = {r['mae_mean']:.2f}, "
            f"95% CI [{r['mae_ci_low']:.2f}, {r['mae_ci_high']:.2f}])"
        )
        ax.legend(title="Diagnosis", loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("FAS → MMSE: predicted vs. actual (Linz-style regression)")
    fig.tight_layout()
    fig.savefig(paths["scatter"], dpi=300)
    plt.close(fig)

    for label, path in paths.items():
        logger.info("Wrote %s → %s", label, path)
    return paths


# ──────────────────────────────────────────────────────
# Top-level
# ──────────────────────────────────────────────────────

def planned_summary() -> str:
    return (
        "Planned analysis:\n"
        f"  Input CSV: {DEFAULT_INPUT}\n"
        f"  4 FAS Linz features (WC, MCS, NOS, MWF — SD intentionally absent)\n"
        f"  Spearman correlation table; selection rule |r|>{CORRELATION_THRESHOLD}\n"
        f"  Models: SVR (RBF kernel), Ridge\n"
        f"  StratifiedKFold by diagnosis, {CV_FOLDS} folds\n"
        f"  {BOOTSTRAP_ITERATIONS} bootstrap iterations on held-out predictions\n"
        f"  Confusion matrix + Cohen's κ at MMSE = {MMSE_CUTOFF}\n"
        f"  4 output artefacts: correlation table, MAE table, "
        f"confusion matrix CSV, predicted-vs-actual PNG\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

    if args.dry_run:
        print(planned_summary())
        return 0

    df = load_and_validate(args.input)

    logger.info("Spearman correlations with MMSE (n=%d):", len(df))
    corr_table = correlation_table(df)
    surviving_cols = select_features(corr_table)
    if not surviving_cols:
        logger.error(
            "No features pass the |r|>%.1f threshold; cannot run the regression.",
            CORRELATION_THRESHOLD,
        )
        return 2

    n_before = len(df)
    df = df.dropna(subset=surviving_cols).reset_index(drop=True)
    n_dropped_features = n_before - len(df)
    if n_dropped_features:
        logger.warning(
            "Dropped %d/%d participants with NaN in surviving feature(s) %s.",
            n_dropped_features, n_before, surviving_cols,
        )
    if n_before > 0 and n_dropped_features / n_before > 0.10:
        logger.error(
            "More than 10%% of participants (%d/%d) have NaN in a surviving "
            "feature; refusing to run on heavily-truncated data.",
            n_dropped_features, n_before,
        )
        return 3
    if len(df) < CV_FOLDS:
        logger.error(
            "Only %d participants remain after NaN filtering; need ≥ %d for "
            "%d-fold CV.", len(df), CV_FOLDS, CV_FOLDS,
        )
        return 3

    seed = int(args.random_seed)
    np.random.seed(seed)

    results = []
    logger.info("Fitting SVR …")
    results.append(run_model(
        df, surviving_cols,
        estimator=SVR(kernel="rbf"),
        param_grid={
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1, 1.0],
        },
        model_label="SVR",
        seed=seed,
    ))
    logger.info("Fitting Ridge …")
    results.append(run_model(
        df, surviving_cols,
        estimator=Ridge(),
        param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        model_label="Ridge",
        seed=seed,
    ))

    write_outputs(corr_table, results, args.output_dir, args.figure_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
