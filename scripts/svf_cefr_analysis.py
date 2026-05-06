"""svf_cefr_analysis.py

Tests whether AD speakers' Kelly-covered SVF vocabulary skews toward
simpler CEFR levels than that of HC / MCI / non-AD speakers.

Pre-registered methodological decisions (do not change post-hoc):

1. **CEFR-share denominators are Kelly-covered tokens, not all tokens.**
   The question is whether the *covered* vocabulary distribution differs
   by diagnosis, not whether all responses get CEFR-tagged. Coverage is
   reported as a separate sanity check (Kelly audit averaged ~35%; if a
   diagnosis group's coverage diverges sharply from that, every share
   metric for that group should be read with caution).

2. **Spearman, not Pearson.** Consistent with svf_linz_regression.py:
   MMSE has known ceiling effects and a heavily skewed distribution;
   Pearson's linearity assumption is misleading.

3. **Linear CEFR mapping (A1=1, A2=2, B1=3, B2=4, C1=5, C2=6).** The
   simplest defensible ordinal-scale weighting; cefr_score is a
   token-weighted mean over Kelly-covered tokens. Lower = simpler
   vocabulary. Single-number summary for correlation with MMSE.

4. **Lenient lookup with suffix-stripping** (-en/-et/-n/-t), matching
   the Kelly coverage audit (audit_kelly_coverage.py). Definite-form
   inflections like 'hunden' resolve to the 'hund' lemma rather than
   missing, since the audit established this is the dominant Swedish
   inflection in animal-fluency responses.

Per-participant features computed:

- ``a1_share`` … ``c2_share`` — fraction of Kelly-covered tokens at each
  CEFR level (six values; sum to 1.0 by construction). NaN when the
  participant has zero Kelly-covered tokens.
- ``coverage`` — fraction of all responses found in Kelly. Sanity check.
- ``cefr_score`` — token-weighted mean over Kelly-covered tokens, with
  A1=1 … C2=6. Lower = simpler.

Group-level / correlation tests:

- Kruskal–Wallis H/p across HC/MCI/non-AD/AD for each share + cefr_score.
- Spearman r/p between MMSE and each share + cefr_score.

Outputs:
- data/processed/cefr/cefr_per_participant.csv
- data/processed/cefr/cefr_by_diagnosis.csv  (means + KW H/p per metric)
- data/processed/cefr/cefr_mmse_correlation.csv
- figures/cefr/cefr_a1_share_by_diagnosis.png
- figures/cefr/cefr_score_vs_mmse.png

Headline numbers (logged to stdout): K-W H/p for a1_share, K-W H/p for
cefr_score, Spearman r/p for cefr_score vs MMSE.

Usage:
    python scripts/svf_cefr_analysis.py
    python scripts/svf_cefr_analysis.py --svf-data path/to.xlsx
    python scripts/svf_cefr_analysis.py --kelly-path path/to/kelly.xml
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
from scipy.stats import kruskal, spearmanr  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("svf_cefr_analysis")


# ──────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────

CEFR_LEVELS: tuple[str, ...] = ("A1", "A2", "B1", "B2", "C1", "C2")
CEFR_NUMERIC: dict[str, int] = {lvl: i + 1 for i, lvl in enumerate(CEFR_LEVELS)}
SHARE_COLUMNS: tuple[str, ...] = tuple(f"{lvl.lower()}_share" for lvl in CEFR_LEVELS)
METRIC_COLUMNS: tuple[str, ...] = (*SHARE_COLUMNS, "cefr_score")
DIAGNOSIS_ORDER: tuple[str, ...] = ("HC", "MCI", "non-AD", "AD")

DIAGNOSIS_COLOURS = {
    "HC": "#2ca02c",
    "MCI": "#ff7f0e",
    "non-AD": "#9467bd",
    "AD": "#d62728",
}

DEFAULT_SVF_XLSX = Path("data/xlsx/sweSVF-syntheticData_v3.xlsx")
DEFAULT_KELLY_PATH = Path("data/lexical/kelly.xml")
DEFAULT_OUTPUT_DIR = Path("data/processed/cefr")
DEFAULT_FIGURE_DIR = Path("figures/cefr")


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Per-participant CEFR-distribution analysis on Kelly-covered "
            "SVF responses, with Kruskal–Wallis by diagnosis and Spearman "
            "vs MMSE."
        ),
    )
    parser.add_argument(
        "--svf-data", type=Path, default=DEFAULT_SVF_XLSX,
        help="SVF XLSX with per-participant responses (default: v3 file).",
    )
    parser.add_argument(
        "--kelly-path", type=Path, default=DEFAULT_KELLY_PATH,
        help="Kelly LMF XML (default: data/lexical/kelly.xml).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR,
    )
    return parser


# ──────────────────────────────────────────────────────
# Per-participant metrics
# ──────────────────────────────────────────────────────

def per_participant_metrics(
    participants: list[dict],
    provider,
) -> pd.DataFrame:
    """Compute coverage, six CEFR shares, and cefr_score for each participant.

    Iterates over raw response *tokens* (with repetitions kept), since
    the question is about the speaker's vocabulary mix on the task, not
    type-level diversity. A token is "covered" if Kelly returns any
    entry for it (lenient lookup); shares are fractions over covered
    tokens; cefr_score is a token-weighted mean of CEFR_NUMERIC over
    covered tokens. Missing CEFR (e.g., entry without a level tag) is
    excluded from cefr_score's denominator but counts toward coverage.
    """
    rows: list[dict] = []
    for p in participants:
        responses = [str(w).strip().lower() for w in p["responses"] if w]
        n_tokens = len(responses)
        cefr_counts: dict[str, int] = {lvl: 0 for lvl in CEFR_LEVELS}
        n_covered = 0
        weighted_sum = 0.0
        n_for_score = 0
        for w in responses:
            zipf = provider.zipf_frequency(w)
            level = provider.cefr_level(w)
            if zipf > 0 or level is not None:
                n_covered += 1
            if level in cefr_counts:
                cefr_counts[level] += 1
                weighted_sum += CEFR_NUMERIC[level]
                n_for_score += 1

        row: dict = {
            "participant_id": p["participant_id"],
            "diagnosis": p["diagnosis"],
            "mmse": p["mmse"],
            "n_responses": n_tokens,
            "n_kelly_covered": n_covered,
            "coverage": (n_covered / n_tokens) if n_tokens else float("nan"),
        }
        if n_for_score == 0:
            for lvl in CEFR_LEVELS:
                row[f"{lvl.lower()}_share"] = float("nan")
            row["cefr_score"] = float("nan")
        else:
            for lvl in CEFR_LEVELS:
                row[f"{lvl.lower()}_share"] = cefr_counts[lvl] / n_for_score
            row["cefr_score"] = weighted_sum / n_for_score

        rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────
# Group + correlation analyses
# ──────────────────────────────────────────────────────

def kruskal_wallis_by_diagnosis(
    df: pd.DataFrame,
    metrics: tuple[str, ...] = METRIC_COLUMNS,
) -> pd.DataFrame:
    """Per-metric: per-diagnosis means + Kruskal–Wallis H/p across groups."""
    out_rows = []
    diagnoses = [d for d in DIAGNOSIS_ORDER if d in df["diagnosis"].unique()]
    for metric in metrics:
        per_diag_values: list[np.ndarray] = []
        means: dict[str, float] = {}
        ns: dict[str, int] = {}
        for diag in diagnoses:
            vals = (
                df.loc[df["diagnosis"] == diag, metric]
                .dropna()
                .astype(float)
                .to_numpy()
            )
            per_diag_values.append(vals)
            means[diag] = float(vals.mean()) if vals.size else float("nan")
            ns[diag] = int(vals.size)
        non_empty = [v for v in per_diag_values if v.size > 0]
        if len(non_empty) >= 2 and all(v.size > 0 for v in non_empty):
            try:
                h, p = kruskal(*non_empty)
                h, p = float(h), float(p)
            except ValueError:
                h, p = float("nan"), float("nan")
        else:
            h, p = float("nan"), float("nan")

        row = {"metric": metric, "kw_H": h, "kw_p": p}
        for diag in DIAGNOSIS_ORDER:
            row[f"{diag}_mean"] = means.get(diag, float("nan"))
            row[f"{diag}_n"] = ns.get(diag, 0)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def mmse_spearman(
    df: pd.DataFrame,
    metrics: tuple[str, ...] = METRIC_COLUMNS,
) -> pd.DataFrame:
    """Spearman r/p between MMSE and each metric, dropping NaN per pair."""
    rows = []
    for metric in metrics:
        sub = df[[metric, "mmse"]].dropna()
        if len(sub) < 3:
            r, p = float("nan"), float("nan")
        else:
            r_val, p_val = spearmanr(sub[metric], sub["mmse"])
            r, p = float(r_val), float(p_val)
        rows.append({
            "metric": metric,
            "spearman_r": r,
            "p_value": p,
            "n": len(sub),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────

def plot_a1_share_box(df: pd.DataFrame, path: Path) -> None:
    diagnoses = [d for d in DIAGNOSIS_ORDER if d in df["diagnosis"].unique()]
    data = [
        df.loc[df["diagnosis"] == d, "a1_share"].dropna().to_numpy()
        for d in diagnoses
    ]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot(data, tick_labels=diagnoses, patch_artist=True)
    for patch, diag in zip(bp["boxes"], diagnoses):
        patch.set_facecolor(DIAGNOSIS_COLOURS.get(diag, "#cccccc"))
        patch.set_alpha(0.7)
    ax.set_ylabel("A1 share of Kelly-covered tokens")
    ax.set_xlabel("Diagnosis")
    ax.set_title("CEFR A1 share by diagnosis (Kelly-covered SVF tokens)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_score_vs_mmse(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sub = df.dropna(subset=["mmse", "cefr_score"])
    for diag, colour in DIAGNOSIS_COLOURS.items():
        m = sub["diagnosis"] == diag
        if not m.any():
            continue
        ax.scatter(
            sub.loc[m, "mmse"], sub.loc[m, "cefr_score"],
            c=colour, label=diag, alpha=0.75, edgecolor="black",
            linewidth=0.4, s=40,
        )
    if len(sub) >= 3:
        r_val, p_val = spearmanr(sub["cefr_score"], sub["mmse"])
        ax.set_title(
            f"CEFR score vs MMSE  (Spearman r = {float(r_val):+.3f}, "
            f"p = {float(p_val):.3g}, n = {len(sub)})"
        )
    else:
        ax.set_title("CEFR score vs MMSE")
    ax.set_xlabel("MMSE")
    ax.set_ylabel("CEFR score (A1=1 … C2=6, Kelly-covered tokens)")
    ax.legend(title="Diagnosis", loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ──────────────────────────────────────────────────────
# Top-level
# ──────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

    if not args.svf_data.is_file():
        logger.error("SVF XLSX not found: %s", args.svf_data)
        return 2
    if not args.kelly_path.is_file():
        logger.error("Kelly XML not found: %s", args.kelly_path)
        return 2

    from src.thesis_project.lexical.word_frequency import WordFrequencyProvider
    from src.thesis_project.preprocessing.data_loader import load_svf_data

    provider = WordFrequencyProvider(
        source="kelly", kelly_path=args.kelly_path, kelly_lenient=True,
    )
    participants = load_svf_data(args.svf_data)
    logger.info("Loaded %d participants from %s", len(participants), args.svf_data)

    per = per_participant_metrics(participants, provider)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    per_path = args.output_dir / "cefr_per_participant.csv"
    per.to_csv(per_path, index=False)
    logger.info("Wrote per-participant table → %s", per_path)

    by_diag = kruskal_wallis_by_diagnosis(per)
    by_diag_path = args.output_dir / "cefr_by_diagnosis.csv"
    by_diag.to_csv(by_diag_path, index=False)
    logger.info("Wrote diagnosis-group table → %s", by_diag_path)

    corr = mmse_spearman(per)
    corr_path = args.output_dir / "cefr_mmse_correlation.csv"
    corr.to_csv(corr_path, index=False)
    logger.info("Wrote MMSE correlation table → %s", corr_path)

    box_path = args.figure_dir / "cefr_a1_share_by_diagnosis.png"
    plot_a1_share_box(per, box_path)
    logger.info("Wrote box plot → %s", box_path)

    scatter_path = args.figure_dir / "cefr_score_vs_mmse.png"
    plot_score_vs_mmse(per, scatter_path)
    logger.info("Wrote scatter → %s", scatter_path)

    # ── Headline numbers to stdout ──
    a1_row = by_diag.loc[by_diag["metric"] == "a1_share"].iloc[0]
    score_row = by_diag.loc[by_diag["metric"] == "cefr_score"].iloc[0]
    score_corr = corr.loc[corr["metric"] == "cefr_score"].iloc[0]

    print()
    print("=" * 64)
    print("CEFR analysis — headline numbers")
    print("=" * 64)
    print(f"Coverage (mean across participants): "
          f"{per['coverage'].mean():.1%}")
    print()
    print(f"Kruskal–Wallis  a1_share    H = {a1_row['kw_H']:.3f}  "
          f"p = {a1_row['kw_p']:.4g}")
    print(f"Kruskal–Wallis  cefr_score  H = {score_row['kw_H']:.3f}  "
          f"p = {score_row['kw_p']:.4g}")
    print(f"Spearman cefr_score ↔ MMSE  "
          f"r = {score_corr['spearman_r']:+.3f}  "
          f"p = {score_corr['p_value']:.4g}  "
          f"n = {int(score_corr['n'])}")

    print()
    print("Per-diagnosis means:")
    for diag in DIAGNOSIS_ORDER:
        if f"{diag}_mean" not in by_diag.columns:
            continue
        a1 = a1_row[f"{diag}_mean"]
        score = score_row[f"{diag}_mean"]
        n = int(a1_row[f"{diag}_n"])
        print(f"  {diag:<8} n={n:>3d}  a1_share={a1:.3f}  cefr_score={score:.3f}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
