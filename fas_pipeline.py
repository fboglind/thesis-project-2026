"""fas_pipeline.py
FAS (Phonemic Verbal Fluency) Scoring Pipeline
===============================================
Scores participant responses for the FAS test (words beginning with F, A, S).

Two scorers run side by side:
    - Count-based (Tallberg 2008 / Borkowski 1967): totals, proper-noun
      flagging, repetitions, total FAS score, letter asymmetry.
    - Phase 4c chain-method clustering (Troyer 1997 + Pakhomov 2016)
      using the Swedish-orthographic rules in
      ``src/thesis_project/scoring/phonemic_rules.py``.

Cluster analysis is performed on first-occurrence words only;
repetitions are excluded from clustering but remain counted in
``repetitions_count``. MMSE is propagated from the data loader's
``user_meta`` so the per-participant output CSV is the input to
``scripts/fas_linz_regression.py``.

Usage:
    python fas_pipeline.py --data data/xlsx/sweFAS-syntheticData_v3.xlsx
    python fas_pipeline.py  # uses default path from config
"""

import argparse
import logging
import warnings
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import kruskal

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def _filter_letter(words: list[str], flagged: set[str]) -> list[str]:
    """Strip blanks, proper-noun flags, and duplicates while preserving order."""
    cleaned: list[str] = []
    for w in words:
        if not w:
            continue
        wl = w.strip().lower()
        if not wl or wl in flagged:
            continue
        cleaned.append(wl)
    # Preserve order; remove duplicates (cluster analysis on first
    # occurrences only — repetitions remain counted in repetitions_count).
    return list(dict.fromkeys(cleaned))


def main():
    parser = argparse.ArgumentParser(description="FAS Phonemic Verbal Fluency Scoring")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to FAS XLSX file (default: from config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/fas_results_with_mmse.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info(
        "Per-test metadata only; cross-file joins are not valid on v3 data. "
        "See data/processed/harmonization_check_v3.csv."
    )

    from src.thesis_project.lexical.word_frequency import WordFrequencyProvider
    from src.thesis_project.preprocessing.data_loader import FAS_PATH, load_fas_data
    from src.thesis_project.scoring.fas_scorer import score_fas, score_fas_counts

    data_path = args.data if args.data else FAS_PATH

    config_path = Path(__file__).resolve().parent / "configs" / "_default_configs.yaml"
    with open(config_path, "r") as cfg_f:
        cfg = yaml.safe_load(cfg_f)
    fas_freq_cfg = cfg.get("fas_frequency", {}) or {}
    frequency_source = fas_freq_cfg.get("source", "wordfreq")
    frequency_provider = WordFrequencyProvider(source=frequency_source)

    # ── 1. Load ──────────────────────────────────────────
    print(f"Loading FAS data from {data_path}...")
    participants = load_fas_data(data_path)
    print(f"  {len(participants)} participants loaded.")

    diagnoses = [p["diagnosis"] for p in participants]
    diag_counts = pd.Series(diagnoses).value_counts().to_dict()
    print(f"  Diagnoses: {diag_counts}")

    mmse_values = [p["mmse"] for p in participants]
    n_mmse_present = sum(1 for v in mmse_values if v is not None and not pd.isna(v))
    print(f"  MMSE: {n_mmse_present}/{len(participants)} present")
    if n_mmse_present == 0:
        logger.info(
            "No MMSE values present in FAS data; mmse column will be all NaN. "
            "This is expected for legacy v1/v2 inputs."
        )

    # ── 2. Score ─────────────────────────────────────────
    print("\nScoring FAS responses (counts + chain-method clustering)...")
    records = []
    for p in participants:
        counts = score_fas_counts(
            responses_f=p["responses_f"],
            responses_a=p["responses_a"],
            responses_s=p["responses_s"],
            flagged_errors=p["flagged_errors"],
        )

        # Clustering operates on filtered, deduplicated word lists.
        flagged_words: set[str] = {
            w[1:-1].lower()
            for w in (p.get("flagged_errors") or [])
            if w.startswith("<") and w.endswith(">")
        }
        f_words = _filter_letter(p["responses_f"], flagged_words)
        a_words = _filter_letter(p["responses_a"], flagged_words)
        s_words = _filter_letter(p["responses_s"], flagged_words)

        clusters = score_fas(
            f_words, a_words, s_words,
            word_freq_provider=frequency_provider,
        )

        records.append(
            {
                "participant_id": p["participant_id"],
                "diagnosis": p["diagnosis"],
                "age": p["age"],
                "gender": p["gender"],
                "mmse": p["mmse"],
                **counts,
                **clusters,
            }
        )

    scored = pd.DataFrame(records)

    # ── 3. Summary ───────────────────────────────────────
    diag_order = ["HC", "MCI", "non-AD", "AD"]

    print("\n" + "=" * 70)
    print("RESULTS: FAS Scoring by Diagnostic Group")
    print("=" * 70)

    summary_rows = []
    for diag in diag_order:
        sub = scored[scored["diagnosis"] == diag]
        if sub.empty:
            continue
        summary_rows.append(
            {
                "Diagnosis": diag,
                "N": len(sub),
                "Total_FAS_mean": sub["total_fas_score"].mean(),
                "Total_FAS_std": sub["total_fas_score"].std(),
                "Valid_F_mean": sub["valid_f"].mean(),
                "Valid_A_mean": sub["valid_a"].mean(),
                "Valid_S_mean": sub["valid_s"].mean(),
                "Asymmetry_mean": sub["letter_asymmetry"].mean(),
                "ProperNouns_mean": sub["proper_nouns_count"].mean(),
                "Repetitions_mean": sub["repetitions_count"].mean(),
                "ClusterCount_mean": sub["cluster_count_total"].mean(),
                "SwitchCount_mean": sub["switch_count_total"].mean(),
                "MeanClusterSize_mean": sub["mean_cluster_size"].mean(),
                "MWF_mean": sub["mean_word_frequency"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False, float_format="%.2f"))

    print("\n" + "-" * 70)
    print("Total FAS score range across all participants:")
    print(
        f"  min={scored['total_fas_score'].min()}, "
        f"max={scored['total_fas_score'].max()}, "
        f"mean={scored['total_fas_score'].mean():.2f}"
    )

    # ── 4. Kruskal-Wallis sanity check ────────────────────
    print("\n" + "-" * 70)
    print("Kruskal-Wallis by diagnosis (sanity check; not pass/fail):")
    for metric in ("cluster_count_total", "mean_cluster_size", "switch_count_total"):
        groups = [
            scored.loc[scored["diagnosis"] == d, metric].dropna().values
            for d in diag_order
            if (scored["diagnosis"] == d).any()
        ]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            h, p = kruskal(*groups)
            print(f"  {metric:24s}  H={h:.3f}  p={p:.4g}")
        else:
            print(f"  {metric:24s}  (insufficient groups)")

    # ── 5. Save ──────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_cols = [
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
    scored[output_cols].to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
