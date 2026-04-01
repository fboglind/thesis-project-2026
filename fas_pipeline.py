"""fas_pipeline.py
FAS (Phonemic Verbal Fluency) Scoring Pipeline
===============================================
Scores participant responses for the FAS test (words beginning with F, A, S).
Analysis is count-based following Tallberg et al. (2008) and Borkowski et al. (1967).

Usage:
    python fas_pipeline.py --data data/xlsx/FAS-syntheticData_v1.xlsx
    python fas_pipeline.py  # uses default path from config
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


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
        default="data/processed/fas_scored_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    # Import here so the package is only required at runtime
    from src.thesis_project.preprocessing.data_loader import FAS_PATH, load_fas_data
    from src.thesis_project.scoring.fas_scorer import score_fas

    data_path = args.data if args.data else FAS_PATH

    # ── 1. Load ──────────────────────────────────────────
    print(f"Loading FAS data from {data_path}...")
    participants = load_fas_data(data_path)
    print(f"  {len(participants)} participants loaded.")

    diagnoses = [p["diagnosis"] for p in participants]
    diag_counts = pd.Series(diagnoses).value_counts().to_dict()
    print(f"  Diagnoses: {diag_counts}")

    # ── 2. Score ─────────────────────────────────────────
    print("\nScoring FAS responses...")
    records = []
    for p in participants:
        metrics = score_fas(
            responses_f=p["responses_f"],
            responses_a=p["responses_a"],
            responses_s=p["responses_s"],
            flagged_errors=p["flagged_errors"],
        )
        records.append(
            {
                "participant_id": p["participant_id"],
                "diagnosis": p["diagnosis"],
                "age": p["age"],
                "gender": p["gender"],
                **metrics,
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

    # ── 4. Save ──────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select and order output columns per specification
    output_cols = [
        "participant_id", "diagnosis", "age", "gender",
        "total_f", "total_a", "total_s",
        "valid_f", "valid_a", "valid_s",
        "total_fas_score",
        "proper_nouns_count", "repetitions_count",
        "letter_asymmetry",
    ]
    scored[output_cols].to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
