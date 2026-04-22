"""Threshold calibration for SVF cluster metrics.

Sweeps similarity thresholds and finds the value that maximises
Kruskal-Wallis H for group discrimination on mean_cluster_size,
switch_count, and cluster_count.

The embedding step is the expensive part, so score_svf is called
once per participant and detect_clusters is then called repeatedly
on the pre-computed consecutive similarities for each candidate
threshold.

Usage:
    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --model sbert
    python scripts/calibrate_threshold.py --model kbbert
    python scripts/calibrate_threshold.py --mock
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

MODEL_CHOICES = ["kbbert", "sbert"]
METRICS = ["mean_cluster_size", "switch_count", "cluster_count"]
DIAG_ORDER = ["HC", "MCI", "non-AD", "AD"]


def _build_encoder(model_name: str):
    from thesis_project.embeddings.encoder import (
        KBBertEmbedder,
        SentenceTransformerEmbedder,
    )

    if model_name == "kbbert":
        return KBBertEmbedder()
    if model_name == "sbert":
        return SentenceTransformerEmbedder("KBLab/sentence-bert-swedish-cased")
    raise ValueError(f"Unknown model: {model_name}")


def _kruskal_groups(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    """Return Kruskal-Wallis (H, p) for metric across diagnostic groups.

    Drops NaNs and any group with fewer than 2 values. Returns (nan, nan)
    if fewer than two valid groups remain.
    """
    groups = []
    for diag in DIAG_ORDER:
        vals = df.loc[df["diagnosis"] == diag, metric].dropna().to_numpy()
        if len(vals) >= 2:
            groups.append(vals)
    if len(groups) < 2:
        return float("nan"), float("nan")
    h, p = stats.kruskal(*groups)
    return float(h), float(p)


def main():
    parser = argparse.ArgumentParser(
        description="SVF threshold calibration via Kruskal-Wallis H maximisation"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to SVF XLSX file (default: from config)",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="sbert",
        help="Embedding model (ignored if --mock). Default: sbert",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock embeddings (no GPU required; scores not meaningful)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/threshold_calibration.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--min-threshold", type=float, default=0.30, help="Sweep start (inclusive)"
    )
    parser.add_argument(
        "--max-threshold", type=float, default=0.70, help="Sweep end (inclusive)"
    )
    parser.add_argument(
        "--step", type=float, default=0.025, help="Sweep step size"
    )
    args = parser.parse_args()

    from thesis_project.embeddings.encoder import MockEmbedder
    from thesis_project.preprocessing.data_loader import SVF_PATH, load_svf_data
    from thesis_project.scoring.svf_scorer import detect_clusters, score_svf

    data_path = args.data if args.data else SVF_PATH

    # ── 1. Load ──────────────────────────────────────────
    print(f"Loading SVF data from {data_path}...")
    participants = load_svf_data(data_path)
    print(f"  {len(participants)} participants loaded.")

    # ── 2. Embedder ───────────────────────────────────────
    if args.mock:
        print("\n⚠ Using MOCK embeddings (random vectors). "
              "Calibration results are NOT meaningful.\n")
        encoder = MockEmbedder()
    else:
        print(f"\nLoading model: {args.model}...")
        encoder = _build_encoder(args.model)
        print("Model loaded.")

    # ── 3. Embed once per participant; cache consecutive similarities ──
    print("\nEmbedding and computing consecutive similarities (once per participant)...")
    cached: list[dict] = []
    for p in participants:
        metrics = score_svf(responses=p["responses"], encoder=encoder)
        cached.append(
            {
                "participant_id": p["participant_id"],
                "diagnosis": p["diagnosis"],
                "responses": p["responses"],
                "consec_sims": metrics["consecutive_similarities"],
            }
        )

    # ── 4. Sweep thresholds ──────────────────────────────
    # np.arange misses the endpoint due to float accumulation; add step/2 guard.
    thresholds = np.arange(
        args.min_threshold, args.max_threshold + args.step / 2, args.step
    )
    print(
        f"\nSweeping {len(thresholds)} thresholds from "
        f"{thresholds[0]:.3f} to {thresholds[-1]:.3f}..."
    )

    sweep_rows = []
    for t in thresholds:
        per_participant = []
        for c in cached:
            cluster_result = detect_clusters(
                c["responses"], c["consec_sims"], threshold=float(t), method="chain"
            )
            per_participant.append(
                {
                    "diagnosis": c["diagnosis"],
                    "mean_cluster_size": cluster_result["mean_cluster_size"],
                    "switch_count": cluster_result["switch_count"],
                    "cluster_count": cluster_result["cluster_count"],
                }
            )
        df_t = pd.DataFrame(per_participant)

        row = {"threshold": float(t)}
        for metric in METRICS:
            h, p = _kruskal_groups(df_t, metric)
            row[f"{metric}_H"] = h
            row[f"{metric}_p"] = p
        sweep_rows.append(row)

    sweep_df = pd.DataFrame(sweep_rows)

    # ── 5. Print per-metric table + optimum ──────────────
    print("\n" + "=" * 72)
    print("Threshold sweep results (Kruskal-Wallis H, p)")
    print("=" * 72)

    for metric in METRICS:
        print(f"\n── {metric} ──")
        subset = sweep_df[["threshold", f"{metric}_H", f"{metric}_p"]].copy()
        subset.columns = ["threshold", "H", "p"]
        print(subset.to_string(index=False, float_format="%.4f"))

        valid = subset.dropna(subset=["H"])
        if valid.empty:
            print(f"  No valid H values for {metric}.")
            continue
        best = valid.loc[valid["H"].idxmax()]
        print(
            f"  → optimal threshold for {metric}: "
            f"{best['threshold']:.3f}  (H={best['H']:.3f}, p={best['p']:.4f})"
        )

    # ── 6. Save ──────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(output_path, index=False)
    print(f"\nCalibration results saved to {output_path}")


if __name__ == "__main__":
    main()
