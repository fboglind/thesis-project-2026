"""svf_pipeline.py
SVF (Semantic Verbal Fluency) Scoring Pipeline
===============================================
Scores participant responses for the SVF animal-fluency test using
cosine similarity metrics following Pakhomov et al. (2016) / Troyer et al. (1997).

Usage:
    python svf_pipeline.py --data data/xlsx/SVF-syntheticData_v1.xlsx
    python svf_pipeline.py --mock                   # mock embeddings (no GPU needed)
    python svf_pipeline.py --model sbert-swedish    # use Swedish SBERT
    python svf_pipeline.py                          # defaults (kb-bert, path from config)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODEL_CHOICES = ["kb-bert", "sbert-swedish", "e5-large", "e5-large-instruct"]


def _build_encoder(model_name: str):
    from src.thesis_project.embeddings.encoder import (
        KBBertEmbedder,
        SentenceTransformerEmbedder,
    )

    if model_name == "kb-bert":
        return KBBertEmbedder()
    if model_name == "sbert-swedish":
        return SentenceTransformerEmbedder("KBLab/sentence-bert-swedish-cased")
    if model_name == "e5-large":
        return SentenceTransformerEmbedder(
            "intfloat/multilingual-e5-large", prefix="query: "
        )
    if model_name == "e5-large-instruct":
        return SentenceTransformerEmbedder(
            "intfloat/multilingual-e5-large-instruct", prefix="query: "
        )
    raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="SVF Semantic Verbal Fluency Scoring")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to SVF XLSX file (default: from config)",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="kb-bert",
        help="Embedding model preset (ignored if --mock). Default: kb-bert",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock embeddings for testing (no GPU required, scores not meaningful)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/svf_scored_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    from src.thesis_project.embeddings.encoder import MockEmbedder
    from src.thesis_project.preprocessing.data_loader import SVF_PATH, load_svf_data
    from src.thesis_project.scoring.svf_scorer import score_svf

    data_path = args.data if args.data else SVF_PATH

    # ── 1. Load ──────────────────────────────────────────
    print(f"Loading SVF data from {data_path}...")
    participants = load_svf_data(data_path)
    print(f"  {len(participants)} participants loaded.")

    diagnoses = [p["diagnosis"] for p in participants]
    diag_counts = pd.Series(diagnoses).value_counts().to_dict()
    print(f"  Diagnoses: {diag_counts}")

    total_responses = sum(len(p["responses"]) for p in participants)
    print(f"  Total responses to embed: {total_responses}")

    # ── 2. Embedder ───────────────────────────────────────
    if args.mock:
        print("\n⚠ Using MOCK embeddings (random vectors). "
              "Similarity scores are NOT meaningful.")
        print("  Run without --mock to use a real model.\n")
        encoder = MockEmbedder()
    else:
        print(f"\nLoading model: {args.model}...")
        encoder = _build_encoder(args.model)
        print("Model loaded.")

    # ── 3. Score ─────────────────────────────────────────
    print("\nScoring SVF responses...")
    records = []
    for p in participants:
        metrics = score_svf(responses=p["responses"], encoder=encoder)
        records.append(
            {
                "participant_id": p["participant_id"],
                "diagnosis": p["diagnosis"],
                "age": p["age"],
                "gender": p["gender"],
                "total_words": metrics["total_words"],
                "unique_words": metrics["unique_words"],
                "repetitions": metrics["repetitions"],
                "mean_consecutive_similarity": metrics["mean_consecutive_similarity"],
                "pairwise_similarity_mean": metrics["pairwise_similarity_mean"],
                "temporal_gradient": metrics["temporal_gradient"],
            }
        )

    scored = pd.DataFrame(records)

    # ── 4. Summary ───────────────────────────────────────
    diag_order = ["HC", "MCI", "non-AD", "AD"]

    print("\n" + "=" * 70)
    print("RESULTS: SVF Scoring by Diagnostic Group")
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
                "TotalWords_mean": sub["total_words"].mean(),
                "UniqueWords_mean": sub["unique_words"].mean(),
                "Repetitions_mean": sub["repetitions"].mean(),
                "MeanConsecSim_mean": sub["mean_consecutive_similarity"].mean(),
                "PairwiseSim_mean": sub["pairwise_similarity_mean"].mean(),
                "TemporalGrad_mean": sub["temporal_gradient"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False, float_format="%.3f"))

    print("\n" + "-" * 70)
    print("Total words range across all participants:")
    print(
        f"  min={scored['total_words'].min()}, "
        f"max={scored['total_words'].max()}, "
        f"mean={scored['total_words'].mean():.1f}"
    )

    # ── 5. Save ──────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
