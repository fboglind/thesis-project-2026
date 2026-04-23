"""bnt_pipeline.py
BNT Cosine Similarity Scoring Pipeline
=======================================
Computes graded semantic similarity scores for Boston Naming Test responses
using a configurable Swedish embedding model and cosine similarity.

Usage:
    python bnt_pipeline.py --data path/to/BNT-syntheticData_v2.xlsx
    python bnt_pipeline.py --mock                   # mock embeddings (no GPU needed)
    python bnt_pipeline.py --model sbert-swedish    # use Swedish SBERT
    python bnt_pipeline.py --model e5-large         # use multilingual e5-large
    python bnt_pipeline.py                          # defaults (kb-bert, path from config)
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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


# ──────────────────────────────────────────────────────
# 1. PREPROCESSING
# ──────────────────────────────────────────────────────

# Responses that indicate failure to name (not semantically scorable)
NON_RESPONSES = {
    "hhhm jag vet inte", "jag vet inte", "vet inte", "pass",
    "ingen aning", "vet ej", "hmm", "hm", "jag kan inte",
    "något sånt", "nåt sånt",
}


def normalize_response(text: str) -> str | None:
    """Normalize a BNT response for embedding comparison.

    Steps:
        1. Lowercase and strip whitespace
        2. Remove trailing hedges ("kanske", "tror jag")
        3. Remove leading articles ("en", "ett")
        4. Remove leading filler phrases ("det är", "jag tror")
        5. Flag non-responses (returns None)
    """
    if pd.isna(text) or str(text).strip() == "":
        return None

    text = str(text).strip().lower()

    if text in NON_RESPONSES:
        return None

    text = re.sub(r"\s*(kanske|tror jag|eller nåt|är det|va)$", "", text).strip()

    text = re.sub(
        r"^(det är |det ser ut som |jag tror det är |jag tror |en slags |nån slags |typ |liksom |bild på |säkert )",
        "",
        text,
    ).strip()

    text = re.sub(r"^(en|ett|den|det|de)\s+", "", text).strip()

    if text in NON_RESPONSES or text == "" or len(text) < 2:
        return None

    return text


def preprocess_responses(items_df: pd.DataFrame, user_meta: pd.DataFrame) -> pd.DataFrame:
    """Build long-format response table with normalized responses.

    Returns DataFrame with columns:
        gold, user, diagnosis, age, gender, raw_response, normalized,
        is_exact_match, is_non_response
    """
    user_cols = user_meta["user"].tolist()
    meta_lookup = user_meta.set_index("user")

    records = []
    for _, row in items_df.iterrows():
        gold = row["gold"]
        for user in user_cols:
            raw_resp = row[user]
            norm = normalize_response(raw_resp)

            is_exact = False
            if norm is not None:
                is_exact = (norm == gold)
                if not is_exact:
                    tokens = norm.split()
                    is_exact = (tokens[-1] == gold) if tokens else False

            records.append(
                {
                    "gold": gold,
                    "user": user,
                    "diagnosis": meta_lookup.loc[user, "diagnosis"],
                    "age": meta_lookup.loc[user, "age"],
                    "raw_response": raw_resp,
                    "normalized": norm,
                    "is_exact_match": is_exact,
                    "is_non_response": norm is None,
                }
            )

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────
# 2. EMBEDDING & SIMILARITY
# ──────────────────────────────────────────────────────

def compute_similarity_scores(
    responses: pd.DataFrame,
    embedder,
) -> pd.DataFrame:
    """Compute cosine similarity between each response and its gold target.

    For exact matches: score = 1.0
    For non-responses: score = 0.0
    For other responses: cosine_similarity(embed(response), embed(gold))

    Adds columns: cosine_sim, binary_score
    """
    df = responses.copy()
    df["cosine_sim"] = np.nan

    gold_words = df["gold"].unique().tolist()
    print(f"Embedding {len(gold_words)} gold words...")
    gold_embeddings = {w: embedder.embed(w) for w in gold_words}

    needs_embedding = df[~df["is_exact_match"] & ~df["is_non_response"]]["normalized"]
    unique_responses = needs_embedding.dropna().unique().tolist()
    print(f"Embedding {len(unique_responses)} unique responses...")

    resp_embeddings: dict[str, np.ndarray] = {}
    batch_size = 32
    for i in range(0, len(unique_responses), batch_size):
        batch = unique_responses[i:i + batch_size]
        embs = embedder.embed_batch(batch)
        for text, emb in zip(batch, embs):
            resp_embeddings[text] = emb

    scores = []
    for _, row in df.iterrows():
        if row["is_exact_match"]:
            scores.append(1.0)
        elif row["is_non_response"]:
            scores.append(0.0)
        elif row["normalized"] in resp_embeddings:
            gold_emb = gold_embeddings[row["gold"]].reshape(1, -1)
            resp_emb = resp_embeddings[row["normalized"]].reshape(1, -1)
            sim = cosine_similarity(gold_emb, resp_emb)[0, 0]
            # Clamp to [0, 1] — negative cosine similarities → 0
            scores.append(max(0.0, float(sim)))
        else:
            scores.append(0.0)

    df["cosine_sim"] = scores
    df["binary_score"] = df["is_exact_match"].astype(int)

    return df


# ──────────────────────────────────────────────────────
# 3. ANALYSIS
# ──────────────────────────────────────────────────────

def analyze_results(scored: pd.DataFrame) -> None:
    """Print analysis comparing binary vs graded scoring across groups."""

    diag_order = ["HC", "MCI", "non-AD", "AD"]

    print("\n" + "=" * 70)
    print("RESULTS: Binary vs. Graded Scoring by Diagnostic Group")
    print("=" * 70)

    summary = []
    for diag in diag_order:
        sub = scored[scored["diagnosis"] == diag]
        n_users = sub["user"].nunique()
        summary.append(
            {
                "Diagnosis": diag,
                "N_users": n_users,
                "Binary_mean": sub["binary_score"].mean(),
                "Graded_mean": sub["cosine_sim"].mean(),
                "Graded_std": sub.groupby("user")["cosine_sim"].mean().std(),
                "Non_response_rate": sub["is_non_response"].mean(),
            }
        )

    summary_df = pd.DataFrame(summary)
    print("\nPer-group means:")
    print(summary_df.to_string(index=False, float_format="%.3f"))

    print("\n" + "-" * 70)
    print("KEY: Graded - Binary (information gained by graded scoring)")
    print("-" * 70)
    for _, row in summary_df.iterrows():
        diff = row["Graded_mean"] - row["Binary_mean"]
        print(
            f"  {row['Diagnosis']:8s}: +{diff:.3f} "
            f"({'graded captures more' if diff > 0.05 else 'similar'})"
        )

    print("\n" + "-" * 70)
    print("Items with most variation in graded scores (interesting for analysis)")
    print("-" * 70)
    item_stats = scored.groupby("gold").agg(
        binary_mean=("binary_score", "mean"),
        graded_mean=("cosine_sim", "mean"),
        graded_std=("cosine_sim", "std"),
        non_response=("is_non_response", "mean"),
    ).sort_values("graded_std", ascending=False)
    print(item_stats.head(10).to_string(float_format="%.3f"))

    print("\n" + "-" * 70)
    print("Example: Graded scores for 'kamel' by diagnosis")
    print("-" * 70)
    kamel = scored[scored["gold"] == "kamel"]
    for diag in diag_order:
        sub = kamel[kamel["diagnosis"] == diag]
        if len(sub) > 0:
            s = sub["cosine_sim"]
            print(
                f"  {diag:8s}: mean={s.mean():.3f}, "
                f"std={s.std():.3f}, "
                f"min={s.min():.3f}, max={s.max():.3f}"
            )

    print("\n  Sample response scores:")
    kamel_unique = kamel[~kamel["is_exact_match"] & ~kamel["is_non_response"]]
    kamel_unique = kamel_unique.drop_duplicates(subset="normalized")
    kamel_unique = kamel_unique.sort_values("cosine_sim", ascending=False)
    for _, row in kamel_unique.head(10).iterrows():
        print(f"    '{row['normalized']:25s}' → {row['cosine_sim']:.3f} ({row['diagnosis']})")


def save_results(scored: pd.DataFrame, output_path: str) -> None:
    """Save scored results to CSV."""
    scored.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


# ──────────────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BNT Cosine Similarity Scoring")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to BNT XLSX file (default: from config)",
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
        default="data/processed/bnt_scored_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    from src.thesis_project.embeddings.encoder import MockEmbedder
    from src.thesis_project.preprocessing.data_loader import BNT_PATH, load_bnt_data

    data_path = args.data if args.data else BNT_PATH

    # ── 1. Load ──────────────────────────────────────────
    print(f"Loading BNT data from {data_path}...")
    items_df, user_meta = load_bnt_data(data_path)
    print(f"  {len(items_df)} items, {len(user_meta)} users")
    print(f"  Diagnoses: {dict(user_meta['diagnosis'].value_counts())}")

    # ── 2. Preprocess ────────────────────────────────────
    print("\nPreprocessing responses...")
    responses = preprocess_responses(items_df, user_meta)
    n_total = len(responses)
    n_non = responses["is_non_response"].sum()
    n_exact = responses["is_exact_match"].sum()
    n_score = n_total - n_non - n_exact
    print(
        f"  {n_total} total, {n_exact} exact matches, "
        f"{n_non} non-responses, {n_score} to score with embeddings"
    )

    # ── 3. Embedder ───────────────────────────────────────
    if args.mock:
        print("\n⚠ Using MOCK embeddings (random vectors). "
              "Similarity scores are NOT meaningful.")
        print("  Run without --mock to use a real model.\n")
        embedder = MockEmbedder()
    else:
        print(f"\nLoading model: {args.model}...")
        embedder = _build_encoder(args.model)
        print("Model loaded.")

    # ── 4. Score ─────────────────────────────────────────
    print("\nComputing similarity scores...")
    scored = compute_similarity_scores(responses, embedder)

    # ── 5. Analyse ───────────────────────────────────────
    analyze_results(scored)

    # ── 6. Save ──────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(scored, str(output_path))


if __name__ == "__main__":
    main()
