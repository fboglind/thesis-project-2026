"""graded_scorer.py

Graded semantic similarity scoring using cosine similarity between embeddings.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..embeddings.encoder import Embedder


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        emb1: First embedding vector (1D array).
        emb2: Second embedding vector (1D array).

    Returns:
        Cosine similarity score clamped to [0, 1].
    """
    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0]
    return max(0.0, float(sim))


def compute_similarity_scores(
    responses: pd.DataFrame,
    embedder: Embedder,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Compute cosine similarity between each response and its gold target.

    Expected columns in responses DataFrame:
        - gold: The target/correct answer
        - normalized: The normalized user response
        - is_exact_match: Boolean indicating exact match
        - is_non_response: Boolean indicating non-response (e.g., "I don't know")

    Scoring logic:
        - Exact matches: score = 1.0
        - Non-responses: score = 0.0
        - Other responses: cosine_similarity(embed(response), embed(gold))

    Args:
        responses: DataFrame with response data.
        embedder: Embedder instance for computing text embeddings.
        batch_size: Batch size for embedding computation.

    Returns:
        DataFrame with added columns: cosine_sim, binary_score
    """
    df = responses.copy()
    df["cosine_sim"] = np.nan

    # Pre-embed all gold words
    gold_words = df["gold"].unique().tolist()
    gold_embeddings = {w: embedder.embed(w) for w in gold_words}

    # Get unique normalized responses that need embedding
    needs_embedding = df[~df["is_exact_match"] & ~df["is_non_response"]]["normalized"]
    unique_responses = needs_embedding.dropna().unique().tolist()

    # Embed responses in batches
    resp_embeddings: dict[str, np.ndarray] = {}
    for i in range(0, len(unique_responses), batch_size):
        batch = unique_responses[i : i + batch_size]
        embs = embedder.embed_batch(batch)
        for text, emb in zip(batch, embs):
            resp_embeddings[text] = emb

    # Compute similarity scores
    scores = []
    for _, row in df.iterrows():
        if row["is_exact_match"]:
            scores.append(1.0)
        elif row["is_non_response"]:
            scores.append(0.0)
        elif row["normalized"] in resp_embeddings:
            gold_emb = gold_embeddings[row["gold"]]
            resp_emb = resp_embeddings[row["normalized"]]
            sim = compute_cosine_similarity(gold_emb, resp_emb)
            scores.append(sim)
        else:
            scores.append(0.0)

    df["cosine_sim"] = scores
    df["binary_score"] = df["is_exact_match"].astype(int)

    return df


class GradedScorer:
    """Graded scorer using cosine similarity between embeddings.

    This class provides a stateful interface for computing graded similarity
    scores between responses and gold answers.
    """

    def __init__(self, embedder: Embedder):
        """Initialize the graded scorer.

        Args:
            embedder: Embedder instance for computing text embeddings.
        """
        self.embedder = embedder

    def score(self, response: str, gold: str) -> float:
        """Compute similarity score between a response and gold answer.

        Args:
            response: The user response text.
            gold: The gold/target answer text.

        Returns:
            Cosine similarity score clamped to [0, 1].
        """
        resp_emb = self.embedder.embed(response)
        gold_emb = self.embedder.embed(gold)
        return compute_cosine_similarity(resp_emb, gold_emb)

    def score_batch(
        self,
        responses: list[str],
        golds: list[str],
    ) -> list[float]:
        """Compute similarity scores for a batch of response-gold pairs.

        Args:
            responses: List of user response texts.
            golds: List of gold/target answer texts.

        Returns:
            List of cosine similarity scores.
        """
        if len(responses) != len(golds):
            raise ValueError("responses and golds must have the same length")

        resp_embs = self.embedder.embed_batch(responses)
        gold_embs = self.embedder.embed_batch(golds)

        scores = []
        for resp_emb, gold_emb in zip(resp_embs, gold_embs):
            sim = compute_cosine_similarity(resp_emb, gold_emb)
            scores.append(sim)

        return scores

    def score_dataframe(
        self,
        df: pd.DataFrame,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """Compute similarity scores for a DataFrame of responses.

        This is a convenience wrapper around compute_similarity_scores.

        Args:
            df: DataFrame with columns: gold, normalized, is_exact_match, is_non_response
            batch_size: Batch size for embedding computation.

        Returns:
            DataFrame with added columns: cosine_sim, binary_score
        """
        return compute_similarity_scores(df, self.embedder, batch_size)
