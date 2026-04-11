"""svf_scorer.py

Scoring for Semantic Verbal Fluency (SVF) test responses.
Computes count-based and embedding-based metrics following
Pakhomov et al. (2016) and Troyer et al. (1997).
"""

import itertools

import numpy as np

from .graded_scorer import compute_cosine_similarity


# ──────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────

def score_svf(
    responses: list[str],
    encoder,
    semantic_resource=None,
) -> dict:
    """Compute SVF metrics for a single participant's response sequence.

    Args:
        responses: Ordered list of animal names produced (Nones already
            stripped by the loader). Repetitions are expected and meaningful.
        encoder: An Embedder instance (KBBertEmbedder or MockEmbedder) used
            to compute cosine similarity between consecutive and all-pairs words.
        semantic_resource: Placeholder for future Saldo lexical resource
            integration (https://spraakbanken.gu.se/resurser/saldo).
            Not used — pass None.

    Returns:
        dict with keys:

        Count-based:
            total_words              — len(responses), including repetitions
            unique_words             — number of distinct words produced
            repetitions              — total_words - unique_words

        Embedding-based:
            consecutive_similarities — list[float], cosine sim for each
                                       adjacent pair (len = total_words - 1)
            mean_consecutive_similarity — mean of the above
            pairwise_similarity_mean — mean cosine sim across all unique-word
                                       pairs (Pakhomov semantic diversity)
            temporal_gradient        — mean(first-half consecutive sims) minus
                                       mean(second-half consecutive sims);
                                       positive = more clustering early
    """
    n = len(responses)
    unique = list(set(responses))

    result: dict = {
        "total_words": n,
        "unique_words": len(unique),
        "repetitions": n - len(unique),
        "consecutive_similarities": [],
        "mean_consecutive_similarity": float("nan"),
        "pairwise_similarity_mean": float("nan"),
        "temporal_gradient": float("nan"),
    }

    if n < 2:
        return result

    # Embed all unique words in one batch for efficiency
    unique_embs: dict[str, np.ndarray] = {}
    emb_matrix = encoder.embed_batch(unique)
    for word, emb in zip(unique, emb_matrix):
        unique_embs[word] = emb

    # Consecutive similarities
    consec_sims = [
        compute_cosine_similarity(unique_embs[responses[i]], unique_embs[responses[i + 1]])
        for i in range(n - 1)
    ]
    result["consecutive_similarities"] = consec_sims
    result["mean_consecutive_similarity"] = float(np.mean(consec_sims))

    # Pairwise similarity across all unique words (semantic diversity)
    if len(unique) >= 2:
        pair_sims = [
            compute_cosine_similarity(unique_embs[w1], unique_embs[w2])
            for w1, w2 in itertools.combinations(unique, 2)
        ]
        result["pairwise_similarity_mean"] = float(np.mean(pair_sims))

    # Temporal gradient: first-half vs second-half mean consecutive similarity
    mid = len(consec_sims) // 2
    if mid > 0:
        first_half = float(np.mean(consec_sims[:mid]))
        second_half = float(np.mean(consec_sims[mid:]))
        result["temporal_gradient"] = first_half - second_half

    return result


def detect_clusters(
    responses: list[str],
    consecutive_similarities: list[float],
    threshold: float = 0.45,
    method: str = "chain",
) -> dict:
    """Identify semantic clusters in an SVF response sequence.

    Chain method (Pakhomov et al., 2016): consecutive pairs with similarity
    exceeding *threshold* are assigned to the same cluster.

    Cluster method (Troyer et al., 1997): all pairwise combinations within
    a candidate group must exceed *threshold* to form a cluster.

    Args:
        responses: Ordered list of animal names.
        consecutive_similarities: Pre-computed consecutive similarity values
            (length = len(responses) - 1); obtain from score_svf().
        threshold: Similarity threshold for cluster membership.
            Default 0.45 — to be calibrated empirically.
        method: "chain" or "cluster".

    Returns:
        dict with keys:
            clusters, cluster_count, mean_cluster_size,
            max_cluster_size, switch_count
    """
    raise NotImplementedError("Cluster detection — future extension.")
