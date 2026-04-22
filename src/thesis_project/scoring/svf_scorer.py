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
    threshold: float = 0.45,
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
        threshold: Cosine similarity threshold for chain-based cluster
            detection. Passed through to detect_clusters.

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

        Cluster-based (chain method, Pakhomov 2016):
            cluster_count            — number of clusters (including singletons)
            switch_count             — cluster_count - 1
            mean_cluster_size        — sum(cluster_sizes) / cluster_count
            max_cluster_size         — size of largest cluster
            clusters                 — list[list[str]] of grouped responses
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
        "cluster_count": 0,
        "switch_count": 0,
        "mean_cluster_size": float("nan"),
        "max_cluster_size": 0,
        "clusters": [],
    }

    if n < 2:
        if n == 1:
            result["cluster_count"] = 1
            result["switch_count"] = 0
            result["mean_cluster_size"] = 1.0
            result["max_cluster_size"] = 1
            result["clusters"] = [list(responses)]
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

    # Cluster/switch analysis (chain method)
    cluster_result = detect_clusters(
        responses=responses,
        consecutive_similarities=consec_sims,
        threshold=threshold,
        method="chain",
    )
    result["cluster_count"] = cluster_result["cluster_count"]
    result["switch_count"] = cluster_result["switch_count"]
    result["mean_cluster_size"] = cluster_result["mean_cluster_size"]
    result["max_cluster_size"] = cluster_result["max_cluster_size"]
    result["clusters"] = cluster_result["clusters"]

    return result


def detect_clusters(
    responses: list[str],
    consecutive_similarities: list[float],
    threshold: float = 0.45,
    method: str = "chain",
) -> dict:
    """Identify semantic clusters in an SVF response sequence.

    Chain method (Pakhomov et al., 2016): walk the response sequence left to
    right; consecutive words whose cosine similarity meets or exceeds
    *threshold* are assigned to the same cluster. When similarity drops below
    threshold, a switch has occurred and a new cluster begins.

    Follows the Pakhomov (2016) cluster-size convention: singletons count as
    clusters of size 1 (unlike Troyer 1997, which subtracts 1 from each
    cluster size). This matches VFClust and avoids special-casing singletons.

    The Troyer "cluster" method (requiring every pairwise similarity within a
    candidate group to exceed threshold) is a documented future extension.

    Args:
        responses: Ordered list of animal names.
        consecutive_similarities: Pre-computed consecutive similarity values
            (length = len(responses) - 1); obtain from score_svf().
        threshold: Similarity threshold for cluster membership. Default 0.45 —
            to be calibrated empirically.
        method: Only "chain" is implemented; anything else raises ValueError.

    Returns:
        dict with keys:
            clusters          — list[list[str]], e.g. [["hund","katt"], ["cykel"]]
            cluster_sizes     — list[int], e.g. [2, 1]
            cluster_count     — int, number of clusters (including singletons)
            switch_count      — int, cluster_count - 1
            mean_cluster_size — float, sum(cluster_sizes) / cluster_count
            max_cluster_size  — int, largest cluster size
    """
    if method != "chain":
        raise ValueError(
            f"Unknown cluster method: {method!r}. Only 'chain' is implemented."
        )

    if len(responses) == 0:
        return {
            "clusters": [],
            "cluster_sizes": [],
            "cluster_count": 0,
            "switch_count": 0,
            "mean_cluster_size": float("nan"),
            "max_cluster_size": 0,
        }

    if len(responses) == 1:
        return {
            "clusters": [list(responses)],
            "cluster_sizes": [1],
            "cluster_count": 1,
            "switch_count": 0,
            "mean_cluster_size": 1.0,
            "max_cluster_size": 1,
        }

    clusters: list[list[str]] = []
    current_cluster: list[str] = [responses[0]]

    for i, sim in enumerate(consecutive_similarities):
        if sim >= threshold:
            current_cluster.append(responses[i + 1])
        else:
            clusters.append(current_cluster)
            current_cluster = [responses[i + 1]]

    clusters.append(current_cluster)

    cluster_sizes = [len(c) for c in clusters]
    cluster_count = len(clusters)

    return {
        "clusters": clusters,
        "cluster_sizes": cluster_sizes,
        "cluster_count": cluster_count,
        "switch_count": cluster_count - 1,
        "mean_cluster_size": sum(cluster_sizes) / cluster_count,
        "max_cluster_size": max(cluster_sizes),
    }
