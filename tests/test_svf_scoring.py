"""test_svf_scoring.py

Unit tests for the SVF scorer. Uses MockEmbedder — no GPU required.
"""

import math

import numpy as np
import pytest

from thesis_project.embeddings.encoder import MockEmbedder
from thesis_project.scoring.svf_scorer import detect_clusters, score_svf

ENCODER = MockEmbedder()

ANIMALS = ["hund", "katt", "kisse", "hund", "fisk", "katt"]
# total=6, unique={'hund','katt','kisse','fisk'}=4, repetitions=2


# ──────────────────────────────────────────────────────
# score_svf — count + similarity metrics
# ──────────────────────────────────────────────────────

def test_count_metrics():
    result = score_svf(ANIMALS, ENCODER)
    assert result["total_words"] == 6
    assert result["unique_words"] == 4
    assert result["repetitions"] == 2


def test_consecutive_similarities_length():
    result = score_svf(ANIMALS, ENCODER)
    # One similarity per adjacent pair
    assert len(result["consecutive_similarities"]) == len(ANIMALS) - 1


def test_consecutive_similarities_range():
    result = score_svf(ANIMALS, ENCODER)
    for sim in result["consecutive_similarities"]:
        assert 0.0 <= sim <= 1.0, f"Similarity {sim} out of [0, 1]"


def test_mean_consecutive_similarity_finite():
    result = score_svf(ANIMALS, ENCODER)
    assert math.isfinite(result["mean_consecutive_similarity"])


def test_pairwise_similarity_mean_finite():
    result = score_svf(ANIMALS, ENCODER)
    assert math.isfinite(result["pairwise_similarity_mean"])


def test_temporal_gradient_finite():
    result = score_svf(ANIMALS, ENCODER)
    assert math.isfinite(result["temporal_gradient"])


def test_single_response_no_similarities():
    result = score_svf(["hund"], ENCODER)
    assert result["total_words"] == 1
    assert result["unique_words"] == 1
    assert result["repetitions"] == 0
    assert result["consecutive_similarities"] == []
    assert math.isnan(result["mean_consecutive_similarity"])
    assert math.isnan(result["pairwise_similarity_mean"])
    assert math.isnan(result["temporal_gradient"])


def test_empty_responses():
    result = score_svf([], ENCODER)
    assert result["total_words"] == 0
    assert result["consecutive_similarities"] == []


def test_all_same_word():
    result = score_svf(["hund", "hund", "hund"], ENCODER)
    assert result["total_words"] == 3
    assert result["unique_words"] == 1
    assert result["repetitions"] == 2
    # Same-word pairs: cosine similarity of identical vectors = 1.0
    for sim in result["consecutive_similarities"]:
        assert abs(sim - 1.0) < 1e-5


# ──────────────────────────────────────────────────────
# detect_clusters — chain method
# ──────────────────────────────────────────────────────

def test_chain_basic():
    """Basic chain clustering with clear clusters."""
    responses = ["hund", "katt", "ko", "elefant", "kamel", "cykel"]
    consec_sims = [0.64, 0.52, 0.38, 0.60, 0.34]
    result = detect_clusters(responses, consec_sims, threshold=0.50)

    assert result["cluster_count"] == 3
    assert result["switch_count"] == 2
    assert result["cluster_sizes"] == [3, 2, 1]
    assert result["mean_cluster_size"] == 2.0
    assert result["max_cluster_size"] == 3
    assert result["clusters"] == [
        ["hund", "katt", "ko"],
        ["elefant", "kamel"],
        ["cykel"],
    ]


def test_chain_all_above_threshold():
    """All consecutive similarities above threshold → one big cluster."""
    responses = ["hund", "katt", "ko"]
    consec_sims = [0.70, 0.65]
    result = detect_clusters(responses, consec_sims, threshold=0.50)

    assert result["cluster_count"] == 1
    assert result["switch_count"] == 0
    assert result["mean_cluster_size"] == 3.0


def test_chain_all_below_threshold():
    """All below threshold → all singletons."""
    responses = ["hund", "cykel", "bord"]
    consec_sims = [0.20, 0.15]
    result = detect_clusters(responses, consec_sims, threshold=0.50)

    assert result["cluster_count"] == 3
    assert result["switch_count"] == 2
    assert result["mean_cluster_size"] == 1.0


def test_chain_single_word():
    """Single word → one singleton cluster."""
    result = detect_clusters(["hund"], [], threshold=0.50)
    assert result["cluster_count"] == 1
    assert result["switch_count"] == 0
    assert result["mean_cluster_size"] == 1.0


def test_chain_empty():
    """Empty input."""
    result = detect_clusters([], [], threshold=0.50)
    assert result["cluster_count"] == 0
    assert result["switch_count"] == 0


def test_chain_threshold_boundary():
    """Similarity exactly at threshold should be included in cluster."""
    responses = ["a", "b", "c"]
    consec_sims = [0.50, 0.49]
    result = detect_clusters(responses, consec_sims, threshold=0.50)

    assert result["cluster_count"] == 2
    assert result["cluster_sizes"] == [2, 1]


def test_invalid_method():
    """Non-chain method raises ValueError."""
    with pytest.raises(ValueError):
        detect_clusters(["a", "b"], [0.5], threshold=0.5, method="cluster")


# ──────────────────────────────────────────────────────
# score_svf — cluster metrics integration + similarity_slope
# ──────────────────────────────────────────────────────

def test_score_svf_includes_cluster_metrics():
    """score_svf output includes cluster/switch metrics."""
    result = score_svf(["hund", "katt", "ko", "elefant"], ENCODER)

    assert "cluster_count" in result
    assert "switch_count" in result
    assert "mean_cluster_size" in result
    assert "max_cluster_size" in result
    assert "similarity_slope" in result
    assert isinstance(result["cluster_count"], int)
    assert isinstance(result["similarity_slope"], float)


def test_score_svf_short_sequence():
    """Single-word sequence returns nan/zero for cluster metrics."""
    result = score_svf(["hund"], ENCODER)

    assert result["cluster_count"] == 1
    assert result["mean_cluster_size"] == 1.0
    assert np.isnan(result["similarity_slope"])


def test_score_svf_empty_sequence():
    """Empty sequence → zero clusters and nan mean."""
    result = score_svf([], ENCODER)

    assert result["cluster_count"] == 0
    assert result["switch_count"] == 0
    assert np.isnan(result["mean_cluster_size"])
    assert np.isnan(result["similarity_slope"])


def test_score_svf_temporal_gradient_still_present():
    """temporal_gradient kept for backwards compatibility."""
    result = score_svf(ANIMALS, ENCODER)
    assert "temporal_gradient" in result
