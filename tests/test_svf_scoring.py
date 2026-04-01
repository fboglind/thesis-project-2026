"""test_svf_scoring.py

Unit tests for the SVF scorer. Uses MockEmbedder — no GPU required.
"""

import math

import pytest

from thesis_project.embeddings.encoder import MockEmbedder
from thesis_project.scoring.svf_scorer import detect_clusters, score_svf

ENCODER = MockEmbedder()

ANIMALS = ["hund", "katt", "kisse", "hund", "fisk", "katt"]
# total=6, unique={'hund','katt','kisse','fisk'}=4, repetitions=2


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


def test_detect_clusters_not_implemented():
    with pytest.raises(NotImplementedError):
        detect_clusters(["hund", "katt"], [0.5])
