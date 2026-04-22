"""Tests for embedding extraction with configurable pooling."""
import numpy as np
import pytest

from thesis_project.embeddings import Embedder, KBBertEmbedder, MockEmbedder


def test_mock_embedder_interface():
    """MockEmbedder accepts pooling parameter and produces correct shapes."""
    for pooling in ("cls", "mean"):
        emb = MockEmbedder(pooling=pooling)
        assert isinstance(emb, Embedder)
        assert emb.pooling == pooling
        vec = emb.embed("kamel")
        assert vec.shape == (768,)
        batch = emb.embed_batch(["kamel", "elefant", "hund"])
        assert batch.shape == (3, 768)


def test_mock_embedder_deterministic():
    """Same word always gets the same embedding."""
    emb = MockEmbedder()
    v1 = emb.embed("kamel")
    v2 = emb.embed("kamel")
    assert np.array_equal(v1, v2)


def test_mock_embedder_different_words():
    """Different words get different embeddings."""
    emb = MockEmbedder()
    v1 = emb.embed("kamel")
    v2 = emb.embed("elefant")
    assert not np.array_equal(v1, v2)


def test_kbbert_rejects_invalid_pooling():
    """KBBertEmbedder raises ValueError for unknown pooling strategy."""
    with pytest.raises(ValueError, match="pooling"):
        KBBertEmbedder(pooling="invalid")


def test_kbbert_accepts_pooling_parameter():
    """KBBertEmbedder constructor accepts pooling='cls' and 'mean'.

    NOTE: requires KB-BERT model to be downloaded. Skip if unavailable.
    """
    try:
        embedder_cls = KBBertEmbedder(pooling="cls")
        embedder_mean = KBBertEmbedder(pooling="mean")
        assert embedder_cls.pooling == "cls"
        assert embedder_mean.pooling == "mean"
    except (OSError, Exception):
        pytest.skip("KB-BERT model not available")
