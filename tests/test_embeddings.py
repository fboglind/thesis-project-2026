"""Tests for embedding extraction with configurable pooling."""
import os

import numpy as np
import pytest

from thesis_project.embeddings import (
    Embedder,
    KBBertEmbedder,
    MockEmbedder,
    SentenceTransformerEmbedder,
)


_HEAVY = pytest.mark.skipif(
    os.environ.get("RUN_HEAVY_TESTS") != "1",
    reason="Heavy test, set RUN_HEAVY_TESTS=1 to run.",
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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


# ── SentenceTransformerEmbedder tests ──

def test_sentence_transformer_rejects_missing_model_name():
    """SentenceTransformerEmbedder requires model_name argument."""
    with pytest.raises(TypeError):
        SentenceTransformerEmbedder()


def test_sentence_transformer_stores_attributes():
    """Constructor stores model_name, prefix, kwargs, and pooling."""
    try:
        emb = SentenceTransformerEmbedder(
            "KBLab/sentence-bert-swedish-cased",
            prefix=None,
        )
        assert emb.model_name == "KBLab/sentence-bert-swedish-cased"
        assert emb.prefix is None
        assert emb.pooling == "mean"
        assert emb.model_kwargs == {}
        assert emb.encode_kwargs == {}
    except (OSError, Exception):
        pytest.skip("Model not available")


def test_sentence_transformer_with_prefix():
    """Prefix is stored and applied (verified via attribute, not output)."""
    try:
        emb = SentenceTransformerEmbedder(
            "intfloat/multilingual-e5-large",
            prefix="query: ",
        )
        assert emb.prefix == "query: "
    except (OSError, Exception):
        pytest.skip("Model not available")


def test_sentence_transformer_stores_kwargs():
    """model_kwargs and encode_kwargs are stored as dicts (None -> {})."""
    try:
        emb = SentenceTransformerEmbedder(
            "KBLab/sentence-bert-swedish-cased",
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"task": "text-matching"},
        )
        assert emb.model_kwargs == {"trust_remote_code": True}
        assert emb.encode_kwargs == {"task": "text-matching"}
    except (OSError, Exception):
        pytest.skip("Model not available")


def test_sentence_transformer_embed_shapes():
    """embed() returns 1D array, embed_batch() returns 2D array."""
    try:
        emb = SentenceTransformerEmbedder("KBLab/sentence-bert-swedish-cased")
        vec = emb.embed("kamel")
        assert vec.ndim == 1

        batch = emb.embed_batch(["kamel", "elefant", "hund"])
        assert batch.ndim == 2
        assert batch.shape[0] == 3
        assert batch.shape[1] == vec.shape[0]
    except (OSError, Exception):
        pytest.skip("Model not available")


def test_sentence_transformer_is_embedder():
    """SentenceTransformerEmbedder is a subclass of Embedder."""
    assert issubclass(SentenceTransformerEmbedder, Embedder)


# ── Heavy smoke tests for the Phase 4a comparison models ──
# Gated by RUN_HEAVY_TESTS=1 so CI doesn't pull multi-GB weights.

_QWEN3_INSTRUCT_PREFIX = (
    "Instruct: Retrieve semantically similar Swedish words\n"
    "Query: "
)


@_HEAVY
def test_qwen3_embedding_smoke():
    """Qwen3-Embedding-0.6B loads and produces sensible Swedish-word similarities."""
    emb = SentenceTransformerEmbedder(
        "Qwen/Qwen3-Embedding-0.6B",
        prefix=_QWEN3_INSTRUCT_PREFIX,
        model_kwargs={"trust_remote_code": True},
    )
    v_kamel = emb.embed("kamel")
    v_hast = emb.embed("häst")
    v_cykel = emb.embed("cykel")
    assert v_kamel.ndim == 1
    sim_animals = _cosine(v_kamel, v_hast)
    sim_unrelated = _cosine(v_kamel, v_cykel)
    assert sim_animals > sim_unrelated, (
        f"Expected camel-horse > camel-bicycle similarity, "
        f"got animals={sim_animals:.3f}, unrelated={sim_unrelated:.3f}"
    )


@_HEAVY
def test_embeddinggemma_smoke():
    """EmbeddingGemma-300M loads and produces sensible Swedish-word similarities.

    NOTE: HuggingFace ID is a placeholder in YAML — verify
    google/embeddinggemma-300m on the HF hub before first run.
    """
    emb = SentenceTransformerEmbedder(
        "google/embeddinggemma-300m",
        prefix=_QWEN3_INSTRUCT_PREFIX,
    )
    v_kamel = emb.embed("kamel")
    v_hast = emb.embed("häst")
    v_cykel = emb.embed("cykel")
    sim_animals = _cosine(v_kamel, v_hast)
    sim_unrelated = _cosine(v_kamel, v_cykel)
    assert sim_animals > sim_unrelated


@_HEAVY
def test_harrier_smoke():
    """Harrier-OSS-v1 0.6B loads and produces sensible Swedish-word similarities.

    NOTE: HuggingFace ID is a placeholder in YAML — verify the exact
    Microsoft Harrier-OSS-v1 0.6B identifier on the HF hub before first
    run. If the ID fails, do NOT substitute a different size; report and
    stop (see PHASE_4A_INSTRUCTIONS.md §2c).
    """
    emb = SentenceTransformerEmbedder(
        "microsoft/Harrier-OSS-v1-0.6B",
        prefix=_QWEN3_INSTRUCT_PREFIX,
        model_kwargs={"trust_remote_code": True},
    )
    v_kamel = emb.embed("kamel")
    v_hast = emb.embed("häst")
    v_cykel = emb.embed("cykel")
    sim_animals = _cosine(v_kamel, v_hast)
    sim_unrelated = _cosine(v_kamel, v_cykel)
    assert sim_animals > sim_unrelated
