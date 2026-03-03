"""encoder.py

Text embedding classes for computing semantic representations using transformer models.
"""

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Abstract base class for text embedders."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Get embedding for a single text string.

        Args:
            text: Input text to embed.

        Returns:
            1D numpy array representing the text embedding.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts.

        Args:
            texts: List of input texts to embed.

        Returns:
            2D numpy array of shape (n_texts, embedding_dim).
        """
        pass


class KBBertEmbedder(Embedder):
    """Compute embeddings using KB-BERT (Swedish BERT).

    Uses the [CLS] token representation from the last hidden layer.
    Caches embeddings for repeated lookups.
    """

    def __init__(self, model_name: str = "KB/bert-base-swedish-cased"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.cache: dict[str, np.ndarray] = {}
        self._torch = torch

    def embed(self, text: str) -> np.ndarray:
        """Get embedding for a text string. Returns 1D numpy array."""
        if text in self.cache:
            return self.cache[text]

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=64
        )
        with self._torch.no_grad():
            outputs = self.model(**inputs)

        # [CLS] token embedding
        emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        self.cache[text] = emb
        return emb

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (n, dim) numpy array."""
        # Split into cached and uncached
        uncached = [t for t in texts if t not in self.cache]

        if uncached:
            inputs = self.tokenizer(
                uncached, return_tensors="pt", padding=True, truncation=True, max_length=64
            )
            with self._torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            for text, emb in zip(uncached, embeddings):
                self.cache[text] = emb

        return np.array([self.cache[t] for t in texts])


class MockEmbedder(Embedder):
    """Mock embedder for testing pipeline logic without GPU/model.

    Assigns random but deterministic embeddings.
    Same word always gets the same embedding.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        """Get deterministic random embedding for text."""
        if text not in self.cache:
            rng = np.random.RandomState(hash(text) % 2**31)
            self.cache[text] = rng.randn(self.dim).astype(np.float32)
        return self.cache[text]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts."""
        return np.array([self.embed(t) for t in texts])
