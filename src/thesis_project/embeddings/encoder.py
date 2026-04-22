"""encoder.py

Text embedding classes for computing semantic representations using transformer models.
"""

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Abstract base class for text embedders."""

    def __init__(self, pooling: str = "cls"):
        self.pooling = pooling

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

    Supports two pooling strategies over the final hidden layer:
        - "cls": the [CLS] token vector (default, current baseline).
        - "mean": mean of content subword vectors, excluding [CLS] and [SEP].

    Caches embeddings for repeated lookups. Cache keys are prefixed with the
    pooling strategy so CLS and mean vectors for the same text don't collide.
    """

    def __init__(
        self,
        model_name: str = "KB/bert-base-swedish-cased",
        pooling: str = "cls",
    ):
        if pooling not in ("cls", "mean"):
            raise ValueError(
                f"Unknown pooling strategy: {pooling!r}. Expected 'cls' or 'mean'."
            )
        super().__init__(pooling=pooling)

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.cache: dict[str, np.ndarray] = {}
        self._torch = torch

    def _pool(self, last_hidden_state, attention_mask):
        """Apply the configured pooling strategy to transformer output.

        Args:
            last_hidden_state: tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask: tensor of shape [batch_size, seq_len]

        Returns:
            Pooled tensor of shape [batch_size, hidden_dim].
        """
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]

        # "mean": average content subwords, excluding [CLS] (pos 0) and [SEP] (last attended pos).
        content_mask = attention_mask.clone()
        content_mask[:, 0] = 0
        seq_lengths = attention_mask.sum(dim=1)
        for i in range(content_mask.size(0)):
            content_mask[i, seq_lengths[i] - 1] = 0

        content_mask_f = content_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * content_mask_f).sum(dim=1)
        counts = content_mask_f.sum(dim=1).clamp(min=1)
        return summed / counts

    def embed(self, text: str) -> np.ndarray:
        """Get embedding for a text string. Returns 1D numpy array."""
        cache_key = f"{self.pooling}:{text}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=64
        )
        with self._torch.no_grad():
            outputs = self.model(**inputs)

        pooled = self._pool(outputs.last_hidden_state, inputs["attention_mask"])
        emb = pooled.squeeze().numpy()
        self.cache[cache_key] = emb
        return emb

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (n, dim) numpy array."""
        uncached = [t for t in texts if f"{self.pooling}:{t}" not in self.cache]

        if uncached:
            inputs = self.tokenizer(
                uncached, return_tensors="pt", padding=True, truncation=True, max_length=64
            )
            with self._torch.no_grad():
                outputs = self.model(**inputs)

            pooled = self._pool(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = pooled.numpy()
            for text, emb in zip(uncached, embeddings):
                self.cache[f"{self.pooling}:{text}"] = emb

        return np.array([self.cache[f"{self.pooling}:{t}"] for t in texts])


class MockEmbedder(Embedder):
    """Mock embedder for testing pipeline logic without GPU/model.

    Assigns random but deterministic embeddings.
    Same word always gets the same embedding.

    Accepts ``pooling`` for interface compatibility with ``KBBertEmbedder`` but
    ignores it: mock vectors don't depend on pooling strategy.
    """

    def __init__(self, dim: int = 768, pooling: str = "cls"):
        super().__init__(pooling=pooling)
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


class SentenceTransformerEmbedder(Embedder):
    """Embedder using sentence-transformers models.

    Supports any model loadable by the sentence-transformers library,
    including newer-generation models that require load-time or encode-time
    keyword arguments (e.g. trust_remote_code, task selection for LoRA
    adapters).

    Args:
        model_name: HuggingFace model identifier.
        prefix: String to prepend to all inputs before encoding.
            Use "query: " for intfloat/multilingual-e5-* models
            (symmetric similarity tasks). Default: None (no prefix).
        model_kwargs: Optional dict of keyword arguments forwarded to the
            SentenceTransformer constructor. Use this for models that need
            e.g. {"trust_remote_code": True}. Default: None.
        encode_kwargs: Optional dict of keyword arguments forwarded to
            model.encode() on every call. Use this for models that need a
            per-call argument such as {"task": "text-matching"} for Jina v3.
            Default: None.

    Examples:
        # Swedish SBERT — no prefix, no kwargs
        SentenceTransformerEmbedder("KBLab/sentence-bert-swedish-cased")

        # e5 — requires prefix for symmetric similarity
        SentenceTransformerEmbedder(
            "intfloat/multilingual-e5-large",
            prefix="query: ",
        )

        # gte-multilingual — needs trust_remote_code at load time
        SentenceTransformerEmbedder(
            "Alibaba-NLP/gte-multilingual-base",
            model_kwargs={"trust_remote_code": True},
        )

        # Jina v3 — needs both load-time and encode-time kwargs
        SentenceTransformerEmbedder(
            "jinaai/jina-embeddings-v3",
            model_kwargs={"trust_remote_code": True},
            encode_kwargs={"task": "text-matching"},
        )
    """

    def __init__(
        self,
        model_name: str,
        prefix: str | None = None,
        model_kwargs: dict | None = None,
        encode_kwargs: dict | None = None,
    ):
        super().__init__(pooling="mean")

        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.prefix = prefix
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        self.model = SentenceTransformer(model_name, **self.model_kwargs)
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        """Get embedding for a text string. Returns 1D numpy array."""
        if text in self.cache:
            return self.cache[text]
        input_text = f"{self.prefix}{text}" if self.prefix else text
        emb = self.model.encode(input_text, **self.encode_kwargs)
        self.cache[text] = emb
        return emb

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (n, dim) numpy array."""
        uncached = [t for t in texts if t not in self.cache]
        if uncached:
            input_texts = [
                f"{self.prefix}{t}" if self.prefix else t for t in uncached
            ]
            embs = self.model.encode(input_texts, **self.encode_kwargs)
            for text, emb in zip(uncached, embs):
                self.cache[text] = emb
        return np.array([self.cache[t] for t in texts])
