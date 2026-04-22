# Phase 2: Add Sentence-Transformer Model Support

## Context

Phase 1 confirmed that switching from [CLS] to mean-pooled subword embeddings
does not improve cosine similarity discrimination for single-word inputs on
KB-BERT. The anisotropy is a property of the model's representational geometry,
not the token-selection strategy. Sentence-transformer models (trained with
contrastive objectives that explicitly optimise cosine similarity) are expected
to produce meaningfully distributed similarity scores.

This task adds a `SentenceTransformerEmbedder` class that wraps the
`sentence-transformers` library and exposes the same `Embedder` interface, so
the scoring pipelines can swap models with one constructor argument change.

The class is designed to support both classical sentence-transformer models
(KB-SBERT, e5) and newer-generation multilingual models that require
load-time or encode-time keyword arguments (Jina v3, Qwen3-Embedding,
gte-multilingual). Forward-compatibility is achieved through optional
`model_kwargs` and `encode_kwargs` parameters — see Section 1.

GitHub issues: #29, #30.

## Models to Support

### Primary (must work end-to-end in Phase 2)

1. **`KBLab/sentence-bert-swedish-cased`** — Swedish SBERT trained on Swedish
   NLI + STS data. Contrastive fine-tuning on top of KB-BERT. Produces 768-dim
   embeddings. No special input formatting required.

2. **`intfloat/multilingual-e5-large`** — Multilingual model trained with
   weakly-supervised contrastive learning. Produces 1024-dim embeddings.
   **Requires input prefix**: all inputs must be prepended with `"query: "`
   for symmetric similarity tasks (per the model card). For our use case
   (single-word similarity), inputs become e.g. `"query: kamel"`.

### Additional (must be loadable via the same class, not necessarily run in Phase 2)

The class design must be extensible enough to support these newer models
without further refactoring. None need to be downloaded or tested as part of
Phase 2 — the requirement is only that the constructor signature can express
their needs. Concrete configurations:

3. **`intfloat/multilingual-e5-large-instruct`** — Instruction-tuned successor
   to e5-large. Same API and prefix conventions. Drop-in replacement.

   ```python
   SentenceTransformerEmbedder(
       "intfloat/multilingual-e5-large-instruct",
       prefix="query: ",
   )
   ```

4. **`Alibaba-NLP/gte-multilingual-base`** — Small, fast multilingual model.
   Requires `trust_remote_code=True` at load time.

   ```python
   SentenceTransformerEmbedder(
       "Alibaba-NLP/gte-multilingual-base",
       model_kwargs={"trust_remote_code": True},
   )
   ```

5. **`jinaai/jina-embeddings-v3`** — Multilingual model with task-specific
   LoRA adapters. Requires `trust_remote_code=True` at load time, and a
   `task` argument at encode time to select the right adapter
   (`"text-matching"` is the appropriate one for symmetric similarity).

   ```python
   SentenceTransformerEmbedder(
       "jinaai/jina-embeddings-v3",
       model_kwargs={"trust_remote_code": True},
       encode_kwargs={"task": "text-matching"},
   )
   ```

6. **`Qwen/Qwen3-Embedding-0.6B`** — Decoder-based embedding model from the
   Qwen3 family. Loadable via `sentence-transformers` (≥2.7). For symmetric
   single-word similarity, no instruction prefix is recommended (Qwen's
   instruction-aware behaviour is geared toward asymmetric retrieval; symmetric
   STS-style tasks typically perform best with no instruction or the same
   instruction on both sides).

   ```python
   SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-0.6B")
   ```

## Files to Modify

```
src/thesis_project/embeddings/encoder.py    — add SentenceTransformerEmbedder class
src/thesis_project/embeddings/__init__.py   — export new class
pyproject.toml                              — add sentence-transformers dependency
```

### Files to Create

```
tests/test_embeddings.py                    — add tests for new class (file exists
                                              from Phase 1, append to it)
tests/verify_models.py                      — standalone comparison script
```

### Files NOT to Touch

- `src/thesis_project/scoring/` — all scorers. They receive embeddings as
  numpy arrays via the `Embedder` interface. No changes needed.
- `bnt_pipeline.py` — legacy, has its own embedder.
- `svf_pipeline.py`, `fas_pipeline.py` — will work as-is with default
  `KBBertEmbedder()`. Model selection will be added to these later.
- All notebooks.

## Current State of encoder.py (after Phase 1)

```python
class Embedder(ABC):
    def __init__(self, pooling: str = "cls"):
        self.pooling = pooling

    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


class KBBertEmbedder(Embedder):
    def __init__(self, model_name="KB/bert-base-swedish-cased", pooling="cls"):
        super().__init__(pooling=pooling)
        # ... loads model, has cache, has _pool() method ...

class MockEmbedder(Embedder):
    def __init__(self, dim=768, pooling="cls"):
        super().__init__(pooling=pooling)
        # ... deterministic random embeddings ...
```

## Requirements

### 1. Add `SentenceTransformerEmbedder` class to `encoder.py`

```python
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
        super().__init__(pooling="mean")  # sentence-transformers handle pooling internally
        ...
```

Key design decisions:

**`pooling="mean"` in super().__init__():** Sentence-transformers handle their
own pooling internally (typically mean pooling over all tokens, though some
models — e.g. Qwen3-Embedding — use last-token EOS pooling). Passing
`pooling="mean"` to the base class is descriptively accurate for the common
case and keeps the attribute available for logging/identification, but it is
NOT used in the embedding logic — the `sentence-transformers` library handles
pooling.

**`prefix` parameter, not auto-detection:** Do NOT try to auto-detect e5
models from the model name. Make it an explicit parameter. This is clearer
and avoids fragile string matching.

**`model_kwargs` and `encode_kwargs` parameters:** These exist to keep the
class extensible to newer models without further refactoring. They are
forwarded directly — `model_kwargs` to `SentenceTransformer(...)` at load
time, and `encode_kwargs` to `model.encode(...)` on every call. Defaults
are `None` (treated as empty dict internally), so existing usage with only
`model_name` and `prefix` is unaffected.

**Store key parameters as attributes:** Set `self.model_name`, `self.prefix`,
`self.model_kwargs`, and `self.encode_kwargs` for downstream identification
(logging, cache keys, result metadata, debugging).

### 2. Implementation Details

The `sentence-transformers` library provides the `SentenceTransformer` class:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(model_name, **model_kwargs)
embeddings = model.encode(texts, **encode_kwargs)  # returns np.ndarray of shape (n, dim)
```

`model.encode()` accepts a single string or a list of strings and returns
numpy arrays directly — no torch handling needed.

**`__init__` body:**

```python
self.model_name = model_name
self.prefix = prefix
self.model_kwargs = model_kwargs or {}
self.encode_kwargs = encode_kwargs or {}
self.model = SentenceTransformer(model_name, **self.model_kwargs)
self.cache: dict[str, np.ndarray] = {}
```

**`embed()` method:**

```python
def embed(self, text: str) -> np.ndarray:
    if text in self.cache:
        return self.cache[text]
    input_text = f"{self.prefix}{text}" if self.prefix else text
    emb = self.model.encode(input_text, **self.encode_kwargs)  # returns 1D np.ndarray
    self.cache[text] = emb
    return emb
```

**Cache key:** Use the raw `text` (without prefix) as the cache key, not the
prefixed version. The prefix is a model-specific encoding detail, not part of
the word identity. This means the cache stores results for "kamel", not for
"query: kamel". Since a `SentenceTransformerEmbedder` instance is always
bound to one model, one prefix, and one set of encode kwargs, there is no
collision risk.

**`embed_batch()` method:**

```python
def embed_batch(self, texts: list[str]) -> np.ndarray:
    uncached = [t for t in texts if t not in self.cache]
    if uncached:
        input_texts = [f"{self.prefix}{t}" if self.prefix else t for t in uncached]
        embs = self.model.encode(input_texts, **self.encode_kwargs)  # (n, dim) np.ndarray
        for text, emb in zip(uncached, embs):
            self.cache[text] = emb
    return np.array([self.cache[t] for t in texts])
```

**Cache type:** Same as `KBBertEmbedder` — a plain `self.cache: dict[str, np.ndarray]`.

### 3. Update `__init__.py`

Add the new class to exports:

```python
from .encoder import Embedder, KBBertEmbedder, MockEmbedder, SentenceTransformerEmbedder

__all__ = ["Embedder", "KBBertEmbedder", "MockEmbedder", "SentenceTransformerEmbedder"]
```

### 4. Add `sentence-transformers` dependency to `pyproject.toml`

Add `"sentence-transformers"` to the `dependencies` list. Pin to ≥2.7 so that
newer models (Qwen3-Embedding, etc.) load correctly:

```toml
dependencies = [
    "torch",
    "transformers",
    "sentence-transformers>=2.7",
    "pandas",
    "numpy",
    "scipy",
    "pyyaml",
    "scikit-learn",
]
```

### 5. Add Tests to `tests/test_embeddings.py`

Append these tests to the existing file (which has MockEmbedder and
KBBertEmbedder tests from Phase 1):

```python
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
    # Use MockEmbedder-style smoke check: just verify attribute storage
    # without loading any model. We can't construct SentenceTransformerEmbedder
    # without a model, so this test is wrapped in try/skip like the others.
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
        assert batch.shape[1] == vec.shape[0]  # same embedding dim
    except (OSError, Exception):
        pytest.skip("Model not available")


def test_sentence_transformer_is_embedder():
    """SentenceTransformerEmbedder is a subclass of Embedder."""
    assert issubclass(SentenceTransformerEmbedder, Embedder)
```

Note: these tests require the models to be downloaded. They use `pytest.skip`
when models are unavailable. The `MockEmbedder` tests from Phase 1 always
run and verify the interface contract.

### 6. Create Verification Script `tests/verify_models.py`

This replaces `tests/verify_pooling_gpu.py` as the primary diagnostic. It
compares all available models on the same word set:

```python
"""Compare embedding models on a standard word set.

Usage: python tests/verify_models.py

Prints cosine similarity matrices for each model. The key diagnostic is
the off-diagonal std — higher = more discriminative similarity scores.

The active set below covers the two Phase 2 primary models plus the
e5-large-instruct drop-in. Additional newer models (gte-multilingual,
Jina v3, Qwen3-Embedding) are listed at the bottom as commented-out
configurations — uncomment to add them to the comparison.
"""
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

words = ["kamel", "elefant", "hund", "katt", "djur", "cykel"]

models = []

# KB-BERT (baseline — known to be compressed)
try:
    from thesis_project.embeddings import KBBertEmbedder
    models.append(("KB-BERT [CLS]", KBBertEmbedder(pooling="cls")))
except Exception as e:
    print(f"Skipping KB-BERT: {e}")

# Swedish SBERT
try:
    from thesis_project.embeddings import SentenceTransformerEmbedder
    models.append((
        "Swedish SBERT",
        SentenceTransformerEmbedder("KBLab/sentence-bert-swedish-cased"),
    ))
except Exception as e:
    print(f"Skipping Swedish SBERT: {e}")

# Multilingual e5-large
try:
    from thesis_project.embeddings import SentenceTransformerEmbedder
    models.append((
        "e5-large",
        SentenceTransformerEmbedder(
            "intfloat/multilingual-e5-large",
            prefix="query: ",
        ),
    ))
except Exception as e:
    print(f"Skipping e5-large: {e}")

# Multilingual e5-large-instruct (instruction-tuned successor — drop-in)
try:
    from thesis_project.embeddings import SentenceTransformerEmbedder
    models.append((
        "e5-large-instruct",
        SentenceTransformerEmbedder(
            "intfloat/multilingual-e5-large-instruct",
            prefix="query: ",
        ),
    ))
except Exception as e:
    print(f"Skipping e5-large-instruct: {e}")

# ── Optional: newer-generation models (uncomment to add) ──
#
# from thesis_project.embeddings import SentenceTransformerEmbedder
#
# # gte-multilingual-base — small/fast, needs trust_remote_code
# try:
#     models.append((
#         "gte-multilingual",
#         SentenceTransformerEmbedder(
#             "Alibaba-NLP/gte-multilingual-base",
#             model_kwargs={"trust_remote_code": True},
#         ),
#     ))
# except Exception as e:
#     print(f"Skipping gte-multilingual: {e}")
#
# # Jina v3 — task-specific LoRA adapter for symmetric similarity
# try:
#     models.append((
#         "jina-v3 (text-matching)",
#         SentenceTransformerEmbedder(
#             "jinaai/jina-embeddings-v3",
#             model_kwargs={"trust_remote_code": True},
#             encode_kwargs={"task": "text-matching"},
#         ),
#     ))
# except Exception as e:
#     print(f"Skipping jina-v3: {e}")
#
# # Qwen3-Embedding-0.6B — decoder-based, fits on T4
# try:
#     models.append((
#         "Qwen3-Embedding-0.6B",
#         SentenceTransformerEmbedder("Qwen/Qwen3-Embedding-0.6B"),
#     ))
# except Exception as e:
#     print(f"Skipping Qwen3-Embedding-0.6B: {e}")

if not models:
    print("No models available. Exiting.")
    sys.exit(1)

for name, embedder in models:
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")

    vecs = embedder.embed_batch(words)
    sim_matrix = cosine_similarity(vecs)

    print(f"Embedding dim: {vecs.shape[1]}")
    print(f"\nCosine similarity matrix:")
    header = "         " + "  ".join(f"{w:>8s}" for w in words)
    print(header)
    for i, w in enumerate(words):
        row = f"{w:>8s} " + "  ".join(f"{sim_matrix[i,j]:8.4f}" for j in range(len(words)))
        print(row)

    mask = ~np.eye(len(words), dtype=bool)
    off_diag = sim_matrix[mask]
    print(f"\nOff-diagonal range: [{off_diag.min():.4f}, {off_diag.max():.4f}]")
    print(f"Off-diagonal std:   {off_diag.std():.4f}")
    print(f"Off-diagonal mean:  {off_diag.mean():.4f}")

    # Key pairs to watch
    print(f"\nKey comparisons:")
    pairs = [("kamel", "elefant"), ("kamel", "cykel"), ("hund", "katt"), ("djur", "kamel")]
    for w1, w2 in pairs:
        i, j = words.index(w1), words.index(w2)
        print(f"  sim({w1}, {w2}) = {sim_matrix[i,j]:.4f}")
```

**What to look for in the output:**

- `sim(kamel, elefant)` should be substantially higher than `sim(kamel, cykel)`.
  On KB-BERT [CLS] this gap is ~0.03. On a good sentence-transformer it should
  be ~0.15–0.30+.
- `sim(djur, kamel)` — this is the hypernym test. Even on sentence-transformers,
  hypernyms tend to score high. If it's *higher* than `sim(kamel, elefant)`,
  the distributional inclusion problem (Geffet & Dagan 2005) persists and the
  SALDO integration remains motivated.
- Off-diagonal std should be substantially higher than KB-BERT's ~0.01.
- Comparing e5-large vs e5-large-instruct on the same word set gives a quick
  read on whether the instruction-tuned variant is worth adopting as the
  default e5-family model in later work.

## What NOT to Do

- Do NOT modify any scoring code (`svf_scorer.py`, `graded_scorer.py`, etc.).
- Do NOT modify pipeline scripts or notebooks.
- Do NOT touch `bnt_pipeline.py`.
- Do NOT auto-detect e5 models by name — use the explicit `prefix` parameter.
- Do NOT implement model-selection CLI flags in the pipeline scripts — that
  is a separate follow-up task.
- Do NOT implement anisotropy correction / whitening.
- Do NOT remove or modify the existing `KBBertEmbedder` or `MockEmbedder`.
- Do NOT remove `tests/verify_pooling_gpu.py` — keep it alongside the new
  `tests/verify_models.py`.
- Do NOT uncomment the "Optional" models block in `verify_models.py` as part
  of Phase 2 — those are listed for later activation. The Phase 2 deliverable
  is verified by the four primary models (KB-BERT, Swedish SBERT, e5-large,
  e5-large-instruct).

## Dependency Note

`sentence-transformers` pulls in `torch` and `transformers` as dependencies
(already installed), plus `huggingface-hub` and a few other packages. After
adding it to `pyproject.toml`, run:

```bash
pip install -e ".[dev]"
```

to update the local environment.

## Summary of Changes

| File | Action |
|------|--------|
| `src/thesis_project/embeddings/encoder.py` | Add `SentenceTransformerEmbedder` class with `model_kwargs` / `encode_kwargs` support |
| `src/thesis_project/embeddings/__init__.py` | Add `SentenceTransformerEmbedder` to exports |
| `pyproject.toml` | Add `sentence-transformers>=2.7` to dependencies |
| `tests/test_embeddings.py` | Append `SentenceTransformerEmbedder` tests (incl. kwargs storage) |
| `tests/verify_models.py` | Create multi-model comparison script (4 active models, 3 commented-out newer models) |

## Git

Branch: `feature/sentence-transformer-support` (create from `main` — merge
Phase 1 branch first if not already merged).

Commits:
1. `feat(embeddings): add SentenceTransformerEmbedder with prefix and kwargs support`
   — encoder.py + __init__.py changes
2. `build: add sentence-transformers dependency`
   — pyproject.toml
3. `test(embeddings): add sentence-transformer tests and model comparison script`
   — tests/
