# Phase 1: Add Configurable Pooling Strategy to KBBertEmbedder

## Context

This project scores neuropsychological language test responses (BNT, SVF) using
KB-BERT embeddings and cosine similarity. The current embedder extracts the
`[CLS]` token vector, which is known to produce anisotropic embeddings — cosine
similarities compress into a narrow range (~0.84–0.86), eliminating diagnostic
group discrimination on SVF metrics. This task adds a `pooling` parameter so we
can compare `[CLS]` (current baseline) against mean-pooled subword embeddings
(expected to expand the similarity range).

GitHub issues: #25, #28.

## File to Modify

```
src/thesis_project/embeddings/encoder.py
```

This is the single file that needs changes. It contains three classes:
- `Embedder` — abstract base class with `embed()` and `embed_batch()` signatures
- `KBBertEmbedder` — the class to modify
- `MockEmbedder` — must accept the new parameter for interface compatibility

### Files to create

```
tests/test_embeddings.py    — currently empty, write tests here
tests/verify_pooling_gpu.py — standalone manual verification script
```

### Files NOT to touch

- `src/thesis_project/scoring/svf_scorer.py` — calls `encoder.embed_batch()`,
  receives numpy arrays, computes cosine similarity. No changes needed; the
  pooling change is upstream of this.
- `src/thesis_project/scoring/graded_scorer.py` — contains
  `compute_cosine_similarity()` used by SVF scorer. Unchanged.
- `src/thesis_project/embeddings/__init__.py` — already exports the right
  classes (`Embedder`, `KBBertEmbedder`, `MockEmbedder`). No changes needed
  unless you add a new class (you shouldn't).
- `bnt_pipeline.py` — has its OWN duplicate `KBBertEmbedder` (lines 152–206)
  that does NOT import from `src/`. Leave it alone; it's legacy code that will
  be refactored separately.
- `svf_pipeline.py`, `fas_pipeline.py` — import from `src/`. They instantiate
  `KBBertEmbedder()` with no arguments, which will continue to work because
  the default is `pooling="cls"`. No changes needed now.
- All notebooks (`notebooks/*.ipynb`) — run on Kaggle. Will be updated
  separately to pass `pooling="mean"`.

## Current Implementation (encoder.py)

The current `KBBertEmbedder` extracts embeddings like this:

```python
# In embed():
outputs = self.model(**inputs)
emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # [CLS] token

# In embed_batch():
outputs = self.model(**inputs)
embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # [CLS] for all
```

Both methods use `outputs.last_hidden_state[:, 0, :]` — this is the [CLS]
token at position 0. The model is `KB/bert-base-swedish-cased` with hidden
dimension 768.

Caching is a simple `self.cache: dict[str, np.ndarray]` keyed by raw text
string.

## Requirements

### 1. Add `pooling` parameter to `Embedder` base class

Update the abstract base class signature to accept and store pooling:

```python
class Embedder(ABC):
    def __init__(self, pooling: str = "cls"):
        self.pooling = pooling
```

This makes `pooling` part of the interface contract. Subclasses should call
`super().__init__(pooling=pooling)`.

### 2. Add `pooling` parameter to `KBBertEmbedder`

```python
class KBBertEmbedder(Embedder):
    def __init__(self, model_name: str = "KB/bert-base-swedish-cased",
                 pooling: str = "cls"):
        super().__init__(pooling=pooling)
        ...
```

Accepted values: `"cls"` and `"mean"`. Raise `ValueError` for anything else
at construction time (not at embed time).

**`pooling="cls"`**: Preserve current behaviour exactly. Extract
`outputs.last_hidden_state[:, 0, :]`.

**`pooling="mean"`**: Mean-pool all content subword vectors from the final
hidden layer, **excluding** the `[CLS]` token (position 0) and `[SEP]` token.

Implementation for mean pooling:

1. Get `last_hidden_state` — shape `[batch_size, seq_len, 768]`
2. Get `attention_mask` from the tokenizer output — shape `[batch_size, seq_len]`
3. Create a content mask that excludes special tokens:
   - Zero out position 0 (`[CLS]`) in every sequence
   - Zero out the last attended position (`[SEP]`) in every sequence
   - Keep all other attended positions (the actual content subwords)
4. Compute weighted mean:
   ```python
   content_mask = attention_mask.clone()
   content_mask[:, 0] = 0  # mask [CLS]
   # mask [SEP]: find last non-padding position per sequence
   seq_lengths = attention_mask.sum(dim=1)
   for i in range(content_mask.size(0)):
       content_mask[i, seq_lengths[i] - 1] = 0

   masked_hidden = last_hidden_state * content_mask.unsqueeze(-1)
   summed = masked_hidden.sum(dim=1)
   counts = content_mask.sum(dim=1, keepdim=True).clamp(min=1)
   pooled = summed / counts
   ```

**Edge case:** For a single-subword word like "hund", after tokenization you
get `[CLS] hund [SEP]` → 3 tokens. After masking [CLS] and [SEP], there is
exactly 1 content token. The mean is just that token's vector. This is correct.

**Important:** Apply the same pooling logic to both `embed()` and
`embed_batch()`. Factor out a private method to avoid duplicating:

```python
def _pool(self, last_hidden_state, attention_mask):
    """Apply pooling strategy to transformer output.

    Args:
        last_hidden_state: shape [batch_size, seq_len, hidden_dim]
        attention_mask: shape [batch_size, seq_len]

    Returns:
        Pooled embeddings, shape [batch_size, hidden_dim]
    """
    if self.pooling == "cls":
        return last_hidden_state[:, 0, :]
    elif self.pooling == "mean":
        # ... content mask logic as above ...
```

Then in `embed()`: call `_pool()` and `.squeeze().numpy()`.
In `embed_batch()`: call `_pool()` and `.numpy()`.

This also means the tokenizer call in `embed()` must now request
`return_tensors="pt"` for `attention_mask` (it already does — the current
code passes `return_tensors="pt"` which returns both `input_ids` and
`attention_mask` as tensors). The `inputs` dict already contains the
`attention_mask` — you just need to pass it to `_pool()` alongside the
model output.

### 3. Make cache keys pooling-aware

The current cache uses raw text as key: `self.cache[text] = emb`. Since [CLS]
and mean-pooled embeddings for the same text are different vectors, the cache
must distinguish them. Prefix the key with the pooling strategy:

```python
cache_key = f"{self.pooling}:{text}"
```

Update all cache reads and writes in both `embed()` and `embed_batch()` to
use this format.

### 4. Update `MockEmbedder`

Accept and store `pooling` for interface compatibility, but ignore it in the
actual embedding logic (random vectors regardless):

```python
class MockEmbedder(Embedder):
    def __init__(self, dim: int = 768, pooling: str = "cls"):
        super().__init__(pooling=pooling)
        self.dim = dim
        self.cache: dict[str, np.ndarray] = {}
```

The `MockEmbedder` does NOT need pooling-prefixed cache keys because its
vectors don't depend on pooling strategy. But keeping `self.pooling` on the
instance maintains the uniform interface.

### 5. Write tests in `tests/test_embeddings.py`

The test file is currently empty. Write tests using `MockEmbedder` only (no
GPU needed for CI). Structure:

```python
"""Tests for embedding extraction with configurable pooling."""
import numpy as np
import pytest
from thesis_project.embeddings import KBBertEmbedder, MockEmbedder, Embedder


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
```

### 6. Create manual GPU verification script

Create `tests/verify_pooling_gpu.py` — a standalone script (not pytest) for
manual verification on Kaggle with GPU:

```python
"""Manual verification: compare CLS vs mean pooling on KB-BERT.

Run on a machine with the KB-BERT model downloaded:
    python tests/verify_pooling_gpu.py

Expected: mean pooling shows wider off-diagonal similarity range than CLS.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from thesis_project.embeddings import KBBertEmbedder

words = ["kamel", "elefant", "hund", "katt", "djur", "cykel"]

for pooling in ("cls", "mean"):
    print(f"\n{'='*60}")
    print(f"Pooling: {pooling}")
    print(f"{'='*60}")
    emb = KBBertEmbedder(pooling=pooling)
    vecs = emb.embed_batch(words)
    sim_matrix = cosine_similarity(vecs)

    print(f"Shape: {vecs.shape}")
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
```

The key diagnostic: if CLS off-diagonal std is ~0.01–0.02 and mean pooling
is ~0.05–0.10, the anisotropy hypothesis is confirmed and the fix works.

## What NOT to Do

- Do NOT add new dependencies.
- Do NOT refactor scoring, preprocessing, or data loading.
- Do NOT add sentence-transformer models — that is Phase 2 (issues #29, #30).
- Do NOT implement anisotropy correction (whitening, all-but-the-top).
- Do NOT touch `bnt_pipeline.py` — it has its own duplicate embedder.
- Do NOT change default behaviour. `KBBertEmbedder()` with no arguments must
  produce identical output to the current implementation.
- Do NOT modify notebooks or pipeline scripts.

## Summary of Changes

| File | Action |
|------|--------|
| `src/thesis_project/embeddings/encoder.py` | Add `pooling` to `Embedder.__init__`, `KBBertEmbedder.__init__`, `MockEmbedder.__init__`. Add `_pool()` method. Update cache keys. |
| `tests/test_embeddings.py` | Write pytest tests (MockEmbedder-based + KBBertEmbedder constructor checks) |
| `tests/verify_pooling_gpu.py` | Create standalone GPU verification script |

## Git

Branch: `feature/configurable-pooling` (create from `main`).

Commits:
1. `feat(embeddings): add configurable pooling strategy to KBBertEmbedder`
2. `test(embeddings): add pooling strategy tests and GPU verification script`
