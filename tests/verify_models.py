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
    print("\nCosine similarity matrix:")
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
    print("\nKey comparisons:")
    pairs = [("kamel", "elefant"), ("kamel", "cykel"), ("hund", "katt"), ("djur", "kamel")]
    for w1, w2 in pairs:
        i, j = words.index(w1), words.index(w2)
        print(f"  sim({w1}, {w2}) = {sim_matrix[i,j]:.4f}")
