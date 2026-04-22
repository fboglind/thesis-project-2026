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
    print(f"\n{'=' * 60}")
    print(f"Pooling: {pooling}")
    print(f"{'=' * 60}")
    emb = KBBertEmbedder(pooling=pooling)
    vecs = emb.embed_batch(words)
    sim_matrix = cosine_similarity(vecs)

    print(f"Shape: {vecs.shape}")
    print("\nCosine similarity matrix:")
    header = "         " + "  ".join(f"{w:>8s}" for w in words)
    print(header)
    for i, w in enumerate(words):
        row = f"{w:>8s} " + "  ".join(
            f"{sim_matrix[i, j]:8.4f}" for j in range(len(words))
        )
        print(row)

    mask = ~np.eye(len(words), dtype=bool)
    off_diag = sim_matrix[mask]
    print(f"\nOff-diagonal range: [{off_diag.min():.4f}, {off_diag.max():.4f}]")
    print(f"Off-diagonal std:   {off_diag.std():.4f}")
    print(f"Off-diagonal mean:  {off_diag.mean():.4f}")
