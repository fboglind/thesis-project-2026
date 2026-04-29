"""Probe each comparison_model on a small Swedish-word similarity matrix.

Goal: catch silent prompt-formatting bugs before running the full BNT sweep.
A model with off-diagonal std < 0.05 is producing collapsed similarities and
should be investigated (likely missing prompt template).

Usage:
    python scripts/probe_models.py
    python scripts/probe_models.py --models Qwen3-0.6B EmbGemma-300M
    python scripts/probe_models.py --skip Harrier-0.6B  # skip slow ones
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Probe vocabulary — six Swedish words spanning two semantic clusters.
# Animals: kamel, häst, elefant. Objects: cykel, bord, dator.
# A working model should give animal-animal > animal-object similarities.
PROBE_WORDS = ["kamel", "häst", "elefant", "cykel", "bord", "dator"]
ANIMAL_IDX = [0, 1, 2]
OBJECT_IDX = [3, 4, 5]

# BNT-relevant pairs — these come from real BNT items + plausible responses.
DIAGNOSTIC_PAIRS = [
    ("kamel", "häst",       "coordinate (animal)"),
    ("kamel", "elefant",    "coordinate (animal)"),
    ("kamel", "djur",       "superordinate"),
    ("kamel", "dromedar",   "near-synonym"),
    ("kamel", "puckelrygg", "circumlocution"),
    ("kamel", "cykel",      "unrelated"),
    ("kamel", "bord",       "unrelated"),
]


def load_config():
    cfg_path = PROJECT_ROOT / "configs" / "_default_configs.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def make_embedder(model_cfg):
    """Build an Embedder from a comparison_models YAML entry."""
    family = model_cfg["family"]
    name = model_cfg["name"]

    if family == "encoder":
        from thesis_project.embeddings.encoder import KBBertEmbedder
        strategy = model_cfg.get("embedding_strategy", "cls_token")
        pooling_map = {"cls_token": "cls", "mean_pooling": "mean"}
        if strategy not in pooling_map:
            raise ValueError(
                f"Unknown embedding_strategy {strategy!r} for {model_cfg.get('label')!r}; "
                f"expected one of {sorted(pooling_map)}"
            )
        return KBBertEmbedder(name, pooling=pooling_map[strategy])
    else:
        from thesis_project.embeddings.encoder import SentenceTransformerEmbedder
        kwargs = {}
        if "prompt_template" in model_cfg:
            kwargs["prefix"] = model_cfg["prompt_template"].replace("{text}", "")
        if "model_kwargs" in model_cfg:
            kwargs["model_kwargs"] = model_cfg["model_kwargs"]
        if "encode_kwargs" in model_cfg:
            kwargs["encode_kwargs"] = model_cfg["encode_kwargs"]
        return SentenceTransformerEmbedder(name, **kwargs)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def probe_model(label: str, model_cfg: dict) -> dict | None:
    print(f"\n{'='*70}")
    print(f"  {label}  ({model_cfg['name']})")
    print(f"{'='*70}")

    t0 = time.time()
    try:
        emb = make_embedder(model_cfg)
    except Exception as e:
        print(f"  FAILED to load: {type(e).__name__}: {e}")
        return None
    print(f"  loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    try:
        # Some embedders accept lists; some need single calls. Try batch first.
        if hasattr(emb, "embed_batch"):
            vecs = emb.embed_batch(PROBE_WORDS)
        else:
            vecs = np.stack([np.asarray(emb.embed(w)).squeeze() for w in PROBE_WORDS])
        vecs = np.asarray(vecs)
        if vecs.ndim == 3:  # some embedders return [n, 1, d]
            vecs = vecs.squeeze(1)
    except Exception as e:
        print(f"  FAILED to embed: {type(e).__name__}: {e}")
        return None
    print(f"  embedded {len(PROBE_WORDS)} words in {time.time() - t0:.2f}s, dim={vecs.shape[-1]}")

    # Similarity matrix
    sim = np.zeros((len(PROBE_WORDS), len(PROBE_WORDS)))
    for i in range(len(PROBE_WORDS)):
        for j in range(len(PROBE_WORDS)):
            sim[i, j] = cosine(vecs[i], vecs[j])

    # Off-diagonal stats
    iu = np.triu_indices(len(PROBE_WORDS), k=1)
    off_diag = sim[iu]
    print(f"\n  Off-diagonal: mean={off_diag.mean():.3f}, "
          f"std={off_diag.std():.3f}, range=[{off_diag.min():.3f}, {off_diag.max():.3f}]")

    # Health check
    flag = ""
    if off_diag.std() < 0.05:
        flag = "  ⚠ COLLAPSED — check prompt formatting"
    elif off_diag.mean() > 0.95:
        flag = "  ⚠ HIGH-MEAN — likely compressed similarity space"
    if flag:
        print(flag)

    # Animal-animal vs. animal-object
    aa = [sim[i, j] for i in ANIMAL_IDX for j in ANIMAL_IDX if i < j]
    ao = [sim[i, j] for i in ANIMAL_IDX for j in OBJECT_IDX]
    print(f"\n  animal-animal mean similarity: {np.mean(aa):.3f}")
    print(f"  animal-object mean similarity: {np.mean(ao):.3f}")
    print(f"  separation (aa - ao):          {np.mean(aa) - np.mean(ao):+.3f}")
    if np.mean(aa) - np.mean(ao) < 0.05:
        print("  ⚠ POOR SEPARATION — model may not distinguish semantic categories")

    # Print 6×6 similarity matrix
    print(f"\n  Similarity matrix:")
    header = "         " + "  ".join(f"{w:>7s}" for w in PROBE_WORDS)
    print(header)
    for i, w in enumerate(PROBE_WORDS):
        row = f"  {w:>6s} " + "  ".join(f"{sim[i, j]:7.3f}" for j in range(len(PROBE_WORDS)))
        print(row)

    # BNT-relevant diagnostic pairs
    print(f"\n  BNT-relevant pairs (target = kamel):")
    pair_results = []
    for a, b, kind in DIAGNOSTIC_PAIRS:
        try:
            va = np.asarray(emb.embed(a)).squeeze()
            vb = np.asarray(emb.embed(b)).squeeze()
            s = cosine(va, vb)
            pair_results.append((a, b, kind, s))
            print(f"    {a:>10s} - {b:<12s} [{kind:<22s}]: {s:.3f}")
        except Exception as e:
            print(f"    {a:>10s} - {b:<12s} [{kind:<22s}]: FAILED ({e})")

    return {
        "label": label,
        "name": model_cfg["name"],
        "dim": int(vecs.shape[-1]),
        "off_diag_mean": float(off_diag.mean()),
        "off_diag_std": float(off_diag.std()),
        "aa_mean": float(np.mean(aa)),
        "ao_mean": float(np.mean(ao)),
        "separation": float(np.mean(aa) - np.mean(ao)),
        "pairs": pair_results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", help="Subset of model labels to probe")
    p.add_argument("--skip", nargs="+", default=[], help="Model labels to skip")
    args = p.parse_args()

    cfg = load_config()
    models = cfg["comparison_models"]
    if args.models:
        models = [m for m in models if m["label"] in args.models]
    if args.skip:
        models = [m for m in models if m["label"] not in args.skip]

    print(f"Probing {len(models)} model(s) on {len(PROBE_WORDS)} Swedish words")
    print(f"Words: {', '.join(PROBE_WORDS)}")

    results = []
    for m in models:
        r = probe_model(m["label"], m)
        if r is not None:
            results.append(r)

    # Summary table
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<24s} {'dim':>5s} {'mean':>7s} {'std':>7s} "
          f"{'aa':>7s} {'ao':>7s} {'sep':>7s}  flag")
    print("-" * 80)
    for r in results:
        flag = ""
        if r["off_diag_std"] < 0.05:
            flag = "COLLAPSED"
        elif r["separation"] < 0.05:
            flag = "POOR-SEP"
        elif r["off_diag_mean"] > 0.95:
            flag = "HIGH-MEAN"
        else:
            flag = "ok"
        print(f"{r['label']:<24s} {r['dim']:>5d} "
              f"{r['off_diag_mean']:>7.3f} {r['off_diag_std']:>7.3f} "
              f"{r['aa_mean']:>7.3f} {r['ao_mean']:>7.3f} {r['separation']:>+7.3f}  {flag}")


if __name__ == "__main__":
    main()