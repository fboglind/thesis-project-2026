"""Run BNT pipeline for every model in
``configs/_default_configs.yaml::comparison_models``.

Outputs are saved to ``data/processed/bnt_scored_results_<label>.csv``,
where ``<label>`` matches the ``label`` field in the YAML config.

This phase (4a): only ``--tests bnt`` is supported. Phase 4b will add svf
and fas.

Usage::

    python scripts/run_model_comparison.py
    python scripts/run_model_comparison.py --models Qwen3-0.6B Harrier-0.6B
    python scripts/run_model_comparison.py --dry-run
    python scripts/run_model_comparison.py --skip-existing
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Make src/ importable when running as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

logger = logging.getLogger("run_model_comparison")

CONFIG_PATH = _REPO_ROOT / "configs" / "_default_configs.yaml"
SUPPORTED_TESTS_PHASE_4A = {"bnt"}


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _build_prefix(entry: dict) -> str | None:
    """Translate a YAML comparison_models entry into a SentenceTransformer prefix.

    Priority:
      1. ``prompt_template`` like ``"query: {text}"`` → prefix ``"query: "``
         (any non-``{text}`` head before ``{text}`` is taken as the prefix).
      2. ``instruction`` field → prefix
         ``"Instruct: <instruction>\\nQuery: "``.
    Returns None if neither field is present.
    """
    if entry.get("prompt_template"):
        tmpl = entry["prompt_template"]
        marker = "{text}"
        if marker in tmpl:
            return tmpl.split(marker, 1)[0]
        return tmpl
    if entry.get("instruction"):
        return f"Instruct: {entry['instruction']}\nQuery: "
    return None


def _vram_mb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return None


def _build_embedder(entry: dict) -> Any:
    """Instantiate the appropriate Embedder subclass for a YAML entry."""
    from src.thesis_project.embeddings.encoder import (
        KBBertEmbedder,
        SentenceTransformerEmbedder,
    )

    family = entry.get("family", "sbert")
    name = entry["name"]
    model_kwargs = entry.get("model_kwargs") or None
    encode_kwargs = entry.get("encode_kwargs") or None
    prefix = _build_prefix(entry)

    if family == "encoder":
        strategy = entry.get("embedding_strategy", "cls_token")
        pooling = "cls" if strategy.startswith("cls") else "mean"
        return KBBertEmbedder(model_name=name, pooling=pooling)

    if family in ("sbert", "contrastive", "instruction"):
        return SentenceTransformerEmbedder(
            name,
            prefix=prefix,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    raise ValueError(f"Unknown family {family!r} for model {entry.get('label')!r}")


def _output_path(label: str, test_name: str) -> Path:
    if test_name != "bnt":  # phase 4a guard; defensive
        raise ValueError(f"Phase 4a supports BNT only, got {test_name!r}")
    return _REPO_ROOT / "data" / "processed" / f"bnt_scored_results_{label}.csv"


def _run_bnt(entry: dict, output: Path) -> None:
    """Run the BNT pipeline against a single embedder and save CSV."""
    from bnt_pipeline import (
        analyze_results,
        compute_similarity_scores,
        preprocess_responses,
        save_results,
    )
    from src.thesis_project.preprocessing.data_loader import BNT_PATH, load_bnt_data

    items_df, user_meta = load_bnt_data(BNT_PATH)
    print(f"  BNT v3 loaded: {len(items_df)} items, {len(user_meta)} users")

    responses = preprocess_responses(items_df, user_meta)
    print(f"  preprocessed: {len(responses)} (gold, response) pairs")

    embedder = _build_embedder(entry)
    print(f"  embedder ready: {type(embedder).__name__}")

    scored = compute_similarity_scores(responses, embedder)
    analyze_results(scored)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_results(scored, str(output))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Subset of YAML labels to run. Default: all entries in comparison_models.",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=["bnt"],
        help="Tests to run. Phase 4a supports 'bnt' only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the (model, test) pairs that would run, then exit.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (model, test) pairs whose output CSV already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    logger.info(
        "Per-test metadata only; cross-file joins are not valid on v3 data. "
        "See data/processed/harmonization_check_v3.csv."
    )

    unsupported = sorted(set(args.tests) - SUPPORTED_TESTS_PHASE_4A)
    if unsupported:
        print(
            f"Phase 4a supports BNT only; use Phase 4b for SVF/FAS. "
            f"Unsupported: {unsupported}",
            file=sys.stderr,
        )
        return 2

    config = _load_config()
    all_entries: list[dict] = config.get("comparison_models", [])
    if not all_entries:
        print("No comparison_models found in config.", file=sys.stderr)
        return 1

    if args.models:
        wanted = set(args.models)
        entries = [e for e in all_entries if e.get("label") in wanted]
        missing = wanted - {e.get("label") for e in entries}
        if missing:
            print(
                f"Unknown model label(s): {sorted(missing)}. "
                f"Available: {[e.get('label') for e in all_entries]}",
                file=sys.stderr,
            )
            return 2
    else:
        entries = list(all_entries)

    pairs = [(entry, test) for entry in entries for test in args.tests]
    print(f"Planned: {len(pairs)} (model, test) pair(s)")
    for entry, test in pairs:
        out = _output_path(entry["label"], test)
        marker = " [skip-existing]" if args.skip_existing and out.exists() else ""
        print(f"  - {entry['label']:18s}  test={test}  → {out.name}{marker}")

    if args.dry_run:
        return 0

    overall_status = 0
    for entry, test in pairs:
        out = _output_path(entry["label"], test)
        if args.skip_existing and out.exists():
            print(f"\n=== {entry['label']} × {test}: skip (output exists) ===")
            continue

        print(f"\n=== {entry['label']} × {test} ===")
        print(f"  model_name: {entry.get('name')}")
        print(f"  family:     {entry.get('family')}")

        t0 = time.time()
        try:
            _run_bnt(entry, out)
        except Exception as exc:
            elapsed = time.time() - t0
            print(
                f"  FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            overall_status = 1
            continue

        elapsed = time.time() - t0
        vram = _vram_mb()
        vram_str = f", peak VRAM {vram:.0f} MB" if vram is not None else ""
        print(f"  done in {elapsed:.1f}s{vram_str}")

    return overall_status


if __name__ == "__main__":
    raise SystemExit(main())
