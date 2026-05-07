"""SQ3 Stage-1 stratified sample draw.

Draws the 200- or 300-pair stratified rated set, writes the non-blinded
``sq3_sampled_pairs.csv`` master file (treat as confidential while
raters are working), and writes one blinded ``sq3_ratings_<rater>.csv``
per rater plus a 20-pair training set per rater.

CLI::

    python scripts/sq3_sample_pairs.py \
        --input data/processed/bnt_scored_results_sbert.csv \
        --target-size 200 \
        --raters FB,collaborator \
        --output-dir data/processed/sq3/

Per-rater seed: ``20260507 + index_in_list``. Training-set seed
(``20260514``) is fixed by the methodology document.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thesis_project.evaluation.sq3_sampling import (
    PRIMARY_SEED,
    TRAINING_SEED,
    assign_quartile,
    draw_stratified_sample,
    draw_training_sample,
    load_eligible_pairs,
    make_rater_csv,
)

MASTER_COLUMNS_PREFERRED = [
    "pair_id",
    "target",
    "response",
    "cosine_sim",
    "cosine_quartile",
    "participant_id",
    "diagnosis",
    "model",
]


def _build_master(sample: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Return the non-blinded master DataFrame with normalised columns."""
    target_col = "target" if "target" in sample.columns else "gold"
    response_col = (
        "raw_response" if "raw_response" in sample.columns else "normalized"
    )
    out = pd.DataFrame()
    out["pair_id"] = sample["pair_id"].to_numpy()
    out["target"] = sample[target_col].to_numpy()
    out["response"] = sample[response_col].to_numpy()
    out["cosine_sim"] = sample["cosine_sim"].to_numpy()
    out["cosine_quartile"] = sample["cosine_quartile"].to_numpy()
    out["participant_id"] = (
        sample["participant_id"].to_numpy()
        if "participant_id" in sample.columns
        else (sample["user"].to_numpy() if "user" in sample.columns else "")
    )
    out["diagnosis"] = (
        sample["diagnosis"].to_numpy()
        if "diagnosis" in sample.columns
        else ""
    )
    out["model"] = model_name
    return out[MASTER_COLUMNS_PREFERRED]


def _infer_model_name(input_path: Path) -> str:
    """Best-effort short label of the model behind the input CSV."""
    stem = input_path.stem
    prefix = "bnt_scored_results_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SQ3 Stage-1 sample draw.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--target-size",
        type=int,
        default=200,
        choices=(200, 300),
        help="200 (50 per quartile) or 300 (75 per quartile).",
    )
    parser.add_argument(
        "--raters",
        required=True,
        type=str,
        help="Comma-separated rater identifiers, e.g. 'FB,collaborator'.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    raters = [r.strip() for r in args.raters.split(",") if r.strip()]
    if not raters:
        parser.error("--raters must be a non-empty comma-separated list")

    per_quartile = args.target_size // 4

    eligible = load_eligible_pairs(args.input)
    eligible = assign_quartile(eligible)
    sample = draw_stratified_sample(
        eligible, per_quartile=per_quartile, seed=PRIMARY_SEED
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_name = _infer_model_name(args.input)
    master = _build_master(sample, model_name)
    master_path = args.output_dir / "sq3_sampled_pairs.csv"
    master.to_csv(master_path, index=False)

    keys = ["target", "response", "cosine_sim"] if "target" in sample.columns else ["gold", "raw_response", "cosine_sim"]
    sample_tuples = set(
        map(tuple, sample[keys].itertuples(index=False, name=None))
    )
    elig_tuples = list(
        map(tuple, eligible[keys].itertuples(index=False, name=None))
    )
    live_idx = pd.Index(
        [i for i, t in enumerate(elig_tuples) if t in sample_tuples]
    )

    for index, rater in enumerate(raters):
        rater_path = args.output_dir / f"sq3_ratings_{rater}.csv"
        make_rater_csv(
            sample,
            rater_seed=PRIMARY_SEED + index,
            output_path=rater_path,
        )

        train = draw_training_sample(
            eligible, live_idx, n=20, seed=TRAINING_SEED
        )
        train_path = args.output_dir / f"sq3_training_set_{rater}.csv"
        # Each rater gets a permuted view of the same training pool, so the
        # disjointness from the live set is preserved while order varies.
        make_rater_csv(
            train,
            rater_seed=TRAINING_SEED + index,
            output_path=train_path,
        )

    print(f"Wrote master file: {master_path} ({len(master)} rows)")
    for rater in raters:
        print(f"Wrote rater CSV: {args.output_dir / f'sq3_ratings_{rater}.csv'}")
        print(
            f"Wrote training CSV: "
            f"{args.output_dir / f'sq3_training_set_{rater}.csv'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
