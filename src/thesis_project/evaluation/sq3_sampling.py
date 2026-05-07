"""SQ3 stratified sampling.

Stage-1 sampling primitives for the human-rater alignment evaluation
pre-registered in ``phase_5_sq3_methodology.md``.

The methodological commitments below are pinned by that document and must
not be revised in this module.

* Eligibility: ``is_non_response == False`` AND ``is_exact_match == False``.
* Stratification axis: cosine similarity from the primary-model BNT
  scored-results CSV (Swedish SBERT). Quartile cutpoints are computed on
  the eligibility-filtered set; the four other model CSVs are not used
  for stratification.
* Sample size: 200 baseline (50 per quartile) or 300 extension
  (75 per quartile); the first 50 per quartile constitute the rated
  set under either configuration.
* Random seeds: ``20260507`` for the primary stratified sample,
  ``20260514`` for the 20-pair training set.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_SEED = 20260507
TRAINING_SEED = 20260514

RATER_COLUMNS = [
    "pair_id",
    "target",
    "response",
    "rating",
    "category",
    "is_compound",
    "notes",
]

SENSITIVE_COLUMNS = (
    "participant_id",
    "gender",
    "age",
    "diagnosis",
    "mmse",
    "cosine_sim",
)


def load_eligible_pairs(bnt_results_path: Path) -> pd.DataFrame:
    """Load the BNT scored-results CSV and return only eligible pairs.

    Eligibility:
        ``is_non_response == False`` AND ``is_exact_match == False``.

    The returned DataFrame is a copy, indexed ``0..n-1``, with all
    original columns preserved.
    """
    bnt_results_path = Path(bnt_results_path)
    df = pd.read_csv(bnt_results_path)
    required = {"is_non_response", "is_exact_match"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input {bnt_results_path} is missing required column(s): "
            f"{sorted(missing)}"
        )
    eligible = df[(~df["is_non_response"]) & (~df["is_exact_match"])].copy()
    eligible.reset_index(drop=True, inplace=True)
    return eligible


def compute_quartile_cutpoints(
    df: pd.DataFrame,
    cosine_col: str = "cosine_sim",
) -> tuple[float, float, float]:
    """Return the ``(q25, q50, q75)`` cutpoints of ``cosine_col`` in ``df``.

    Cutpoints are reported as floats; the quartile assignment uses
    ``pd.qcut`` with ``labels=False`` on the same data.
    """
    if cosine_col not in df.columns:
        raise KeyError(f"Column {cosine_col!r} not found in DataFrame")
    q25, q50, q75 = df[cosine_col].quantile([0.25, 0.5, 0.75]).tolist()
    return float(q25), float(q50), float(q75)


def assign_quartile(
    df: pd.DataFrame,
    cosine_col: str = "cosine_sim",
    out_col: str = "cosine_quartile",
) -> pd.DataFrame:
    """Add a ``0..3`` quartile column based on ``cosine_col``. Returns a copy."""
    if cosine_col not in df.columns:
        raise KeyError(f"Column {cosine_col!r} not found in DataFrame")
    out = df.copy()
    out[out_col] = pd.qcut(out[cosine_col], q=4, labels=False, duplicates="drop")
    out[out_col] = out[out_col].astype("Int64")
    return out


def _deterministic_pair_id(seed: int, idx: int) -> str:
    """Generate a UUID4-formatted, deterministic pair id."""
    rng = np.random.default_rng(seed + idx)
    raw = rng.bytes(16)
    return str(uuid.UUID(bytes=raw, version=4))


def draw_stratified_sample(
    df: pd.DataFrame,
    quartile_col: str = "cosine_quartile",
    per_quartile: int = 50,
    seed: int = PRIMARY_SEED,
) -> pd.DataFrame:
    """Draw ``per_quartile`` rows from each quartile, deterministic on seed.

    Returns a DataFrame with the sampled rows plus a ``pair_id`` column
    (deterministic UUID4 derived from ``seed``). The output is sorted by
    ``pair_id``, not by quartile, so quartile information does not leak
    to the rater through row order.
    """
    if quartile_col not in df.columns:
        raise KeyError(f"Column {quartile_col!r} not found in DataFrame")
    rng = np.random.default_rng(seed)
    sampled_blocks: list[pd.DataFrame] = []
    for q in sorted(df[quartile_col].dropna().unique()):
        sub = df[df[quartile_col] == q]
        if len(sub) < per_quartile:
            raise ValueError(
                f"Quartile {int(q)} has only {len(sub)} eligible rows, "
                f"need {per_quartile}."
            )
        chosen = rng.choice(sub.index.to_numpy(), size=per_quartile, replace=False)
        sampled_blocks.append(df.loc[chosen])
    sample = pd.concat(sampled_blocks, axis=0).reset_index(drop=True)
    sample["pair_id"] = [
        _deterministic_pair_id(seed, i) for i in range(len(sample))
    ]
    sample = sample.sort_values("pair_id", kind="stable").reset_index(drop=True)
    return sample


def _resolve_response_column(df: pd.DataFrame) -> str:
    """Return the column name to use as the response text.

    The BNT scored-results CSVs ship with both ``raw_response`` and
    ``normalized``. The raw form is what a rater should see; we fall back
    to ``normalized`` if a CSV variant lacks the raw column.
    """
    for candidate in ("raw_response", "normalized"):
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "Sample DataFrame is missing both 'raw_response' and 'normalized' "
        "columns; cannot construct rater-facing CSV."
    )


def _resolve_target_column(df: pd.DataFrame) -> str:
    """Return the column name carrying the target (gold) word."""
    for candidate in ("target", "gold"):
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "Sample DataFrame is missing both 'target' and 'gold' columns; "
        "cannot construct rater-facing CSV."
    )


def make_rater_csv(
    sample: pd.DataFrame,
    rater_seed: int,
    output_path: Path,
) -> None:
    """Write a rater-facing CSV.

    Output columns (in order):

        pair_id, target, response, rating, category, is_compound, notes

    The pair order is permuted by ``rater_seed`` for this rater. The
    ``rating``, ``category``, ``is_compound`` and ``notes`` columns are
    empty.

    Sensitive columns (``participant_id``, ``gender``, ``age``,
    ``diagnosis``, ``mmse``, ``cosine_sim``) are explicitly stripped
    before writing.
    """
    if "pair_id" not in sample.columns:
        raise KeyError("Sample DataFrame must contain a 'pair_id' column")
    target_col = _resolve_target_column(sample)
    response_col = _resolve_response_column(sample)

    rater = pd.DataFrame(
        {
            "pair_id": sample["pair_id"].to_numpy(),
            "target": sample[target_col].to_numpy(),
            "response": sample[response_col].to_numpy(),
            "rating": "",
            "category": "",
            "is_compound": "",
            "notes": "",
        }
    )
    rng = np.random.default_rng(rater_seed)
    perm = rng.permutation(len(rater))
    rater = rater.iloc[perm].reset_index(drop=True)

    if not set(rater.columns) == set(RATER_COLUMNS):
        raise AssertionError(
            "Rater CSV column set drifted from the methodology-pinned set."
        )
    rater = rater[RATER_COLUMNS]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rater.to_csv(output_path, index=False)


def draw_training_sample(
    eligible: pd.DataFrame,
    live_pair_indices: pd.Index,
    n: int = 20,
    seed: int = TRAINING_SEED,
) -> pd.DataFrame:
    """Draw an ``n``-pair training set disjoint from the live sample.

    The training set is sampled from ``eligible`` excluding any rows
    whose original index appears in ``live_pair_indices``. The seed
    ``20260514`` is fixed by the methodology document.
    """
    pool = eligible.drop(index=live_pair_indices, errors="ignore")
    if len(pool) < n:
        raise ValueError(
            f"Training pool has only {len(pool)} rows after excluding "
            f"the live sample; need {n}."
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool.index.to_numpy(), size=n, replace=False)
    train = pool.loc[chosen].reset_index(drop=True)
    train["pair_id"] = [
        _deterministic_pair_id(seed, i) for i in range(len(train))
    ]
    train = train.sort_values("pair_id", kind="stable").reset_index(drop=True)
    return train
