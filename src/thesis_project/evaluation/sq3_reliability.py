"""SQ3 reliability analysis.

Implements §6 of ``phase_5_sq3_methodology.md``:

* Branch A (multi-annotator): full overlap; pairwise quadratic-weighted
  Cohen's kappa across raters; Spearman ρ; unweighted κ on category and
  on the ``is_compound`` boolean.
* Branch B (sole-annotator test-retest): same statistics computed
  between Round 1 and Round 2 on the 50-pair subset.

The interpretability flag is computed on the *mean* pairwise weighted
kappa, with the methodology-doc thresholds:

* ``primary``     if mean weighted-κ ≥ 0.6
* ``cautioned``   if 0.4 ≤ mean weighted-κ < 0.6
* ``exploratory`` if mean weighted-κ < 0.4

Quadratic-weighted kappa cannot be revised here without amending the
methodology document first.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

REQUIRED_COLUMNS = ("pair_id", "rating", "category", "is_compound")
VALID_RATINGS = {0, 1, 2, 3}
VALID_CATEGORIES = {
    "coordinate",
    "hypernym",
    "hyponym",
    "circumlocution",
    "phonological",
    "unrelated",
    "other",
}


@dataclass(frozen=True)
class ReliabilityReport:
    n_pairs: int
    rater_ids: list[str]
    branch: Literal["multi", "sole_test_retest"]
    weighted_kappa: dict[tuple[str, str], float]
    spearman_rho: dict[tuple[str, str], float]
    category_kappa: dict[tuple[str, str], float]
    compound_flag_kappa: dict[tuple[str, str], float]
    interpretability_flag: Literal["primary", "cautioned", "exploratory"]


def _interpretability_flag(mean_kappa: float) -> Literal["primary", "cautioned", "exploratory"]:
    if mean_kappa >= 0.6:
        return "primary"
    if mean_kappa >= 0.4:
        return "cautioned"
    return "exploratory"


def _coerce_compound(series: pd.Series) -> pd.Series:
    """Coerce free-form is_compound entries to ``bool``."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _validate_rating_frame(df: pd.DataFrame, source: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")
    if df["rating"].isna().any():
        bad = df.loc[df["rating"].isna(), "pair_id"].tolist()
        raise ValueError(f"{source}: NaN rating(s) for pair_id(s): {bad}")
    invalid = sorted(set(df["rating"].unique()) - VALID_RATINGS)
    if invalid:
        raise ValueError(
            f"{source}: rating(s) outside {{0,1,2,3}}: {invalid}"
        )
    bad_cats = sorted(
        c for c in df["category"].dropna().unique() if c not in VALID_CATEGORIES
    )
    if bad_cats:
        raise ValueError(
            f"{source}: categor(ies) outside the closed list: {bad_cats}"
        )


def _rater_id_from_path(path: Path, branch: Literal["multi", "sole_test_retest"]) -> str:
    """Extract a rater identifier from a rating CSV path.

    Live files: ``sq3_ratings_<rater>.csv``.
    Test-retest files: ``sq3_ratings_<rater>_round{1,2}.csv``.
    Test fixtures: ``sq3_mock_ratings_<rater>.csv`` is also accepted so the
    test suite can exercise this function without renaming files.
    """
    name = Path(path).stem
    for prefix in ("sq3_ratings_", "sq3_mock_ratings_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _load_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_rating_frame(df, source=str(path))
    df = df.copy()
    df["is_compound"] = _coerce_compound(df["is_compound"])
    df["rating"] = df["rating"].astype(int)
    return df


def _aligned(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``a`` and ``b`` aligned on ``pair_id`` (inner join, sorted)."""
    keys = sorted(set(a["pair_id"]) & set(b["pair_id"]))
    if not keys:
        raise ValueError("No overlapping pair_id between the two rating files")
    a2 = a.set_index("pair_id").loc[keys].reset_index()
    b2 = b.set_index("pair_id").loc[keys].reset_index()
    return a2, b2


def compute_reliability(
    rating_files: list[Path],
    branch: Literal["multi", "sole_test_retest"],
) -> ReliabilityReport:
    """Compute reliability statistics from a list of rating CSVs.

    Branch ``multi`` expects 2-3 fully-overlapping rater files.
    Branch ``sole_test_retest`` expects exactly 2 files (Round 1, Round 2)
    on the same pair_ids.
    """
    if branch == "multi":
        if not 2 <= len(rating_files) <= 3:
            raise ValueError(
                f"Branch 'multi' requires 2 or 3 rating files, got {len(rating_files)}"
            )
    elif branch == "sole_test_retest":
        if len(rating_files) != 2:
            raise ValueError(
                f"Branch 'sole_test_retest' requires exactly 2 rating files, got {len(rating_files)}"
            )
    else:
        raise ValueError(f"Unknown branch: {branch!r}")

    rater_ids: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for p in rating_files:
        rid = _rater_id_from_path(Path(p), branch)
        rater_ids.append(rid)
        frames[rid] = _load_ratings(Path(p))

    weighted_kappa: dict[tuple[str, str], float] = {}
    spearman: dict[tuple[str, str], float] = {}
    cat_kappa: dict[tuple[str, str], float] = {}
    cmp_kappa: dict[tuple[str, str], float] = {}

    for r1, r2 in combinations(rater_ids, 2):
        a, b = _aligned(frames[r1], frames[r2])
        weighted_kappa[(r1, r2)] = float(
            cohen_kappa_score(a["rating"], b["rating"], weights="quadratic", labels=[0, 1, 2, 3])
        )
        rho, _ = spearmanr(a["rating"], b["rating"])
        spearman[(r1, r2)] = float(rho) if not np.isnan(rho) else float("nan")
        cat_kappa[(r1, r2)] = float(
            cohen_kappa_score(a["category"], b["category"])
        )
        cmp_kappa[(r1, r2)] = float(
            cohen_kappa_score(a["is_compound"].astype(int), b["is_compound"].astype(int))
        )

    n_pairs = len(_aligned(frames[rater_ids[0]], frames[rater_ids[1]])[0])
    mean_kappa = float(np.mean(list(weighted_kappa.values())))
    flag = _interpretability_flag(mean_kappa)

    return ReliabilityReport(
        n_pairs=n_pairs,
        rater_ids=rater_ids,
        branch=branch,
        weighted_kappa=weighted_kappa,
        spearman_rho=spearman,
        category_kappa=cat_kappa,
        compound_flag_kappa=cmp_kappa,
        interpretability_flag=flag,
    )
