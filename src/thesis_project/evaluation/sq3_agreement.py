"""SQ3 rater-model agreement analysis.

Implements §7 of ``phase_5_sq3_methodology.md``. For each model in scope
the function computes Spearman ρ between the human rating (per-pair
mean across raters in Branch A; Round-1 ratings in Branch B) and the
model cosine, with a 1000-resample non-parametric bootstrap 95% CI on
the (rater, cosine) pairs (seed ``20260528``).

Per-quartile and per-category strata are computed identically; per-
category strata with ``n < min_n`` are emitted with NaN ρ and a flag.

The SALDO scorer comparison computes path-length and Wu-Palmer
similarity for the in-vocabulary subset of pairs and reports OOV rate.
``saldo_graph`` is duck-typed against ``SaldoGraph``: it must expose
``lookup``, ``path_length`` and ``wu_palmer``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BOOTSTRAP_SEED = 20260528
N_BOOTSTRAP_DEFAULT = 1000


@dataclass(frozen=True)
class AgreementResult:
    model_name: str
    n: int
    spearman_rho: float
    ci_low: float
    ci_high: float


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Return (low, high) percentiles of bootstrapped Spearman ρ."""
    if len(x) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bx = x[idx]
        by = y[idx]
        if np.unique(bx).size < 2 or np.unique(by).size < 2:
            rhos[i] = np.nan
            continue
        r, _ = spearmanr(bx, by)
        rhos[i] = r
    rhos = rhos[~np.isnan(rhos)]
    if rhos.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def _aligned_xy(
    ratings: pd.DataFrame,
    cosines: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner-join ratings and a model's cosine table on pair_id."""
    if "pair_id" not in ratings.columns or "rating" not in ratings.columns:
        raise KeyError("ratings DataFrame must have pair_id and rating columns")
    if "pair_id" not in cosines.columns or "cosine_sim" not in cosines.columns:
        raise KeyError("cosines DataFrame must have pair_id and cosine_sim columns")
    merged = ratings.merge(cosines, on="pair_id", how="inner")
    return merged["rating"].to_numpy(dtype=float), merged["cosine_sim"].to_numpy(dtype=float)


def rater_model_spearman(
    ratings: pd.DataFrame,
    model_cosines: dict[str, pd.DataFrame],
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, AgreementResult]:
    """For each model, Spearman ρ + bootstrap 95% CI on the (rater, cosine) pairs.

    ``ratings`` carries the consolidated ``rating`` column (per-pair mean
    across raters in Branch A; Round-1 ratings in Branch B). Each
    ``model_cosines`` entry is a DataFrame with at least ``pair_id`` and
    ``cosine_sim`` columns.
    """
    out: dict[str, AgreementResult] = {}
    for model_name, df in model_cosines.items():
        x, y = _aligned_xy(ratings, df)
        if len(x) < 3:
            out[model_name] = AgreementResult(model_name, len(x), float("nan"), float("nan"), float("nan"))
            continue
        rho, _ = spearmanr(x, y)
        low, high = _bootstrap_spearman_ci(x, y, n_bootstrap=n_bootstrap, seed=seed)
        out[model_name] = AgreementResult(
            model_name=model_name,
            n=int(len(x)),
            spearman_rho=float(rho),
            ci_low=low,
            ci_high=high,
        )
    return out


def rater_model_per_quartile(
    ratings: pd.DataFrame,
    sampled_pairs: pd.DataFrame,
    model_cosines: dict[str, pd.DataFrame],
    seed: int = BOOTSTRAP_SEED,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
) -> pd.DataFrame:
    """Spearman ρ within each cosine quartile, per model.

    Returns long-format columns:
        ``model_name, quartile, n, spearman_rho, ci_low, ci_high``.
    """
    if "cosine_quartile" not in sampled_pairs.columns:
        raise KeyError("sampled_pairs must contain a 'cosine_quartile' column")
    rows: list[dict[str, object]] = []
    rated = ratings.merge(
        sampled_pairs[["pair_id", "cosine_quartile"]], on="pair_id", how="inner"
    )
    for model_name, df in model_cosines.items():
        merged = rated.merge(df[["pair_id", "cosine_sim"]], on="pair_id", how="inner")
        for q in sorted(merged["cosine_quartile"].dropna().unique()):
            sub = merged[merged["cosine_quartile"] == q]
            x = sub["rating"].to_numpy(dtype=float)
            y = sub["cosine_sim"].to_numpy(dtype=float)
            if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
                rho = float("nan")
                low = high = float("nan")
            else:
                rho_val, _ = spearmanr(x, y)
                rho = float(rho_val)
                low, high = _bootstrap_spearman_ci(x, y, n_bootstrap=n_bootstrap, seed=seed)
            rows.append(
                {
                    "model_name": model_name,
                    "quartile": int(q),
                    "n": int(len(x)),
                    "spearman_rho": rho,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows, columns=["model_name", "quartile", "n", "spearman_rho", "ci_low", "ci_high"])


def rater_model_per_category(
    ratings: pd.DataFrame,
    model_cosines: dict[str, pd.DataFrame],
    seed: int = BOOTSTRAP_SEED,
    min_n: int = 15,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
) -> pd.DataFrame:
    """Spearman ρ within each rater-assigned primary category, per model.

    Categories with ``n < min_n`` are emitted with NaN ρ/CI and a
    ``below_min_n`` flag column set to ``True``.
    """
    if "category" not in ratings.columns:
        raise KeyError("ratings DataFrame must contain a 'category' column")
    rows: list[dict[str, object]] = []
    for model_name, df in model_cosines.items():
        merged = ratings.merge(df[["pair_id", "cosine_sim"]], on="pair_id", how="inner")
        for cat in sorted(merged["category"].dropna().unique()):
            sub = merged[merged["category"] == cat]
            x = sub["rating"].to_numpy(dtype=float)
            y = sub["cosine_sim"].to_numpy(dtype=float)
            below = len(x) < min_n
            if below or np.unique(x).size < 2 or np.unique(y).size < 2:
                rho = float("nan")
                low = high = float("nan")
            else:
                rho_val, _ = spearmanr(x, y)
                rho = float(rho_val)
                low, high = _bootstrap_spearman_ci(x, y, n_bootstrap=n_bootstrap, seed=seed)
            rows.append(
                {
                    "model_name": model_name,
                    "category": cat,
                    "n": int(len(x)),
                    "spearman_rho": rho,
                    "ci_low": low,
                    "ci_high": high,
                    "below_min_n": bool(below),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "model_name",
            "category",
            "n",
            "spearman_rho",
            "ci_low",
            "ci_high",
            "below_min_n",
        ],
    )


def _saldo_pair_similarity(
    saldo_graph,
    target: str,
    response: str,
) -> tuple[float | None, float | None]:
    """Max-over-sense-pairs SALDO similarity for one (target, response) pair.

    Returns ``(path_sim, wu_palmer_sim)`` where ``path_sim = 1/(1+path_length)``.
    Either entry is ``None`` when no sense pair yields a defined similarity.
    Returns ``(None, None)`` if either side is OOV.
    """
    t_senses = saldo_graph.lookup(target)
    r_senses = saldo_graph.lookup(response)
    if not t_senses or not r_senses:
        return None, None
    best_path: float | None = None
    best_wu: float | None = None
    for ts in t_senses:
        for rs in r_senses:
            pl = saldo_graph.path_length(ts, rs)
            if pl is not None:
                sim = 1.0 / (1.0 + float(pl))
                if best_path is None or sim > best_path:
                    best_path = sim
            wu = saldo_graph.wu_palmer(ts, rs)
            if wu is not None:
                if best_wu is None or wu > best_wu:
                    best_wu = float(wu)
    return best_path, best_wu


@dataclass(frozen=True)
class SaldoAgreementResult:
    n_in_vocab: int
    n_total: int
    oov_rate: float
    path_spearman: AgreementResult
    wu_palmer_spearman: AgreementResult


def rater_saldo_spearman(
    ratings: pd.DataFrame,
    saldo_graph,
    target_response_pairs: pd.DataFrame,
    seed: int = BOOTSTRAP_SEED,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
) -> SaldoAgreementResult:
    """Spearman ρ between rater rating and SALDO-derived similarity.

    For each ``(target, response)`` pair where both have a SALDO sense:

    * Path-length similarity = ``1 / (1 + path_length(t, r))`` (max over
      sense-pair combinations, Pakhomov 2012 default).
    * Wu-Palmer similarity via ``SaldoGraph.wu_palmer`` (max over
      sense-pair combinations).

    Pairs with at least one OOV side are excluded from the Spearman
    computation and counted in the OOV rate.
    """
    if "pair_id" not in ratings.columns or "rating" not in ratings.columns:
        raise KeyError("ratings DataFrame must have pair_id and rating columns")
    required = {"pair_id", "target", "response"}
    missing = required - set(target_response_pairs.columns)
    if missing:
        raise KeyError(f"target_response_pairs missing columns: {missing}")
    merged = ratings.merge(
        target_response_pairs[["pair_id", "target", "response"]],
        on="pair_id",
        how="inner",
    )
    n_total = len(merged)
    rating_arr: list[float] = []
    path_arr: list[float] = []
    wu_arr: list[float] = []
    n_oov = 0
    for _, row in merged.iterrows():
        path_sim, wu_sim = _saldo_pair_similarity(
            saldo_graph, str(row["target"]), str(row["response"])
        )
        if path_sim is None and wu_sim is None:
            n_oov += 1
            continue
        rating_arr.append(float(row["rating"]))
        path_arr.append(float(path_sim) if path_sim is not None else float("nan"))
        wu_arr.append(float(wu_sim) if wu_sim is not None else float("nan"))

    n_in = len(rating_arr)
    oov_rate = float(n_oov) / float(n_total) if n_total else 0.0
    rating_np = np.asarray(rating_arr, dtype=float)
    path_np = np.asarray(path_arr, dtype=float)
    wu_np = np.asarray(wu_arr, dtype=float)

    def _result(name: str, x: np.ndarray, y: np.ndarray) -> AgreementResult:
        mask = ~np.isnan(y)
        x2 = x[mask]
        y2 = y[mask]
        if len(x2) < 3 or np.unique(x2).size < 2 or np.unique(y2).size < 2:
            return AgreementResult(name, int(len(x2)), float("nan"), float("nan"), float("nan"))
        rho, _ = spearmanr(x2, y2)
        low, high = _bootstrap_spearman_ci(x2, y2, n_bootstrap=n_bootstrap, seed=seed)
        return AgreementResult(name, int(len(x2)), float(rho), low, high)

    return SaldoAgreementResult(
        n_in_vocab=n_in,
        n_total=n_total,
        oov_rate=oov_rate,
        path_spearman=_result("saldo_path", rating_np, path_np),
        wu_palmer_spearman=_result("saldo_wu_palmer", rating_np, wu_np),
    )
