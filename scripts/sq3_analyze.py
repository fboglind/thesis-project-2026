"""SQ3 Stage-2 entry point: run all post-rating analyses.

CLI::

    python scripts/sq3_analyze.py \\
        --ratings-dir data/processed/sq3/ \\
        --branch {multi,sole} \\
        --sampled-pairs data/processed/sq3/sq3_sampled_pairs.csv \\
        --models bnt_scored_results_*.csv \\
        --output-dir data/processed/sq3/reports/

The script:

1. Loads ``sq3_ratings_<rater>.csv`` (Branch A) or
   ``sq3_ratings_<rater>_round{1,2}.csv`` (Branch B) from
   ``--ratings-dir``.
2. Validates ratings strictly: all expected pair_ids present, no NaN
   ratings, ratings in {0,1,2,3}, categories from the closed list.
3. Runs reliability, writes ``sq3_reliability_report.md`` with the
   interpretability flag on the first line.
4. Runs rater-model Spearman per-model / per-quartile / per-category;
   writes ``sq3_agreement_report.md`` and ``sq3_agreement_table.csv``.
5. Loads the SaldoGraph from the Phase A pickle if present (raises a
   clear error pointing the user to ``scripts/build_saldo_graph.py``
   if not). Adds rater-SALDO Spearman to the agreement report.
6. Builds the divergence catalog and writes
   ``sq3_divergence_catalog.csv``.
7. Prints a summary to stdout.

The script exits with a clear error message at validation failure.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

from thesis_project.evaluation.sq3_agreement import (
    rater_model_per_category,
    rater_model_per_quartile,
    rater_model_spearman,
    rater_saldo_spearman,
)
from thesis_project.evaluation.sq3_divergence import compute_divergence_catalog
from thesis_project.evaluation.sq3_reliability import (
    VALID_CATEGORIES,
    VALID_RATINGS,
    compute_reliability,
)

SALDO_PICKLE_DEFAULT = Path("data/lexical/saldo.pkl")


def _validate_ratings_dir(rating_files: list[Path], expected_pairs: set[str]) -> None:
    """Strict, fail-fast validation of every rating file."""
    for path in rating_files:
        df = pd.read_csv(path)
        for col in ("pair_id", "rating", "category", "is_compound"):
            if col not in df.columns:
                sys.exit(f"ERROR: {path} is missing required column '{col}'.")
        if df["rating"].isna().any():
            bad = df.loc[df["rating"].isna(), "pair_id"].tolist()
            sys.exit(
                f"ERROR: {path} has NaN rating(s) for pair_id(s): {bad}. "
                f"Out-of-range or missing ratings must be filled before "
                f"running Stage 2."
            )
        invalid = sorted(set(df["rating"].dropna().unique()) - VALID_RATINGS)
        if invalid:
            sys.exit(
                f"ERROR: {path} has rating(s) outside {sorted(VALID_RATINGS)}: {invalid}."
            )
        bad_cats = sorted(
            c for c in df["category"].dropna().unique() if c not in VALID_CATEGORIES
        )
        if bad_cats:
            sys.exit(
                f"ERROR: {path} has categor(ies) outside the closed list: "
                f"{bad_cats}. Allowed: {sorted(VALID_CATEGORIES)}."
            )
        if expected_pairs and not expected_pairs.issubset(set(df["pair_id"])):
            missing = sorted(expected_pairs - set(df["pair_id"]))[:10]
            sys.exit(
                f"ERROR: {path} is missing pair_id(s) (showing up to 10): "
                f"{missing}."
            )


def _load_consolidated_ratings(
    rating_files: list[Path],
    branch: str,
) -> pd.DataFrame:
    """Build the consolidated ratings DataFrame.

    Branch ``multi``: per-pair mean across raters (only pairs where all
    raters provided a rating).
    Branch ``sole``: Round 1's ratings.
    """
    frames = [pd.read_csv(p) for p in rating_files]
    if branch == "multi":
        merged = frames[0][["pair_id", "rating", "category", "is_compound"]].rename(
            columns={"rating": "rating_0"}
        )
        for i, f in enumerate(frames[1:], start=1):
            merged = merged.merge(
                f[["pair_id", "rating"]].rename(columns={"rating": f"rating_{i}"}),
                on="pair_id",
                how="inner",
            )
        rating_cols = [c for c in merged.columns if c.startswith("rating_")]
        merged["rating"] = merged[rating_cols].mean(axis=1)
        return merged[["pair_id", "rating", "category", "is_compound"]]
    if branch == "sole":
        rounds = sorted(rating_files, key=lambda p: ("round1" not in p.stem, p.stem))
        round1 = pd.read_csv(rounds[0])
        return round1[["pair_id", "rating", "category", "is_compound"]]
    sys.exit(f"ERROR: unknown --branch value: {branch!r}")


def _load_saldo_graph(pickle_path: Path):
    """Load the SaldoGraph from its Phase A pickle.

    Raises a clear error if the pickle is absent.
    """
    if not pickle_path.exists():
        sys.exit(
            f"ERROR: SALDO graph pickle not found at {pickle_path}. "
            f"Build it via `python scripts/build_saldo_graph.py` (Phase A) "
            f"before running Stage 2."
        )
    try:
        from thesis_project.lexical.saldo import SaldoGraph  # noqa: F401
    except Exception as exc:  # pragma: no cover - phase-dependent
        sys.exit(
            f"ERROR: cannot import SaldoGraph: {exc!r}. "
            f"Phase A must be implemented before Stage 2 can run on real "
            f"SALDO data. (The unit tests exercise SALDO via a duck-typed "
            f"mock; production runs require Phase A.)"
        )
    from thesis_project.lexical.saldo import SaldoGraph

    return SaldoGraph.from_pickle(pickle_path)


def _write_reliability_report(report, out_path: Path) -> None:
    lines: list[str] = []
    lines.append(f"Interpretability flag: **{report.interpretability_flag}**")
    lines.append("")
    lines.append("# SQ3 Reliability Report")
    lines.append("")
    lines.append(f"Branch: `{report.branch}`")
    lines.append(f"N pairs: {report.n_pairs}")
    lines.append(f"Rater IDs: {', '.join(report.rater_ids)}")
    lines.append("")
    lines.append("## Pairwise quadratic-weighted Cohen's kappa")
    lines.append("")
    for (a, b), v in report.weighted_kappa.items():
        lines.append(f"- {a} × {b}: {v:.4f}")
    lines.append("")
    lines.append("## Pairwise Spearman ρ")
    lines.append("")
    for (a, b), v in report.spearman_rho.items():
        lines.append(f"- {a} × {b}: {v:.4f}")
    lines.append("")
    lines.append("## Pairwise unweighted κ on category")
    lines.append("")
    for (a, b), v in report.category_kappa.items():
        lines.append(f"- {a} × {b}: {v:.4f}")
    lines.append("")
    lines.append("## Pairwise unweighted κ on is_compound")
    lines.append("")
    for (a, b), v in report.compound_flag_kappa.items():
        lines.append(f"- {a} × {b}: {v:.4f}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_agreement_report(
    overall: dict,
    per_quartile: pd.DataFrame,
    per_category: pd.DataFrame,
    saldo_block: str,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# SQ3 Rater-Model Agreement Report")
    lines.append("")
    lines.append("## Overall Spearman ρ (rater vs model cosine)")
    lines.append("")
    for model, res in overall.items():
        lines.append(
            f"- **{model}** (n={res.n}): ρ = {res.spearman_rho:.4f}, "
            f"95% CI [{res.ci_low:.4f}, {res.ci_high:.4f}]"
        )
    lines.append("")
    lines.append("## Per-quartile Spearman ρ")
    lines.append("")
    lines.append(per_quartile.to_string(index=False))
    lines.append("")
    lines.append("## Per-category Spearman ρ")
    lines.append("")
    lines.append(per_category.to_string(index=False))
    lines.append("")
    if saldo_block:
        lines.append("## SALDO scorer comparison")
        lines.append("")
        lines.append(saldo_block)
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SQ3 Stage-2 entry point.")
    parser.add_argument("--ratings-dir", required=True, type=Path)
    parser.add_argument("--branch", required=True, choices=("multi", "sole"))
    parser.add_argument("--sampled-pairs", required=True, type=Path)
    parser.add_argument(
        "--models",
        required=True,
        type=str,
        nargs="+",
        help="Glob pattern(s) for the per-model BNT scored-results CSVs.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--saldo-pickle",
        type=Path,
        default=SALDO_PICKLE_DEFAULT,
        help="Path to the SALDO graph pickle (Phase A artifact).",
    )
    parser.add_argument(
        "--skip-saldo",
        action="store_true",
        help="Skip the SALDO scorer comparison (use only when Phase A "
        "is not yet available; tests exercise this path).",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: discover rating files.
    if args.branch == "multi":
        pattern = "sq3_*ratings_*.csv"
        rating_files = sorted(
            p for p in args.ratings_dir.glob(pattern) if "round" not in p.stem
        )
    else:
        pattern = "sq3_*ratings_*round*.csv"
        rating_files = sorted(args.ratings_dir.glob(pattern))
    if len(rating_files) < 2:
        sys.exit(
            f"ERROR: expected ≥2 rating files in {args.ratings_dir} matching "
            f"{pattern!r}, found {len(rating_files)}."
        )

    # Step 2: validate.
    sampled = pd.read_csv(args.sampled_pairs)
    _validate_ratings_dir(rating_files, expected_pairs=set())

    # Step 3: reliability.
    branch_key = "multi" if args.branch == "multi" else "sole_test_retest"
    report = compute_reliability(rating_files, branch=branch_key)
    _write_reliability_report(report, args.output_dir / "sq3_reliability_report.md")

    # Step 4: rater-model agreement.
    consolidated = _load_consolidated_ratings(rating_files, args.branch)
    model_paths: list[Path] = []
    for pattern in args.models:
        if any(ch in pattern for ch in "*?["):
            model_paths.extend(Path(p) for p in glob.glob(pattern))
        else:
            model_paths.append(Path(pattern))
    if not model_paths:
        sys.exit(f"ERROR: no model CSVs matched --models {args.models}.")

    model_cosines: dict[str, pd.DataFrame] = {}
    for p in model_paths:
        df = pd.read_csv(p)
        if "pair_id" not in df.columns:
            df = df.merge(
                sampled[["pair_id", "target", "response"]],
                left_on=["gold", "raw_response"]
                if "gold" in df.columns and "raw_response" in df.columns
                else ["target", "response"],
                right_on=["target", "response"],
                how="inner",
            )
        model_cosines[p.stem] = df[["pair_id", "cosine_sim"]].copy()

    overall = rater_model_spearman(consolidated, model_cosines)
    per_q = rater_model_per_quartile(consolidated, sampled, model_cosines)
    per_c = rater_model_per_category(consolidated, model_cosines)
    per_q.to_csv(args.output_dir / "sq3_agreement_table.csv", index=False)

    # Step 5: SALDO comparison (skippable for tests that don't need Phase A).
    saldo_block = ""
    if not args.skip_saldo:
        graph = _load_saldo_graph(args.saldo_pickle)
        saldo_res = rater_saldo_spearman(consolidated, graph, sampled)
        saldo_block = (
            f"- N total: {saldo_res.n_total}\n"
            f"- N in-vocab: {saldo_res.n_in_vocab}\n"
            f"- OOV rate: {saldo_res.oov_rate:.3f}\n"
            f"- Path-length ρ: {saldo_res.path_spearman.spearman_rho:.4f} "
            f"[{saldo_res.path_spearman.ci_low:.4f}, "
            f"{saldo_res.path_spearman.ci_high:.4f}]\n"
            f"- Wu-Palmer ρ: {saldo_res.wu_palmer_spearman.spearman_rho:.4f} "
            f"[{saldo_res.wu_palmer_spearman.ci_low:.4f}, "
            f"{saldo_res.wu_palmer_spearman.ci_high:.4f}]"
        )

    _write_agreement_report(
        overall,
        per_q,
        per_c,
        saldo_block,
        args.output_dir / "sq3_agreement_report.md",
    )

    # Step 6: divergence catalog.
    primary_name = next(iter(model_cosines))
    primary_cos = model_cosines[primary_name]

    class _NullSaldo:
        def lookup(self, w): return []
        def path_length(self, *a): return None
        def wu_palmer(self, *a): return None

    saldo_for_div = _NullSaldo() if args.skip_saldo else _load_saldo_graph(args.saldo_pickle)
    catalog = compute_divergence_catalog(consolidated, sampled, primary_cos, saldo_for_div)
    catalog.to_csv(args.output_dir / "sq3_divergence_catalog.csv", index=False)

    # Step 7: stdout summary.
    primary_overall = overall[primary_name]
    n_div = int(catalog["is_divergence_case"].sum())
    print(f"Interpretability flag: {report.interpretability_flag}")
    print(
        f"Primary-model rater-model ρ: {primary_overall.spearman_rho:.4f} "
        f"[{primary_overall.ci_low:.4f}, {primary_overall.ci_high:.4f}] "
        f"(n={primary_overall.n}, model={primary_name})"
    )
    print(f"Divergence cases: {n_div} / {len(catalog)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
