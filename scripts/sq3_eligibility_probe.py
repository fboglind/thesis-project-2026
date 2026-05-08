"""SQ3 Stage-1 eligibility probe.

Reports the eligibility-filtered set summary required by §14 of
``phase_5_sq3_methodology.md``: total / eligible row counts, the
exclusion breakdown, the cosine quartile cutpoints, and a
quartile-by-diagnosis contingency table for FB to confirm each quartile
contains a reasonable mix of diagnostic groups.

The script is idempotent — running it twice on the same input produces
identical output. It does not draw any sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thesis_project.evaluation.sq3_sampling import (
    assign_quartile,
    compute_quartile_cutpoints,
    load_eligible_pairs,
)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Avoids the ``tabulate`` optional dependency.
    """
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    rows = []
    for idx, row in df.iterrows():
        rows.append([str(idx)] + [str(v) for v in row.tolist()])
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([header, sep, body])


def _build_summary(
    input_path: Path,
    df_full: pd.DataFrame,
    counts: dict[str, int],
    cutpoints: tuple[float, float, float],
    contingency: pd.DataFrame,
) -> str:
    n_both = int((df_full["is_non_response"] & df_full["is_exact_match"]).sum())
    q25, q50, q75 = cutpoints

    lines: list[str] = []
    lines.append("# SQ3 Eligibility Probe")
    lines.append("")
    lines.append(f"Input file: `{input_path}`")
    lines.append("")
    lines.append("## Row counts")
    lines.append("")
    lines.append(f"- Total rows: {counts['n_total']}")
    lines.append(f"- Excluded (is_non_response): {counts['n_non_response']}")
    lines.append(f"- Excluded (is_exact_match): {counts['n_exact_match']}")
    lines.append(f"- Excluded (both flags True): {n_both}")
    lines.append(
        f"- Eligible rows (row-level, pre-dedup): {counts['n_after_filter']}"
    )
    lines.append(
        f"- Duplicates dropped at (gold, normalized): "
        f"{counts['n_duplicates_dropped']}"
    )
    lines.append(
        f"- Eligible pairs (post-dedup, used for stratification): "
        f"{counts['n_after_dedup']}"
    )
    lines.append("")
    lines.append("## Cosine quartile cutpoints (eligibility-filtered)")
    lines.append("")
    lines.append(f"- Q25: {q25:.6f}")
    lines.append(f"- Q50: {q50:.6f}")
    lines.append(f"- Q75: {q75:.6f}")
    lines.append("")
    lines.append("## Quartile × diagnosis contingency")
    lines.append("")
    lines.append(_df_to_markdown(contingency))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SQ3 Stage-1 eligibility probe."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the BNT scored-results CSV (primary model).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory in which to write sq3_eligibility_probe.md.",
    )
    args = parser.parse_args(argv)

    df_full = pd.read_csv(args.input)
    eligible, counts = load_eligible_pairs(args.input, return_counts=True)
    eligible = assign_quartile(eligible)
    cutpoints = compute_quartile_cutpoints(eligible)

    if "diagnosis" in eligible.columns:
        contingency = (
            eligible.assign(_q=eligible["cosine_quartile"].astype("Int64"))
            .pivot_table(
                index="_q",
                columns="diagnosis",
                values="cosine_sim",
                aggfunc="count",
                fill_value=0,
            )
            .rename_axis(index="cosine_quartile")
        )
    else:
        contingency = pd.DataFrame(
            {"note": ["diagnosis column not present in input"]}
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_summary(args.input, df_full, counts, cutpoints, contingency)
    out_path = args.output_dir / "sq3_eligibility_probe.md"
    out_path.write_text(summary, encoding="utf-8")

    print(f"Total rows: {counts['n_total']}")
    print(
        f"Excluded — non-response: {counts['n_non_response']}, "
        f"exact-match: {counts['n_exact_match']}"
    )
    print(
        f"Eligible rows (pre-dedup): {counts['n_after_filter']}; "
        f"duplicates dropped: {counts['n_duplicates_dropped']}; "
        f"eligible pairs (post-dedup): {counts['n_after_dedup']}"
    )
    q25, q50, q75 = cutpoints
    print(f"Quartile cutpoints: Q25={q25:.6f}, Q50={q50:.6f}, Q75={q75:.6f}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
