# Phase 5 — SQ3: Sampling, Reliability, and Rater-Model Agreement

## Context

This phase implements the code infrastructure for SQ3 (human-rater alignment),
following the pre-registered protocol in `phase_5_sq3_methodology.md`.

The phase has **two execution stages**, but all code is written in this
phase (only execution is staged).

- **Stage 1 — Pre-rating (runs now).** Eligibility probe + sampling. Produces
  the rater-facing CSV file(s) used to collect human ratings.
- **Stage 2 — Post-rating (waits for completed rating CSVs).** Reliability
  analysis, rater-model agreement, divergence catalog. The code is written
  in this phase and runs once FB (and any second rater) returns the
  filled-in rating CSVs.

The single SQ3 evaluation runs on the BNT only. SVF and FAS are out of scope
for this phase — SQ3 itself is BNT-scoped per the methodology document.

GitHub issue: #N (to be created).

## Critical Methodological Decisions (Pre-Registered)

These decisions are pinned by the methodology document. They cannot be
revised in this phase or by Claude Code under any circumstances. Document
them in the relevant module docstrings exactly as stated.

**Eligibility filter.** Pairs with `is_non_response == True` OR
`is_exact_match == True` are excluded before stratification. This filter is
applied once, deterministically, and the resulting eligible-pair count is
reported.

**Stratification model.** Cosine similarity from `bnt_scored_results_-
SwedishSBERT.csv` is the single stratification axis. Quartiles are computed
on the eligible-pair set. The four other model CSVs are *not* used for
stratification, only for rater-model agreement reporting in Stage 2.

**Sample size.** 200 baseline (50 per quartile), with optional extension to
300 (75 per quartile). The first 50 per quartile constitute the rated set
under either configuration; the additional 25 per quartile are appended
deterministically using the same seed.

**Random seeds.**
- `20260507` — primary stratified sample.
- `20260514` — separate 20-pair training set (excluded from analysis).
- `20260521` — test-retest 50-pair subset (Branch B only).
- `20260528` — bootstrap resampling for Spearman CIs.

**Reliability metric.** Quadratic-weighted Cohen's kappa is the primary
reliability metric in both Branch A (multi-annotator IAA) and Branch B
(sole-annotator test-retest). Spearman ρ is the secondary metric. Cohen's
kappa (unweighted) is used for category-assignment agreement.

**Divergence threshold.** A pair is a divergence case if its
`|rater_score_in_unit_interval - cosine_similarity|` exceeds the 90th
percentile of that quantity across the rated set. Threshold is fixed at
the 90th percentile; resulting divergence count is data-determined.

**Bootstrap CI.** 1000 resamples with replacement on the (rater, cosine)
pairs; report 2.5th and 97.5th percentiles as the 95% CI.

## Files to Create

```
src/thesis_project/evaluation/sq3_sampling.py        # NEW — eligibility, quartiles, stratified sampling
src/thesis_project/evaluation/sq3_reliability.py     # NEW — kappa & Spearman between raters / rounds
src/thesis_project/evaluation/sq3_agreement.py       # NEW — rater-model Spearman with bootstrap CI
src/thesis_project/evaluation/sq3_divergence.py      # NEW — divergence catalog construction
src/thesis_project/evaluation/__init__.py            # UPDATE — re-export new public functions
scripts/sq3_eligibility_probe.py                     # NEW — Stage 1, prints eligibility summary
scripts/sq3_sample_pairs.py                          # NEW — Stage 1, draws and writes rater CSVs
scripts/sq3_analyze.py                               # NEW — Stage 2, runs all post-rating analyses
tests/test_sq3_sampling.py                           # NEW
tests/test_sq3_reliability.py                        # NEW
tests/test_sq3_agreement.py                          # NEW
tests/test_sq3_divergence.py                         # NEW
tests/fixtures/sq3_mock_ratings_rater1.csv           # NEW — small fixture for tests
tests/fixtures/sq3_mock_ratings_rater2.csv           # NEW
tests/fixtures/sq3_mock_ratings_rater_round1.csv     # NEW — Branch B fixture
tests/fixtures/sq3_mock_ratings_rater_round2.csv     # NEW
```

## Files to Modify

Only `src/thesis_project/evaluation/__init__.py`, to add re-exports for the
new public functions. No other existing file should be modified by this
phase.

## Files NOT to Touch

- `src/thesis_project/preprocessing/` — all loaders are upstream and frozen.
- `src/thesis_project/scoring/` — all scorers are frozen for this phase.
- `src/thesis_project/lexical/saldo.py` — `SaldoGraph` is read-only here. We
  call its public methods (`lookup`, `path_length`, `wu_palmer`, etc.); we
  do not modify it.
- `src/thesis_project/embeddings/` — embeddings are upstream.
- `bnt_pipeline.py`, `svf_pipeline.py`, `fas_pipeline.py` — none of these
  are touched by SQ3.
- All existing scored-results CSVs (`bnt_scored_results_*.csv`) — read-only
  inputs.
- All notebooks — they will be re-run separately.
- `phase_5_sq3_methodology.md` — this is the governance document. The
  instructions doc cannot deviate from it; if a tension is found, surface
  it as a question, do not edit the methodology doc.

## Requirements

### Stage 1 — Pre-rating

#### 1. Eligibility filter and quartile computation

`src/thesis_project/evaluation/sq3_sampling.py` exposes:

```python
import pandas as pd
from pathlib import Path
from typing import Literal

def load_eligible_pairs(
    bnt_results_path: Path,
) -> pd.DataFrame:
    """Load the BNT scored-results CSV and return only eligible pairs.

    Eligibility:
        is_non_response == False AND is_exact_match == False

    The returned DataFrame is a copy, indexed 0..n-1, with all original
    columns preserved.
    """

def compute_quartile_cutpoints(
    df: pd.DataFrame,
    cosine_col: str = "cosine_sim",
) -> tuple[float, float, float]:
    """Return the (q25, q50, q75) cutpoints of `cosine_col` in `df`.

    Cutpoints are reported as floats; the quartile assignment uses
    pd.qcut with labels=False on the same data.
    """

def assign_quartile(
    df: pd.DataFrame,
    cosine_col: str = "cosine_sim",
    out_col: str = "cosine_quartile",
) -> pd.DataFrame:
    """Add a 0..3 quartile column based on `cosine_col`. Returns a copy."""
```

#### 2. Eligibility probe script

`scripts/sq3_eligibility_probe.py` is a top-level script that summarises
the eligibility-filtered set. CLI:

```
python scripts/sq3_eligibility_probe.py \
    --input data/processed/bnt_scored_results_sbert.csv\
    --output-dir data/processed/sq3/
```

Behaviour:

- Loads the input via `load_eligible_pairs`.
- Reports total row count, eligible row count, exclusion breakdown
  (non-response count, exact-match count).
- Computes quartile cutpoints and prints them to stdout.
- Reports per-quartile counts crossed with diagnosis (4 × 4 contingency
  table) so FB can confirm each quartile has a reasonable mix of
  diagnostic groups.
- Writes a markdown summary to
  `<output-dir>/sq3_eligibility_probe.md` containing all of the above.

The script is idempotent — running it twice on the same input produces
identical output. The script does not draw any sample.

#### 3. Stratified sampling

`sq3_sampling.py` further exposes:

```python
def draw_stratified_sample(
    df: pd.DataFrame,
    quartile_col: str = "cosine_quartile",
    per_quartile: int = 50,
    seed: int = 20260507,
) -> pd.DataFrame:
    """Draw `per_quartile` rows from each quartile, deterministic on seed.

    Returns a DataFrame with the sampled rows in original order, plus a
    `pair_id` column (UUID4 generated from the seed for reproducibility).
    The output is sorted by pair_id, not by quartile, to avoid leaking
    quartile information to the rater through row order.
    """

def make_rater_csv(
    sample: pd.DataFrame,
    rater_seed: int,
    output_path: Path,
) -> None:
    """Write a rater-facing CSV.

    Output columns (in this order):
        pair_id, target, response, rating, category, is_compound, notes

    The pair order is permuted by `rater_seed` for this rater.
    The rating, category, is_compound, notes columns are empty.

    Sensitive columns (participant_id, gender, age, diagnosis, mmse,
    cosine_sim) are explicitly stripped before writing.
    """
```

Test verification: a rater-facing CSV must not contain any of the
sensitive columns. The test asserts the column set is exactly
`{pair_id, target, response, rating, category, is_compound, notes}`.

#### 4. Sample-pairs script

`scripts/sq3_sample_pairs.py` is the Stage-1 entry point. CLI:

```
python scripts/sq3_sample_pairs.py \
    --input data/processed/bnt_scored_results_sbert.csv \
    --target-size 200 \
    --raters FB,collaborator \
    --output-dir data/processed/sq3/
```

Behaviour:

- `--target-size` is 200 or 300. Per-quartile is 50 or 75 respectively.
- `--raters` is a comma-separated list. For each rater, the script
  writes `<output-dir>/sq3_ratings_<rater>.csv` (empty rating columns)
  using the rater-specific permutation seed
  `seed = 20260507 + index_in_list`.
- The script also writes `<output-dir>/sq3_sampled_pairs.csv` — the
  underlying sample with all metadata (pair_id, target, response,
  cosine_sim, cosine_quartile, participant_id, diagnosis, model). This
  is the **non-blinded** master file used in Stage 2 analysis. Treat
  this file as confidential while raters are working.
- Also writes a 20-pair training set
  `<output-dir>/sq3_training_set_<rater>.csv` per rater, drawn with
  seed `20260514` from the *same eligibility-filtered set* but
  excluded from the main 200/300 sample.

The training set is a separate sample with no overlap with the live
rated set. Verify the disjointness in a unit test.

### Stage 2 — Post-rating analyses

Stage 2 scripts are written in this phase but are not run by Claude Code
in this phase. They run later, when rating CSVs are filled in.

#### 5. Reliability analysis

`src/thesis_project/evaluation/sq3_reliability.py` exposes:

```python
from pandas import DataFrame
from dataclasses import dataclass

@dataclass(frozen=True)
class ReliabilityReport:
    n_pairs: int
    rater_ids: list[str]
    branch: Literal["multi", "sole_test_retest"]
    weighted_kappa: dict[tuple[str, str], float]   # pairwise quadratic-weighted κ
    spearman_rho: dict[tuple[str, str], float]     # pairwise ρ on rating
    category_kappa: dict[tuple[str, str], float]   # unweighted κ on category
    compound_flag_kappa: dict[tuple[str, str], float]
    interpretability_flag: Literal["primary", "cautioned", "exploratory"]


def compute_reliability(
    rating_files: list[Path],
    branch: Literal["multi", "sole_test_retest"],
) -> ReliabilityReport:
    """Compute reliability statistics from a list of completed rating CSVs.

    Branch 'multi' expects 2-3 rater files, all over the same pair_ids,
    fully overlapping.
    Branch 'sole_test_retest' expects exactly 2 files representing
    Round 1 and Round 2 ratings from the same rater on a 50-pair subset.

    Interpretability flag (per the methodology doc):
        'primary'      if mean weighted-κ ≥ 0.6
        'cautioned'    if 0.4 ≤ mean weighted-κ < 0.6
        'exploratory'  if mean weighted-κ < 0.4
    """
```

Use `sklearn.metrics.cohen_kappa_score(weights="quadratic")` for the
similarity-rating κ. Use the same function with `weights=None` for the
unweighted category and compound-flag agreement. Use
`scipy.stats.spearmanr` for ρ.

#### 6. Rater-model agreement

`src/thesis_project/evaluation/sq3_agreement.py` exposes:

```python
@dataclass(frozen=True)
class AgreementResult:
    model_name: str
    n: int
    spearman_rho: float
    ci_low: float
    ci_high: float

def rater_model_spearman(
    ratings: DataFrame,
    model_cosines: dict[str, DataFrame],   # model_name -> df with pair_id, cosine_sim
    n_bootstrap: int = 1000,
    seed: int = 20260528,
) -> dict[str, AgreementResult]:
    """For each model, Spearman ρ between rater (or rater-mean) rating
    and model cosine, with bootstrap 95% CI on the (rater, cosine) pairs.

    `ratings` is the multi-rater-merged DataFrame: in Branch A, the rating
    column is the per-pair mean across raters (only computed for pairs
    where all raters provided a rating); in Branch B, the rating column
    is Round 1's ratings (Round 2 is used only for reliability).
    """

def rater_model_per_quartile(
    ratings: DataFrame,
    sampled_pairs: DataFrame,
    model_cosines: dict[str, DataFrame],
    seed: int = 20260528,
) -> DataFrame:
    """Spearman ρ within each cosine quartile per model; long-format result
    with columns: model_name, quartile, n, spearman_rho, ci_low, ci_high.
    """

def rater_model_per_category(
    ratings: DataFrame,
    model_cosines: dict[str, DataFrame],
    seed: int = 20260528,
    min_n: int = 15,
) -> DataFrame:
    """Spearman ρ within each rater-assigned primary category, per model.
    Categories with n < min_n are emitted with NaN ρ and a flag column."""

def rater_saldo_spearman(
    ratings: DataFrame,
    saldo_graph,                               # SaldoGraph instance
    target_response_pairs: DataFrame,          # pair_id, target, response
    seed: int = 20260528,
) -> AgreementResult:
    """Spearman ρ between rater rating and SALDO-derived similarity.

    For each (target, response) pair where both have a SALDO sense:
    - Compute path-length similarity as 1 / (1 + path_length(t, r))
      taking the max across sense-pair combinations (Pakhomov default).
    - Compute Wu-Palmer via SaldoGraph.wu_palmer with the same max-over-
      sense-pairs convention.
    Return both ρ values plus the OOV rate (fraction of pairs where at
    least one of target/response had no SALDO sense).
    """
```

OOV handling: any pair where `SaldoGraph.lookup` returns no senses for
either target or response is excluded from the SALDO-Spearman
computation; the count of excluded pairs is reported as the OOV rate.

#### 7. Divergence catalog

`src/thesis_project/evaluation/sq3_divergence.py` exposes:

```python
def compute_divergence_catalog(
    ratings: DataFrame,
    sampled_pairs: DataFrame,
    primary_model_cosines: DataFrame,           # Swedish SBERT
    saldo_graph,
    threshold_percentile: float = 90.0,
) -> DataFrame:
    """Return a catalog of divergence cases.

    For each pair, compute disagreement = |rating_in_unit_interval - cosine|
    where rating_in_unit_interval = rating / 3.0 (since the scale is 0-3).

    Flag pairs above the `threshold_percentile`-th percentile of disagreement
    as divergence cases.

    Returned columns:
        pair_id, target, response, rater_mean_rating,
        cosine_sim_primary_model, disagreement, rater_category,
        is_compound, saldo_relation_summary

    `saldo_relation_summary` is a short string built from the SALDO graph:
        'mother(t→r)' if response is the primary descriptor of target
        'mother(r→t)' if target is the primary descriptor of response
        'm-sibling'    if target and response share a primary descriptor
        'far'          if both in SALDO but no relation within 3 hops
        'oov(t)' / 'oov(r)' / 'oov(both)'  for OOV cases
    """
```

The catalog is written to `data/processed/sq3/sq3_divergence_catalog.csv`
in disagreement-descending order.

#### 8. Stage-2 entry-point script

`scripts/sq3_analyze.py`:

```
python scripts/sq3_analyze.py \
    --ratings-dir data/processed/sq3/ \
    --branch {multi,sole} \
    --sampled-pairs data/processed/sq3/sq3_sampled_pairs.csv \
    --models bnt_scored_results_*.csv \
    --output-dir data/processed/sq3/reports/
```

Steps performed by the script:

1. Loads all `sq3_ratings_<rater>.csv` (Branch A) or
   `sq3_ratings_<rater>_round{1,2}.csv` (Branch B) from `--ratings-dir`.
2. Validates: all expected pair_ids present, no NaN ratings, ratings in
   {0, 1, 2, 3}, categories from the closed list.
3. Runs reliability analysis. Writes `sq3_reliability_report.md` with
   all reliability statistics and the interpretability flag.
4. Loads all five `bnt_scored_results_*.csv` files; runs per-model
   rater-model Spearman, per-quartile, and per-category. Writes
   `sq3_agreement_report.md` and `sq3_agreement_table.csv`.
5. Loads the SaldoGraph (via the existing `scripts/build_saldo_graph.py`
   pickle if present, otherwise builds it). Runs SALDO scorer comparison.
   Adds output to `sq3_agreement_report.md`.
6. Builds divergence catalog. Writes `sq3_divergence_catalog.csv`.
7. Prints a summary to stdout listing the interpretability flag, the
   primary-model rater-model ρ with CI, and the divergence count.

The script must exit with a clear error message at step 2 if validation
fails. Do not silently coerce out-of-range ratings.

## Test fixtures

`tests/fixtures/sq3_mock_ratings_rater1.csv` and `_rater2.csv` contain
30 rows each with the seven expected columns. Ratings should be in
{0, 1, 2, 3} with deliberate disagreement on ~5 pairs to exercise the
kappa code. Categories from the closed list, with at least one example
of each category.

For Branch B, `sq3_mock_ratings_rater_round1.csv` has 30 rows;
`_round2.csv` has the same 30 pair_ids but with Round-2 ratings on a
disjoint random permutation of the row order.

Tests cover:

- Eligibility filter excludes the right rows.
- Quartile cutpoints match `pd.qcut` on the eligibility-filtered set.
- Sample is deterministic given seed.
- Rater CSVs do not contain sensitive columns.
- Training and live samples are disjoint.
- Reliability statistics match scipy/sklearn reference values on the
  fixture data.
- Per-category rater-model Spearman with n < min_n returns NaN and the
  flag, not a failure.
- SALDO scorer Spearman on a fixture computed via the existing
  `tests/fixtures/saldo_mini.xml`.
- Divergence threshold is the 90th percentile, strictly.
- The Stage-2 script's validation fails clearly on out-of-range ratings.

## What NOT to do

- Do **not** modify any of the BNT/SVF/FAS pipelines or scorers.
- Do **not** modify `SaldoGraph` or any other lexical-module file.
- Do **not** rebuild the SALDO graph from XML in this phase. The Stage-2
  script reads the pickle produced by Phase A. If the pickle is absent,
  raise a clear error pointing the user to `scripts/build_saldo_graph.py`.
- Do **not** implement an alternative reliability metric (Krippendorff's α,
  ICC, Fleiss' κ) and offer it as a flag. The methodology document pins
  weighted Cohen's kappa.
- Do **not** add a category beyond the seven listed in the methodology
  doc. The category list is closed; if a rater wants a new category, that
  is a methodological decision for FB to make outside this code path.
- Do **not** infer rater-rater overlap automatically. Branch A and Branch B
  are explicit `--branch` flags on `sq3_analyze.py`. Auto-detection is
  fragile and the choice has reporting implications.
- Do **not** commit the `sq3_sampled_pairs.csv` master file or any rater
  CSV to the public repo while raters are still working. Add these
  filenames to `.gitignore` for now; commit them only after rating
  completes (the methodology document will cite them).
- Do **not** silently drop pairs where the rater left fields blank.
  Validate strictly and error out with the offending pair_ids.
- Do **not** generate any figures in this phase. Figures for §4.3 are
  built in a later phase as part of the thesis write-up.
- Do **not** present the script outputs to a "clinical interface" — that
  stretch goal is intentionally NOT in this phase's scope.
- Do **not** attempt to run Stage 2 on real rating data. Stage 2 is
  exercised only via the test fixtures in this phase.

## Summary of Changes

| File | Action |
|------|--------|
| `src/thesis_project/evaluation/sq3_sampling.py` | NEW — eligibility filter, quartile cutpoints, stratified sample |
| `src/thesis_project/evaluation/sq3_reliability.py` | NEW — quadratic-weighted κ + Spearman + category κ |
| `src/thesis_project/evaluation/sq3_agreement.py` | NEW — rater-model Spearman, bootstrap CI, SALDO scorer |
| `src/thesis_project/evaluation/sq3_divergence.py` | NEW — divergence catalog |
| `src/thesis_project/evaluation/__init__.py` | UPDATE — re-exports |
| `scripts/sq3_eligibility_probe.py` | NEW — Stage 1 |
| `scripts/sq3_sample_pairs.py` | NEW — Stage 1 |
| `scripts/sq3_analyze.py` | NEW — Stage 2 entry-point |
| `tests/test_sq3_sampling.py` | NEW |
| `tests/test_sq3_reliability.py` | NEW |
| `tests/test_sq3_agreement.py` | NEW |
| `tests/test_sq3_divergence.py` | NEW |
| `tests/fixtures/sq3_mock_ratings_*.csv` | NEW — 4 small fixture CSVs |
| `.gitignore` | UPDATE — add `data/processed/sq3/sq3_sampled_pairs.csv` and `data/processed/sq3/sq3_ratings_*.csv` |

## Acceptance Criteria

A successful run of this phase produces:

1. All existing tests still pass.
2. New tests in all four `test_sq3_*.py` files pass under both
   `pytest -m "not heavy"` and full `pytest`.
3. `python scripts/sq3_eligibility_probe.py --input data/processed/bnt_scored_results_sbert.csv --output-dir /tmp/sq3` runs to completion
   in under 10 seconds and produces `/tmp/sq3/sq3_eligibility_probe.md`
   containing: total rows, eligible rows, exclusion breakdown,
   quartile cutpoints, and the 4×4 quartile-by-diagnosis contingency
   table.
4. `python scripts/sq3_sample_pairs.py --input data/processed/bnt_scored_results_sbert.csv --target-size 200 --raters FB --output-dir
   /tmp/sq3` produces:
   - `/tmp/sq3/sq3_sampled_pairs.csv` (200 rows, all metadata)
   - `/tmp/sq3/sq3_ratings_FB.csv` (200 rows, blinded columns
     only, empty rating/category/is_compound/notes columns)
   - `/tmp/sq3/sq3_training_set_FB.csv` (20 rows, disjoint from
     the sampled pairs)
5. Re-running the same command produces byte-identical output.
6. `python scripts/sq3_sample_pairs.py … --raters FB,collaborator`
   produces both rater CSVs with the same `pair_id` set but different
   row orders.
7. `python scripts/sq3_analyze.py --ratings-dir tests/fixtures
   --branch multi --sampled-pairs tests/fixtures/sq3_mock_sampled.csv
   --models data/processed/bnt_scored_results_sbert.csv --output-dir
   /tmp/sq3/reports` runs end-to-end on the fixtures and produces:
   - `sq3_reliability_report.md`
   - `sq3_agreement_report.md`
   - `sq3_agreement_table.csv`
   - `sq3_divergence_catalog.csv`
8. The Stage-2 script exits with a clear error message if any rating
   is outside `{0, 1, 2, 3}` or any category is outside the closed list.
9. The reliability report contains the interpretability flag
   (`primary` / `cautioned` / `exploratory`) on its first line, derived
   strictly from the methodology-doc thresholds.

## Followups (out of scope this phase)

- **Live execution of Stage 1** by FB. The script is written here;
  FB runs it on his machine, commits the rater CSVs separately
  once decided, and (after rating) provides the filled-in CSVs as
  inputs to Stage 2.
- **Live execution of Stage 2** after rating completes. The methodology
  document's §6 reliability threshold determines the reporting language
  for §4.3.1; FB's responsibility, not Claude Code's.
- **Clinical interface mockup.** Stretch goal per the methodology
  document; deferred to a separate phase if pursued.
- **Thesis write-up of §4.3 and §5.4.** Done by FB after Stage 2
  outputs are reviewed.
- **Multi-model SALDO comparison reporting.** This phase computes SALDO
  Spearman against the rater on the in-vocabulary subset; cross-model
  SALDO comparison tables are a thesis-write-up task, not a phase
  deliverable.

## Git

Branch: `feature/sq3-evaluation` (to be checked out fresh from `main`).

Commits (one logical change per commit; conventional commits format):

1. `feat(evaluation): add sq3_sampling module with eligibility and quartiles`
   — `src/thesis_project/evaluation/sq3_sampling.py`, with eligibility
   filter, quartile cutpoint computation, stratified sample.
2. `test(evaluation): coverage for sq3 sampling`
   — `tests/test_sq3_sampling.py`.
3. `feat(scripts): add sq3_eligibility_probe.py`
   — Stage-1 probe script.
4. `feat(scripts): add sq3_sample_pairs.py with rater CSV writer`
   — Stage-1 sample-draw script and rater-facing CSV blinding.
5. `chore(gitignore): exclude in-progress sq3 rating CSVs`
   — Add `data/processed/sq3/sq3_sampled_pairs.csv` and
   `data/processed/sq3/sq3_ratings_*.csv` patterns.
6. `feat(evaluation): add sq3_reliability with weighted Cohen kappa`
   — `sq3_reliability.py` with multi and sole_test_retest branches.
7. `test(evaluation): coverage for sq3 reliability`
   — `tests/test_sq3_reliability.py` with both branch fixtures.
8. `feat(evaluation): add sq3_agreement with bootstrap and SALDO scorer`
   — `sq3_agreement.py` with rater-model and rater-SALDO functions.
9. `test(evaluation): coverage for sq3 agreement`
   — `tests/test_sq3_agreement.py` against `saldo_mini.xml`.
10. `feat(evaluation): add sq3_divergence catalog`
    — `sq3_divergence.py` plus tests.
11. `feat(scripts): add sq3_analyze entry-point`
    — Stage-2 script with strict validation and clear error messages.
12. `docs(phase): Phase 5 SQ3 instructions and methodology cross-link`
    — Commit this instructions document and ensure
    `phase_5_sq3_methodology.md` is in the docs folder if not already.

PR title: `feat: SQ3 sampling and rater-model agreement infrastructure (Phase 5)`

PR description should reference issue #N, summarise the two execution
stages, list the files produced by Stage 1 on the synthetic data, and
note that Stage 2 is exercised only via fixtures in this phase. Include
a one-line note that any methodology change must edit
`phase_5_sq3_methodology.md` and update this instructions doc, not the
other way around.
