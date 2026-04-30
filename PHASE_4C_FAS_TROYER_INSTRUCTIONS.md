# Phase 4c-FAS-Troyer: Phonemic Clustering, FAS MMSE Plumbing, and Linz-Style Regression

## Context

This phase has three parts that build on each other:

1. **FAS phonemic clustering (Troyer 1997, Swedish-adapted).** Implement a
   `fas_scorer` that walks each per-letter response sequence and produces
   chain-method clusters following Pakhomov's convention (singletons count as
   size 1, matching Phase 3 SVF). The cluster-link predicate is the union of
   three Troyer (1997) rules adapted to Swedish orthography: shared first two
   letters, rhyme (shared rime), and vowel-only single-character difference.
   Troyer's fourth rule (homonyms) is dropped — see Pre-Registered Decisions.

2. **FAS MMSE plumbing.** Mirror the Phase 4b-Linz work for FAS. The shared
   data loader already extracts MMSE for all three tests (Phase 4a). This
   phase updates `fas_pipeline.py` to merge `user_meta["mmse"]` into the
   per-participant output CSV.

3. **Linz-style regression on FAS features against MMSE.** Mirror Phase 4b-Linz
   structure for FAS. Predict MMSE from four FAS-derived features (WC, MCS,
   NOS, MWF) using Support Vector Regression and Ridge Regression with 5-fold
   cross-validation. Report MAE with 95% bootstrap CI. Produce the same four
   end-of-day artefacts: a Spearman correlation table, a regression MAE table,
   a confusion matrix at MMSE = 24 with Cohen's κ, and a predicted-vs-actual
   scatter coloured by diagnosis.

This phase is scoped to **FAS only**. BNT and SVF scorers, embedders, and the
SVF Linz pipeline are upstream and not modified.

GitHub issue: #41 (to be created).

## Critical Methodological Decisions (Pre-Registered)

Five decisions are pre-registered here so they cannot become post-hoc
rationalisations later. Document them in the FAS scorer's module docstring
and in the regression script's module docstring exactly as stated.

**Phonemic similarity rules: shared-first-two-letters ∪ shared-rime ∪
vowel-only-substitution.** Two words form a cluster link if any of the
following holds:

- *R1 (shared first two letters)*: `w1[:2] == w2[:2]`. Direct port of
  Troyer's first rule.
- *R2 (shared rime)*: the substring of each word from its last vowel onwards
  is identical and is at least 2 characters long. Vowel set:
  `{a, e, i, o, u, y, å, ä, ö}`. The minimum-length guard prevents trivial
  matches on words ending in a single vowel (Swedish has many word-final
  `-a` infinitives and definite suffixes).
- *R3 (vowel-only single-char difference)*: Levenshtein edit distance is
  exactly 1, the operation is a substitution (not insertion/deletion), and
  both substituted characters belong to the Swedish vowel set above.

The fourth Troyer rule (homonyms) is **dropped** for the Swedish adaptation.
Troyer's homonym rule presupposes a pronunciation dictionary or a participant
verbal cue ("some and the other sum") which is not available for text-only
synthetic data. Swedish has substantially fewer orthographic homophones than
English. Document as a limitation in the thesis methodology section.

**Clustering method: chain (Pakhomov 2016), not Troyer's all-pairwise
"cluster" method.** Two consecutive valid words form a cluster link if any
of R1–R3 holds. Singletons count as clusters of size 1. This matches the
Phase 3 SVF convention and keeps the SVF/FAS architectural stack consistent.

**Per-letter computation, then aggregation.** Each letter (F, A, S) is
clustered independently. The output CSV exposes both per-letter metrics
(`cluster_count_f/a/s`, `mean_cluster_size_f/a/s`, `switch_count_f/a/s`) and
three aggregates used by the Linz regression: `cluster_count_total` (sum
across letters), `switch_count_total` (sum across letters), and
`mean_cluster_size` (grand mean across all clusters from all three letters,
pooled).

**Feature selection rule: |r| > 0.3 with MMSE.** Mirrors Linz et al. (2017)
and Phase 4b-Linz. Compute Spearman correlations between every FAS Linz
feature (WC, MCS, NOS, MWF) and MMSE; features with |r| < 0.3 are excluded
from the regression. The rule is fixed a priori; which features actually
survive is determined by the data. Report the full correlation table and the
surviving subset. The SD feature from Phase 4b-Linz (semantic distance) has
no defensible analog for phonemic fluency and is dropped — the FAS Linz
feature set is four features, not five.

**Bootstrap CI on MAE: error-based, 1000 iterations.** Run 5-fold CV, collect
held-out predictions for all 100 participants. Bootstrap-resample the
resulting (predicted, actual) pairs 1000 times with replacement; compute MAE
per resample; report 2.5th and 97.5th percentiles as the 95% CI. Identical
to Phase 4b-Linz.

## Files to Modify

```
src/thesis_project/preprocessing/fas_pipeline.py
configs/_default_configs.yaml          # add fas_clustering section
tests/test_fas_pipeline.py             # adjust schema assertions for new columns
```

(Filenames may differ slightly in the repo. The architectural points are
fixed: the FAS pipeline propagates MMSE and the new clustering metrics; the
config gains a `fas_clustering` section pinning the rule set; the regression
script is a new top-level script.)

## Files to Create

```
src/thesis_project/scoring/phonemic_rules.py    # NEW — three rule predicates + helpers
src/thesis_project/scoring/fas_scorer.py        # NEW — chain clustering for F/A/S
scripts/fas_linz_regression.py                  # NEW — Linz-style regression analysis for FAS
tests/test_phonemic_rules.py                    # NEW
tests/test_fas_scorer.py                        # NEW
tests/test_fas_linz_regression.py               # NEW
```

The phonemic rules live in their own module so they're independently
unit-testable. The FAS scorer imports them.

## Files NOT to Touch

- `bnt_pipeline.py`, `bnt_scoring.py`, `bnt_pipeline_legacy.py` — out of scope.
- `svf_pipeline.py`, `svf_scoring.py` — Phase 3/4b artefacts; frozen here.
- `src/thesis_project/preprocessing/data_loader.py` — Phase 4a loader contract
  is fixed.
- `src/thesis_project/embeddings/encoder.py` — embeddings not used in FAS
  clustering. Phonemic rules are orthographic.
- `src/thesis_project/lexical/word_frequency.py` — reuse the Phase 4b-Linz
  `WordFrequencyProvider` verbatim. Do not modify it; just import.
- `scripts/svf_linz_regression.py` — Phase 4b-Linz artefact. The FAS
  regression script is a new file, not an extension of the SVF one.
- `src/thesis_project/lexical/saldo*.py` (if present) — separate research
  thread.
- `scripts/run_model_comparison.py` — BNT-only sweep.
- All notebooks. They will be re-run separately.

## Requirements

### 1. Phonemic rules module

Create `src/thesis_project/scoring/phonemic_rules.py`. Module docstring must
state the Pre-Registered rule set verbatim and cite Troyer et al. (1997).

#### 1a. Public API

```python
SWEDISH_VOWELS: frozenset[str] = frozenset("aeiouyåäö")

def shared_first_two_letters(w1: str, w2: str) -> bool:
    """R1: True iff w1 and w2 share their first two characters.

    Both words are assumed lowercased and stripped. Returns False if either
    word has fewer than 2 characters.
    """

def shared_rime(w1: str, w2: str, min_rime_length: int = 2) -> bool:
    """R2: True iff the rime of w1 and w2 are identical and at least
    min_rime_length characters long.

    The rime is the substring from the last vowel (in SWEDISH_VOWELS)
    to the end of the word. If a word has no vowels, returns False.

    The min_rime_length guard avoids trivial matches on words ending in
    a single vowel (e.g., '-a' infinitives, definite suffixes).
    """

def vowel_only_substitution(w1: str, w2: str) -> bool:
    """R3: True iff w1 and w2 differ by exactly one substitution and both
    substituted characters are Swedish vowels.

    Insertions and deletions do not count. Words of different length always
    return False.
    """

def linked(w1: str, w2: str) -> bool:
    """True iff any of R1, R2, R3 holds. The cluster-link predicate."""
```

#### 1b. Implementation notes

- All comparisons are case-sensitive on lowercased input. The FAS pipeline
  is responsible for lowercasing before calling these functions.
- `shared_rime`: walk the word from the right, find the index of the last
  character in `SWEDISH_VOWELS`, return `w[idx:]`. The two rimes must be
  string-equal AND `len(rime) >= min_rime_length`.
- `vowel_only_substitution`: only handle the equal-length case. Iterate
  position by position; count differences; if exactly one difference and
  both characters at that position are in `SWEDISH_VOWELS`, return True.
  Do NOT call out to a Levenshtein library — the equal-length-one-substitution
  case is trivial and avoids a dependency.
- Edge cases: empty string, single-character word, identical words. Identical
  words trivially satisfy R1 (and R2). The scorer is responsible for filtering
  repetitions before clustering — see §3 — so the rule predicates should
  *not* try to deduplicate.

### 2. FAS scorer

Create `src/thesis_project/scoring/fas_scorer.py`.

#### 2a. Public API

```python
def cluster_letter(words: list[str]) -> dict:
    """Chain-method clustering for one letter's response sequence.

    Args:
        words: ordered list of valid words (already lowercased, with
            proper-noun markers stripped, repetitions removed).

    Returns: dict with keys
        clusters: list[list[str]]
        cluster_sizes: list[int]
        cluster_count: int
        switch_count: int    # = max(0, cluster_count - 1)
        mean_cluster_size: float    # NaN if cluster_count == 0
        max_cluster_size: int

    Edge cases:
        - words == []: cluster_count=0, switch_count=0, mean=NaN, max=0
        - len(words) == 1: cluster_count=1, switch_count=0, mean=1.0, max=1
    """

def score_fas(
    f_words: list[str],
    a_words: list[str],
    s_words: list[str],
    word_freq_provider: WordFrequencyProvider | None = None,
) -> dict:
    """Compute per-letter and aggregated FAS clustering metrics.

    Inputs are already filtered (proper nouns stripped, repetitions removed)
    by the FAS pipeline. This function does NOT re-filter.

    Returns a flat dict with these keys:

        # Per-letter (3 × 3 = 9 keys)
        cluster_count_f, mean_cluster_size_f, switch_count_f
        cluster_count_a, mean_cluster_size_a, switch_count_a
        cluster_count_s, mean_cluster_size_s, switch_count_s

        # Aggregates (Linz feature inputs)
        cluster_count_total: int          # sum across letters
        switch_count_total: int           # sum across letters
        mean_cluster_size: float          # grand mean across pooled clusters

        # MWF (computed here so the pipeline does not need to know about it)
        mean_word_frequency: float        # NaN if word_freq_provider is None
                                          # or if F+A+S is empty
    """
```

#### 2b. Aggregation rules

- `cluster_count_total = cluster_count_f + cluster_count_a + cluster_count_s`
- `switch_count_total = switch_count_f + switch_count_a + switch_count_s`
- `mean_cluster_size`: pool all cluster sizes from all three letters into
  one list, then take the arithmetic mean. NaN if the pooled list is empty.
  This is *not* the mean of per-letter means — pooling is the chosen
  convention because it weights by cluster (each cluster contributes
  equally), which matches the per-letter MCS interpretation.
- `mean_word_frequency`: pool all words from F, A, S (post-filter), call
  `word_freq_provider.mean_word_frequency(pooled_words)`. NaN if the pooled
  list is empty or `word_freq_provider is None`.

#### 2c. Reusing existing chain logic

The chain-clustering logic is conceptually identical to Phase 3's SVF
`detect_clusters`, but the link predicate is different (orthographic rules
instead of cosine similarity threshold). Do **not** import or extend
`detect_clusters` — write a small dedicated walk in `cluster_letter` that
takes the link predicate as either implicit (via `linked`) or as a callable
parameter. The two functions live in different modules and serve different
semantic domains; cross-importing creates coupling we don't want.

### 3. FAS pipeline integration

Modify `fas_pipeline.py`:

1. Propagate `user_meta["mmse"]` into the per-participant output CSV
   (mirror the Phase 4a BNT pattern and Phase 4b-Linz SVF pattern).
2. After the existing per-participant filtering step (which strips
   proper-noun markers and computes `valid_f/a/s`), call `score_fas` on
   the filtered word lists.
3. Repetitions: the existing pipeline already counts `repetitions_count`.
   Before passing to `score_fas`, deduplicate each letter's word list while
   preserving order (e.g., `list(dict.fromkeys(words))`). Document this
   in the pipeline as: "Cluster analysis is performed on first-occurrence
   words only; repetitions are excluded from clustering but remain
   counted in `repetitions_count`."
4. Construct the `WordFrequencyProvider` once at pipeline startup
   (`source` from config; default `"wordfreq"`) and pass into each
   `score_fas` call.
5. Merge `score_fas` output into the per-participant row.

The output CSV should have the column order:

```
participant_id, diagnosis, age, gender, mmse,
total_f, total_a, total_s,
valid_f, valid_a, valid_s,
total_fas_score, proper_nouns_count, repetitions_count, letter_asymmetry,
cluster_count_f, mean_cluster_size_f, switch_count_f,
cluster_count_a, mean_cluster_size_a, switch_count_a,
cluster_count_s, mean_cluster_size_s, switch_count_s,
cluster_count_total, switch_count_total, mean_cluster_size,
mean_word_frequency
```

If the FAS input file does not contain MMSE (legacy v1/v2 data), the column
is preserved with NaN values. Emit an INFO-level warning at pipeline startup
if all MMSE values are NaN. Mirror Phase 4a BNT behaviour exactly.

### 4. Config additions

Update `configs/_default_configs.yaml` to add a `fas_clustering` section:

```yaml
fas_clustering:
  rules: ["shared_first_two_letters", "shared_rime", "vowel_only_substitution"]
  rime_min_length: 2
  homonyms: false   # documented limitation; see Phase 4c-FAS-Troyer instructions
  cluster_size_convention: "pakhomov"   # singletons = size 1
```

And a `fas_frequency` section mirroring SVF's:

```yaml
fas_frequency:
  source: "wordfreq"   # or "sucx" if data/external/sucx_frequencies.csv exists
```

The config is documentary as much as functional — it locks the rule set in
the repo so any future re-run uses the same configuration unless explicitly
changed.

### 5. Linz-style regression for FAS

Create `scripts/fas_linz_regression.py`. Structure mirrors
`scripts/svf_linz_regression.py` (Phase 4b-Linz) but operates on the FAS
output CSV and uses four features instead of five.

#### 5a. Module docstring

State the pre-registered methodological decisions verbatim. Cite Linz et al.
(2017) and Phase 4b-Linz as the structural template. Note that SD is dropped
for FAS because phonemic fluency has no defensible semantic-distance analog.

#### 5b. CLI

```
python scripts/fas_linz_regression.py \
    --input data/processed/fas_results_with_mmse.csv \
    --output-dir data/processed/linz_fas/ \
    --figure-dir figures/linz_fas/ \
    --random-seed 42
```

Add `--dry-run` that prints the planned analysis without running it
(4 features, 2 models, 5-fold CV, 1000 bootstrap iterations, 4 output
artefacts).

#### 5c. Feature mapping (FAS Linz feature set)

| Linz code | FAS feature column | Source |
|-----------|---------------------|--------|
| WC | `total_fas_score` | existing pipeline (sum of valid F+A+S) |
| MCS | `mean_cluster_size` | new aggregate from §2b |
| NOS | `switch_count_total` | new aggregate from §2b |
| MWF | `mean_word_frequency` | new from §2b |

SD is intentionally absent. Do not invent a phonemic-distance substitute;
that is a documented out-of-scope extension (§ Followups).

#### 5d. Pipeline structure

Mirror `scripts/svf_linz_regression.py`:

1. Load FAS CSV, drop rows with NaN MMSE, log dropped count.
2. Compute Spearman correlations of each of the 4 features against MMSE.
   Save `linz_fas_correlation_table.csv` with columns
   `feature, spearman_r, p_value, n, passes_threshold` (passes_threshold
   is True iff `|r| > 0.3`).
3. Filter to surviving features. If the surviving set is empty, exit with
   a clear error message (NOT a silent regression on an empty matrix).
4. For each model in {SVR with RBF, Ridge}: run 5-fold CV, collect
   held-out predictions. Bootstrap-resample (preds, actuals) pairs 1000
   times; compute MAE per resample; report median + 2.5/97.5 percentile CI.
5. Save `linz_fas_regression_mae.csv` (2 rows, model × MAE/CI).
6. For each model: build a 2×2 confusion matrix at MMSE = 24
   (≤24 = "impaired", >24 = "normal"). Compute Cohen's κ.
7. Save `linz_fas_confusion_matrix.csv` (one 2×2 matrix per model, stacked)
   and `linz_fas_kappa.csv` (2 rows).
8. Render `fas_predicted_vs_actual.png` (300 DPI, two panels SVR/Ridge,
   diagnosis-coloured points, diagonal reference, MMSE = 24 cutoff line).

Reuse the helper functions from `scripts/svf_linz_regression.py` if they
were factored out into a shared module during Phase 4b-Linz. If they
weren't, copy them — do not refactor as part of this phase. Refactoring
shared utility code into a `linz_utils.py` module is a natural follow-up
but adds scope risk to today's work.

### 6. Tests

#### 6a. `tests/test_phonemic_rules.py`

```python
import pytest
from thesis_project.scoring.phonemic_rules import (
    shared_first_two_letters,
    shared_rime,
    vowel_only_substitution,
    linked,
)

# R1
def test_r1_basic():
    assert shared_first_two_letters("frukt", "frid")
    assert shared_first_two_letters("art", "arm")
    assert not shared_first_two_letters("frukt", "fisk")

def test_r1_short_words():
    assert not shared_first_two_letters("a", "an")
    assert not shared_first_two_letters("", "")

# R2
def test_r2_basic():
    assert shared_rime("sand", "stand")        # "and"
    assert shared_rime("fågel", "tagel")       # "el"
    assert not shared_rime("sand", "salt")     # "and" vs "alt"

def test_r2_min_length_guard():
    # words ending in single vowel — rime length 1, should NOT match
    assert not shared_rime("saga", "vaga")     # rime "a", length 1
    assert not shared_rime("krypa", "skapa")   # rime "a", length 1

def test_r2_no_vowels():
    # synthetic edge case — no vowels in word
    assert not shared_rime("xyz", "xyz")

# R3
def test_r3_basic():
    assert vowel_only_substitution("fil", "fal")    # i → a
    assert vowel_only_substitution("fas", "fos")    # a → o
    assert not vowel_only_substitution("fil", "fjäl")   # different lengths

def test_r3_consonant_substitution_rejected():
    assert not vowel_only_substitution("fil", "fis")    # l → s, both consonants
    assert not vowel_only_substitution("hund", "rund")  # h → r, both consonants

def test_r3_mixed_substitution_rejected():
    # one vowel, one consonant — Troyer's rule requires both to be vowels
    # (interpreted as "differ only by a vowel sound")
    pass    # this case rarely arises with len-1 substitutions, but document the choice

# linked
def test_linked_any_rule():
    assert linked("frukt", "frid")        # R1
    assert linked("sand", "stand")        # R2
    assert linked("fil", "fal")           # R3
    assert not linked("hund", "elefant")  # nothing
```

#### 6b. `tests/test_fas_scorer.py`

```python
import math
import pytest
from thesis_project.scoring.fas_scorer import cluster_letter, score_fas
from thesis_project.lexical.word_frequency import WordFrequencyProvider

def test_cluster_letter_empty():
    r = cluster_letter([])
    assert r["cluster_count"] == 0
    assert r["switch_count"] == 0
    assert math.isnan(r["mean_cluster_size"])
    assert r["max_cluster_size"] == 0

def test_cluster_letter_single():
    r = cluster_letter(["fil"])
    assert r["cluster_count"] == 1
    assert r["switch_count"] == 0
    assert r["mean_cluster_size"] == 1.0
    assert r["max_cluster_size"] == 1

def test_cluster_letter_all_linked():
    # frukt-frid (R1), frid-frost (R1) — one big cluster
    r = cluster_letter(["frukt", "frid", "frost"])
    assert r["cluster_count"] == 1
    assert r["mean_cluster_size"] == 3.0

def test_cluster_letter_no_links():
    # mutually unrelated F-words
    r = cluster_letter(["fil", "frukt", "färg"])
    # fil-frukt: R1? "fi" vs "fr" no. R2? "il" vs "ukt" no. R3? len diff. → no
    # frukt-färg: R1? "fr" vs "fä" no. R2? "ukt" vs "ärg" no. R3? len diff. → no
    assert r["cluster_count"] == 3
    assert r["switch_count"] == 2
    assert r["mean_cluster_size"] == 1.0

def test_cluster_letter_mixed():
    # frukt-frid linked (R1), frid-färg unlinked, färg singleton
    r = cluster_letter(["frukt", "frid", "färg"])
    assert r["cluster_count"] == 2
    assert r["cluster_sizes"] == [2, 1]
    assert r["switch_count"] == 1
    assert r["mean_cluster_size"] == 1.5

def test_score_fas_aggregates():
    f = ["frukt", "frid", "färg"]    # 2 clusters: [frukt, frid], [färg]
    a = ["arm", "art", "arv"]        # all R1-linked: 1 cluster of 3
    s = ["sand", "stand"]            # R2-linked: 1 cluster of 2
    r = score_fas(f, a, s, word_freq_provider=None)
    assert r["cluster_count_total"] == 2 + 1 + 1
    assert r["switch_count_total"] == 1 + 0 + 0
    # pooled cluster sizes: [2, 1, 3, 2] → mean 2.0
    assert r["mean_cluster_size"] == pytest.approx(2.0)
    assert math.isnan(r["mean_word_frequency"])

def test_score_fas_with_frequency():
    provider = WordFrequencyProvider(source="wordfreq")
    r = score_fas(["fil"], ["arm"], ["sand"], word_freq_provider=provider)
    # exact value depends on wordfreq; just check it's a finite float
    assert isinstance(r["mean_word_frequency"], float)
    assert not math.isnan(r["mean_word_frequency"])
```

#### 6c. `tests/test_fas_pipeline.py` additions

Adjust the schema-shape assertion to expect the new columns. Add a test that
runs the pipeline end-to-end on a small synthetic input and verifies:

- `mmse` column present
- All nine per-letter cluster columns present
- All three aggregate columns present
- `mean_word_frequency` column present
- Repetition deduplication: a participant whose F-list contains
  `["fil", "fil", "frukt"]` produces `cluster_count_f` based on
  `["fil", "frukt"]`, with `repetitions_count` reflecting the duplicate.

#### 6d. `tests/test_fas_linz_regression.py`

Smoke test + dry-run test, mirroring `tests/test_svf_linz_regression.py`.
Also: a test that asserts a clear error is raised when no features pass
the |r| > 0.3 filter (build a fixture CSV where MMSE is uncorrelated with
all four features).

## Acceptance Criteria

1. All existing tests pass.
2. New tests in all four test files pass under both `pytest -m "not heavy"`
   (fast CI) and `pytest` (full).
3. `python fas_pipeline.py --input data/raw/sweFASsyntheticData_v3.xlsx`
   produces `data/processed/fas_results_with_mmse.csv` containing the
   columns listed in §3 in the order shown.
4. `python scripts/fas_linz_regression.py --dry-run` prints the planned
   analysis: 4 features, 2 models, 5-fold CV, 1000 bootstrap iterations,
   4 output artefacts.
5. `python scripts/fas_linz_regression.py --random-seed 42` runs to
   completion in under 2 minutes on CPU and produces:
   - `data/processed/linz_fas/linz_fas_correlation_table.csv` (4 rows,
     columns `feature, spearman_r, p_value, n, passes_threshold`)
   - `data/processed/linz_fas/linz_fas_regression_mae.csv` (2 rows, one
     per model)
   - `data/processed/linz_fas/linz_fas_confusion_matrix.csv` (one 2×2
     matrix per model, stacked)
   - `data/processed/linz_fas/linz_fas_kappa.csv` (2 rows, one per model)
   - `figures/linz_fas/fas_predicted_vs_actual.png` (300 DPI, two panels,
     diagnosis-coloured points, diagonal reference, MMSE = 24 cutoff line)
6. The regression CSV contains MAE values within sane bounds: MAE ≤ 8 for
   both models on this synthetic data. MAE > 8 indicates a bug, not poor
   performance. The trivial "predict the mean" baseline is around 5–7.
7. Cohen's κ is reported even when poor (κ < 0.2). Do not suppress
   negative results.
8. The script exits with a clear error message if no features pass the
   |r| > 0.3 filter, rather than running a regression on an empty
   feature matrix.
9. Kruskal-Wallis test across the four diagnostic groups (HC, MCI,
   non-AD, AD) is run on `cluster_count_total`, `mean_cluster_size`,
   and `switch_count_total` at the end of `fas_pipeline.py`. The H
   statistic and p-value for each metric are printed to stdout. This is
   a sanity check, not pass/fail — the synthetic FAS data has a known
   two-batch artefact in repetitions, and the diagnostic gradient on
   clustering metrics may or may not survive depending on how the
   synthetic generator handled phonemic structure. Whatever the result,
   it is logged and proceeds.

## What NOT to Do

- Do NOT add a phonetic dictionary dependency (CMUdict, espeak, etc.).
  The orthographic approximations in §1 are explicitly the chosen approach.
- Do NOT implement Troyer's all-pairwise "cluster" method. Chain only.
- Do NOT compute embedding-based similarity for FAS. Phonemic clustering
  is orthographic by construction.
- Do NOT extend `WordFrequencyProvider`. Reuse it.
- Do NOT refactor `scripts/svf_linz_regression.py` into a shared utility
  module. Copy whatever is needed; refactoring is a separate follow-up.
- Do NOT add a phonemic-distance feature to substitute for SD. The Linz
  feature set for FAS is four features, by pre-registration.
- Do NOT modify the BNT or SVF pipelines. This phase is FAS-only.
- Do NOT touch notebooks.
- Do NOT compute homonym links, even via a small hand-curated list. The
  rule is dropped, and this is documented as a methodological limitation.

## Followups (out of scope this phase)

- **Phonetic-dictionary-based clustering.** Replace orthographic R2/R3
  with phonetic similarity computed via a Swedish G2P system. Ryan et
  al. (2013) implements this for English (CMUdict + edit distance);
  the Swedish equivalent would use a Swedish lexicon or G2P model.
  Worth flagging in the thesis discussion as the natural extension.
- **Phonemic distance feature (SD substitute).** Mean Levenshtein
  distance between consecutive valid words within a letter, averaged
  across F/A/S. Adds a 5th feature to the Linz regression. Defensible
  but lacks literature precedent.
- **Cluster-method (all-pairwise) comparison.** Run Troyer's original
  cluster method alongside the chain method and report agreement.
- **Subword-segmentation for compound responses.** Some FAS responses
  may be compounds (e.g., `fågelbur`); decomposing them affects R1/R2.
  Out of scope; acknowledge in limitations.
- **Refactor shared Linz-regression utilities.** Once SVF and FAS Linz
  scripts both exist, extract common code (correlation table, bootstrap
  CI, confusion matrix at threshold, scatter rendering) into
  `src/thesis_project/evaluation/linz_utils.py`.
- **Multi-model FAS sweep.** Not applicable — FAS clustering is
  embedding-free. The model-comparison machinery from BNT/SVF does not
  port to FAS.

## Git

Branch: `feature/fas-troyer-regression` (create from `main`).

Commits (one logical change per commit; conventional commits format):

1. `feat(scoring): add phonemic rules module (R1/R2/R3) for Swedish FAS`
   — `phonemic_rules.py` with `shared_first_two_letters`, `shared_rime`,
   `vowel_only_substitution`, `linked`.
2. `test(scoring): coverage for phonemic rules including Swedish edge cases`
   — `tests/test_phonemic_rules.py`.
3. `feat(scoring): add chain-method FAS scorer with per-letter and aggregated metrics`
   — `fas_scorer.py` with `cluster_letter` and `score_fas`. Includes MWF
   computation by injection.
4. `test(scoring): coverage for FAS scorer aggregation and edge cases`
   — `tests/test_fas_scorer.py`.
5. `feat(pipelines): propagate MMSE and clustering metrics to FAS output CSV`
   — `fas_pipeline.py` updated to merge `user_meta["mmse"]`, deduplicate
   per-letter words, call `score_fas`, and emit the expanded schema.
6. `feat(config): add fas_clustering and fas_frequency sections`
   — `_default_configs.yaml` updated. Comment cites pre-registered rules.
7. `test(pipelines): coverage for FAS pipeline schema and dedup behaviour`
   — `tests/test_fas_pipeline.py` adjustments and additions.
8. `feat(scripts): add fas_linz_regression.py for SQ4 analysis on FAS`
   — new script with SVR + Ridge models, Spearman correlation table,
   bootstrap CI, confusion matrix at MMSE=24, scatter plot.
9. `test(scripts): smoke and dry-run tests for FAS Linz regression`
   — `tests/test_fas_linz_regression.py`.
10. `docs(phase): Phase 4c-FAS-Troyer instructions and methodology notes`
    — commit this instructions document and a brief methodology note in
    the docs folder describing the rule set, dropped-homonym rationale,
    and the four-feature Linz set.

PR title: `feat: FAS phonemic clustering + MMSE plumbing + Linz-style regression (Phase 4c-FAS-Troyer)`

PR description should reference issue #41, summarise the four
deliverables, list the three Kruskal-Wallis results from the pipeline
run, and link to the four output CSVs and one figure produced by a
representative run.

## Summary of Changes

| File | Action |
|------|--------|
| `src/thesis_project/scoring/phonemic_rules.py` | Create |
| `src/thesis_project/scoring/fas_scorer.py` | Create |
| `src/thesis_project/preprocessing/fas_pipeline.py` | Modify (MMSE plumbing, scorer integration, dedup) |
| `configs/_default_configs.yaml` | Modify (add `fas_clustering` and `fas_frequency` sections) |
| `scripts/fas_linz_regression.py` | Create |
| `tests/test_phonemic_rules.py` | Create |
| `tests/test_fas_scorer.py` | Create |
| `tests/test_fas_pipeline.py` | Modify (schema assertions, dedup test) |
| `tests/test_fas_linz_regression.py` | Create |
| `PHASE_4C_FAS_TROYER_INSTRUCTIONS.md` | Create (this document) |

Everything else: untouched.
