# Phase 4c-FAS-Troyer — methodology notes

This document records the pre-registered methodological decisions
behind the FAS phonemic clustering work and the
`scripts/fas_linz_regression.py` analysis. The scorer, rule module,
and regression script all quote these decisions in their docstrings;
this file is the longer-form companion for the thesis appendix.

## 1. Phonemic similarity rules (Swedish-orthographic, Troyer 1997 adapted)

Two consecutive valid words form a cluster link if **any** of the
following holds:

- **R1 — shared first two letters.** `w1[:2] == w2[:2]`. Direct port
  of Troyer's first rule.
- **R2 — shared rime.** The substring of each word from its last
  vowel onwards is identical and is at least **2 characters long**.
  Vowel set `{a, e, i, o, u, y, å, ä, ö}`. The minimum-length guard
  prevents trivial matches on words ending in a single vowel —
  Swedish has many word-final `-a` infinitives and definite suffixes
  that would otherwise cluster spuriously.
- **R3 — vowel-only single-character substitution.** The words have
  the same length, differ at exactly one position, and both
  substituted characters are Swedish vowels.

Troyer's fourth rule (homonyms) is **dropped** for the Swedish
adaptation. It presupposes a pronunciation dictionary or a
participant verbal cue, neither of which is available for text-only
synthetic data, and Swedish has substantially fewer orthographic
homophones than English. This is a documented limitation, not a
silent omission.

The rules live in `src/thesis_project/scoring/phonemic_rules.py` and
are independently unit-tested.

## 2. Clustering method: chain (Pakhomov 2016)

We use the chain method consistently with the Phase 3 SVF scorer:
walk the response sequence left to right; consecutive words satisfying
the link predicate join the current cluster; otherwise a switch
occurs and a new cluster begins. **Singletons count as clusters of
size 1** (Pakhomov convention; matches the project's SVF stack).

Troyer's all-pairwise "cluster" method (every pair within a candidate
group must be linked) is not implemented. Comparing the two methods
is listed as a follow-up.

## 3. Per-letter computation, then aggregation

Each letter (F, A, S) is clustered independently. The output CSV
exposes both the per-letter triples
(`cluster_count_{f,a,s}`, `mean_cluster_size_{f,a,s}`,
`switch_count_{f,a,s}`) and three aggregates used by the Linz
regression:

- `cluster_count_total` = sum across letters.
- `switch_count_total` = sum across letters.
- `mean_cluster_size` = grand mean over **pooled cluster sizes**
  across all three letters. Each cluster contributes once. This is
  not the mean of per-letter means; pooling weights by cluster, which
  matches the per-letter MCS interpretation.

`mean_word_frequency` (Linz MWF feature) is computed by passing the
pooled F+A+S filtered word list to the same `WordFrequencyProvider`
as Phase 4b-Linz. No FAS-specific lemmatisation.

## 4. Linz feature set: four features (SD intentionally absent)

| Code | Column                | Source                                   |
|------|-----------------------|------------------------------------------|
| WC   | `total_fas_score`     | sum of valid F + A + S (existing pipeline)|
| MCS  | `mean_cluster_size`   | new aggregate (this phase)               |
| NOS  | `switch_count_total`  | new aggregate (this phase)               |
| MWF  | `mean_word_frequency` | new (this phase, MWF over pooled words)  |

The SD (semantic distance) feature from Phase 4b-Linz has no
defensible analog for phonemic fluency and is dropped by
pre-registration. A phonemic-distance substitute (mean Levenshtein
across consecutive words) is listed as a follow-up but is not
implemented here, to keep the feature set tied to the Linz template.

## 5. Feature-selection rule: |Spearman r| > 0.3

Same rule as Phase 4b-Linz. Compute Spearman correlations between
each FAS Linz feature and MMSE; features with |r| < 0.3 are excluded
from the regression. Spearman, not Pearson — MMSE has a known ceiling
effect and a heavily skewed distribution. The rule is fixed a priori;
which features survive is determined by the data. The full
correlation table and the surviving subset are both reported.

## 6. Bootstrap CI on MAE: error-based, 1000 iterations

Same as Phase 4b-Linz. Run 5-fold StratifiedKFold (stratified by
diagnosis — **not** by MMSE, which is continuous), collect held-out
predictions for every participant, then bootstrap-resample the
(predicted, actual) pairs 1000 times with replacement; report mean
MAE plus 2.5/97.5 percentiles.

## 7. Repetitions: counted, but excluded from clustering

The pipeline already counts `repetitions_count`. Before passing
words to `score_fas`, each letter's word list is deduplicated while
preserving order (`list(dict.fromkeys(words))`). Cluster analysis is
performed on first-occurrence words only; repetitions remain counted
in the count-based summary.

## 8. Limitations explicitly accepted

- **Orthographic, not phonetic.** No pronunciation dictionary is
  consulted. Compounds, silent letters, and pronunciation-only
  homophones are not modelled. Replacement with a Swedish G2P
  approach is a follow-up.
- **Synthetic data only.** Performance numbers are not directly
  transferable to real H70 data.
- **No lemmatisation in MWF.** Surface-form lookups only, matching
  Phase 4b-Linz.
- **Single random seed.** Reproducible but a multi-seed sensitivity
  check is deferred.

## Output artefacts (one representative run)

A successful run of `python scripts/fas_linz_regression.py
--random-seed 42` produces:

- `data/processed/linz_fas/linz_fas_correlation_table.csv`
- `data/processed/linz_fas/linz_fas_regression_mae.csv`
- `data/processed/linz_fas/linz_fas_confusion_matrix.csv`
- `data/processed/linz_fas/linz_fas_kappa.csv`
- `figures/linz_fas/fas_predicted_vs_actual.png`

Cohen's κ is reported regardless of value — including κ < 0.2 — to
avoid suppressing negative results.
