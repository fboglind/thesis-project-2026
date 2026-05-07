# Phase 4b — SVF Linz regression: methodology notes

This document records the three pre-registered methodological decisions
behind `scripts/svf_linz_regression.py` so they cannot become post-hoc
rationalisations later. The script's module docstring quotes these
verbatim; this file is the longer-form companion.

## 1. SVF cluster threshold pinned at 0.45

The Phase 3 threshold sweep (Kruskal–Wallis between-group testing on
`mean_cluster_size` and `switch_count` at thresholds in
[0.30, 0.70] step 0.025) identified **0.45** as the unique value at
which both metrics cross *p* < 0.001 simultaneously. This is the
Troyer-style balance point at which clustering and switching contribute
roughly equal diagnostic signal.

The Linz regression code does **not** re-calibrate this value: it reads
`svf.cluster_threshold` from `configs/_default_configs.yaml` (pinned at
0.45) and consumes Phase 3's existing scorer outputs. Any future
deviation from 0.45 should require its own threshold-sweep evidence,
not a downstream tweak.

## 2. Feature selection rule: |Spearman r| > 0.3 with MMSE

Linz et al. (2017) excluded MCS from their regression on the basis of
|r| < 0.3 with MMSE in their Table III. We apply the same rule but to
*our* correlation table — the rule is fixed a priori; which features
actually survive is determined by the data.

Spearman, not Pearson. MMSE has a known ceiling effect (a long flat
tail at 30) and a heavily skewed distribution; Pearson assumes
roughly-bivariate-normal data and is misleading here. (Linz's own
analysis is similarly fragile — worth flagging as a limitation in the
thesis.)

The script reports both the full correlation table and the surviving
feature subset.

## 3. Bootstrap CI on MAE: error-based, 1000 iterations

The script runs 5-fold StratifiedKFold (stratified by diagnosis to keep
folds balanced; **not** by MMSE, which is continuous), collecting
held-out predictions for every participant exactly once. The 100
(predicted, actual) pairs are then bootstrap-resampled 1000 times with
replacement; MAE is computed per resample, and the 2.5th / 97.5th
percentiles form the 95% CI.

This is more rigorous than Linz's CV-fold-spread CI, which is
small-N noisy at five folds. The error-based approach also matches
modern recommendations (e.g. Efron & Tibshirani 1993) for confidence
intervals on small evaluation sets.

## Limitations explicitly accepted in this phase

- **No lemmatisation in MWF.** Both the wordfreq and SUCX backends
  operate on lowercased surface forms. Synthetic SVF responses are
  already lemma-like single nouns, so lemmatisation gives marginal
  gain at the cost of an extra dependency and failure mode.
- **Synthetic data only.** The 100-participant synthetic dataset has
  fewer ceiling effects than real H70 data; absolute MAE numbers are
  not directly transferable.
- **Single random seed.** The script is reproducible — same seed →
  byte-identical output CSVs (modulo floating-point determinism) — but
  a multi-seed sensitivity check is deferred.

## Output artefacts

A successful run produces:

- `data/processed/linz/linz_correlation_table.csv` — Spearman r, p,
  n, and the |r|>0.3 pass flag for each Linz feature.
- `data/processed/linz/linz_regression_mae.csv` — per-model MAE with
  bootstrap CI bounds and the surviving feature list.
- `data/processed/linz/linz_confusion_matrix.csv` — 2×2 confusion at
  MMSE = 24 (clinical convention; not data-driven), per model.
- `data/processed/linz/linz_kappa.csv` — Cohen's κ per model.
- `figures/linz/svf_predicted_vs_actual.png` — two-panel scatter,
  diagnosis-coloured, diagonal reference.

Cohen's κ is reported regardless of value — including poor κ < 0.2 —
to avoid suppressing negative results.
