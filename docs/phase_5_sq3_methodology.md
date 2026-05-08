# Phase 5 — SQ3: Human-Rater Alignment Methodology

Pre-registered methodology document for the human-rater alignment evaluation
addressing SQ3 of the thesis. Companion to `phase_4b_linz_methodology.md` and
`phase_4c_fas_troyer_methodology.md`. All design choices below are fixed before
rating commences; deviations made in response to data inspection are recorded
in a dated note at the end of this file.

## 1. Research question and scope

SQ3 (from §1 of the thesis):

> To what extent do graded scores align with human semantic similarity
> judgments, where do they systematically diverge (hypernyms, circumlocutions,
> compounds), and how do these divergence patterns relate to clinically
> recognized error categories?

**Scope.** SQ3 is evaluated on the BNT only. SVF and FAS responses lack the
single-target-per-item structure that makes pairwise rater-model agreement
interpretable in the Pakhomov (2012) sense. SVF and FAS evaluation continues
through SQ2 and SQ4.

**Two analytic strands** are reported, kept methodologically distinct:

1. **Quantitative rater-model agreement** (§4.3.1, planned). Spearman rank
   correlation between human similarity ratings and model cosine, overall and
   stratified by error category. Pakhomov et al. (2012) is the closest
   published precedent.
2. **Qualitative divergence analysis** (§4.3.2, planned, with extension into
   §5.4). Catalog of high-disagreement pairs, interpreted with reference to
   SALDO and discussed with a practising speech-language pathologist
   (Elisabeth) in a semi-structured interview.

The two strands answer different sub-questions and are reported separately.

## 2. SALDO's role in this phase

SALDO is **not** used as an automatic categorization oracle for stratification.
The stratification layer commits only to the model's own cosine distribution
(§4 below). Categorization of responses is performed by the human raters
during rating (§5.2).

SALDO is used in two restricted roles:

- As an **alternative scorer** (Pakhomov 2012 parallel) — path-length and
  Wu-Palmer similarity computed for in-vocabulary pairs, reported alongside
  cosine in the rater-model agreement table.
- As an **interpretive lens** in the qualitative divergence pass — when a
  high-cosine pair receives a low rater score (or vice versa), SALDO's
  relation between target and response is reported as a discussion point,
  not as ground truth.

This restricts SALDO's role to use cases where its associative-rather-than-
taxonomic structure (Borin and Forsberg 2009) does not undermine
interpretation. In particular, no claim is made that SALDO categorizations
represent a gold standard against which the model is evaluated.

## 3. Source data and eligibility

**Source file.** `bnt_scored_results_SwedishSBERT.csv` (3000 rows; 100
participants × 30 items). Swedish SBERT is the primary model for SQ1 and SQ2
in the present draft and is therefore used as the basis for SQ3
stratification. Rater-model agreement is reported against all five models
described in §3.3 (KB-BERT [CLS], Swedish SBERT, mE5-large, plus the two
Qwen-family models in scope at the time of SQ3 execution).

**Eligibility.** A two-stage filter, applied in fixed order: (1) drop rows
where `is_non_response == True` OR `is_exact_match == True`; (2) deduplicate
the remaining set at the `(gold, normalized)` level via stable sort and
keep-first, so the eligibility unit is the unique `(target, response)` pair
rather than the participant-response row. Both counts (row-level pre-dedup
and pair-level post-dedup) are reported in `sq3_eligibility_probe.md`.

**Excluded fields.** Rater-facing data does not include `participant_id`,
`gender`, `age`, `diagnosis`, `mmse`, or `cosine_sim`. The rater sees only the
randomized pair identifier, the target word, and the response.

## 4. Sample selection and stratification

**Stratification variable.** Cosine similarity computed against the primary
model (Swedish SBERT). Eligible pairs are partitioned into four quartiles by
cosine:

| Quartile | Cosine range (data-driven) | Substantive interpretation |
|---|---|---|
| Q1 | low (~0–25th percentile) | distant or unrelated responses |
| Q2 | (~25–50th percentile) | weak semantic relation |
| Q3 | (~50–75th percentile) | moderate relation |
| Q4 | high (~75–100th percentile, but exact matches excluded) | tight relation, includes paraphasias and hypernyms |

Quartile cutpoints are computed on the eligible-pair set and locked at
sampling time.

**Sample size.**
- **Baseline target: 200 pairs** (50 per quartile).
- **Optional extension: 300 pairs** (75 per quartile, with the additional 25
  per quartile sampled with the same random seed extension; the first 50 per
  quartile constitute the rated set even if the extension is not completed).

The 200/300 split is chosen so that an interrupted rating session still yields
a balanced and analyzable sample. If only the baseline is completed, all
analyses run on n=200. If extension is completed, all analyses are reported
on n=300 with a sensitivity check on the original 200.

**Randomization.** Within each quartile, pairs are sampled without replacement
using `numpy.random.default_rng(seed=20260507)`. The seed and the resulting
pair list are committed to the repository before rating begins.

**Pair presentation order.** Each rater receives the pair list in a permutation
generated from a rater-specific seed (`seed = 20260507 + rater_id`). This
randomizes order effects across raters while keeping the underlying pair set
identical and reproducible.

## 5. Rating instrument

### 5.1 Similarity scale

A four-point ordinal scale is used:

| Score | Anchor (Swedish target = `kamel`) | Definition |
|---|---|---|
| 0 | `cykel` (bicycle) | Unrelated. Response and target share no recognizable semantic relation. |
| 1 | `öken` (desert) | Distantly related. Some thematic or contextual association, but distinct semantic categories. |
| 2 | `häst`, `åsna` (horse, donkey) | Clearly related. Same superordinate category and similar semantic profile, but a different word. |
| 3 | `dromedar` (dromedary) | Synonymous or essentially equivalent. The response is a near-identical word for the target. |

Hypernym responses (`djur` for `kamel`) are deliberately *not* placed on this
scale by example — they are categorized separately (§5.2) and rated according
to the rater's intuitive judgment of similarity. The rubric provides this as
a worked case during rater training:

> A hypernym such as *djur* for the target *kamel* is a real and common
> retrieval error. Rate it according to your judgment of how close in
> meaning you find the two words to be — there is no pre-set "correct"
> rating for hypernyms.

This formulation is what makes the per-category Spearman analysis meaningful:
the rater is not anchored to any prior expectation of where hypernyms should
land on the scale.

### 5.2 Categorization

For each pair, raters additionally assign a primary category and a compound
flag.

**Primary category** (mutually exclusive). The rater chooses the category
that best fits the response *as a response to this target*:

| Category | Definition |
|---|---|
| `coordinate` | Same-level semantic neighbour. Same superordinate, similar semantic profile. (Tallberg's "semantic paraphasia.") |
| `hypernym` | Response is a category label that includes the target. (Tallberg's "superordinate.") |
| `hyponym` | Response is a more specific instance subsumed by the target. |
| `circumlocution` | Multi-word description of the target's function or attributes, including the case where a single Swedish noun describes the target's role rather than naming it. |
| `phonological` | Response shares phonological structure with the target but a different meaning (e.g. `kanal` for `kamel`). |
| `unrelated` | No apparent semantic or phonological relation. |
| `other` | Anything not covered above. Free-text annotation required. |

**Compound flag.** Boolean. The response is a (Swedish) noun compound or a
morphologically transparent multi-morpheme word, regardless of its primary
category. This is a *property* of the response, not a competing category.
`puckelkamel` is `coordinate` (or possibly `hyponym`, depending on rater
judgment) AND `is_compound = True`.

The category list is closed; if `other` is chosen for more than 10% of pairs,
the protocol is flagged for review before analysis.

### 5.3 Rating tool

Implementation:

- A CSV file `sq3_ratings_<rater_id>.csv` with columns:
  `pair_id, target, response, rating, category, is_compound, notes`
- Pre-populated columns: `pair_id`, `target`, `response`.
- Empty columns to be filled: `rating`, `category`, `is_compound`, `notes`.
- The rater fills these in any spreadsheet tool (LibreOffice, Excel, Google
  Sheets) and commits the completed file.

No bespoke web interface is built. The simplicity of the tool is deliberate:
a rating tool that is itself a research artifact distracts from the rating
task and complicates reproducibility.

### 5.4 Rater training

Before the live rating session, each rater works through a 20-pair training
set with the same format as the live rating set, drawn from a different
random seed (`seed = 20260514`) and excluded from any analysis. After
training, the rater and the thesis author review the training ratings
together, discuss any disagreements about the rubric, and resolve them
before live rating begins. The training discussion is documented in
`sq3_training_notes.md`.

## 6. Reliability analysis

The reliability layer depends on the rater situation at the time of rating.
Both branches are pre-specified.

### 6.1 Branch A: multi-annotator (≥2 raters)

**Design.** Full overlap. Every rater rates every pair in the sample.

**Primary IAA metric.** Quadratic-weighted Cohen's kappa (κ_w) for the
similarity rating, computed pairwise between raters. With three raters, the
mean of the three pairwise κ_w values is reported, with each pairwise value
reported individually.

**Secondary IAA metric.** Spearman rank correlation between rater pairs
(robustness check; insensitive to absolute scale use differences).

**Category-assignment IAA.** Cohen's kappa (unweighted) on the primary
category column. Cohen's kappa for the `is_compound` boolean.

**Interpretability threshold.** The rater-model analysis is reported as
primary if rater-rater κ_w ≥ 0.6 on similarity ratings (the conventional
"substantial agreement" threshold; Landis and Koch 1977). At κ_w between 0.4
and 0.6, the rater-model analysis is reported with explicit cautioning
language. At κ_w < 0.4 the rater-model analysis is reported only as
exploratory and the qualitative pass is foregrounded instead.

The threshold rule is fixed a priori; whether it is met is determined by the
data.

### 6.2 Branch B: sole annotator

**Design.** Test-retest. The same rater rates the full sample, then re-rates
a 50-pair random subset (`seed = 20260521`) at minimum two weeks later, with
the original ratings withheld and the pair order independently randomized.

**Primary metric.** Quadratic-weighted Cohen's kappa between Round 1 and
Round 2 on the 50-pair subset (intra-rater agreement).

**Secondary metric.** Spearman ρ between rounds. Cohen's kappa on category
agreement between rounds.

**Reporting language.** The Results chapter and Discussion use "intra-rater
reliability" rather than "inter-annotator agreement" throughout. The §1 SQ3
phrasing is amended to remove the plural "annotators" formulation in §5.4
of the draft (a small but necessary edit for honesty).

**Interpretability threshold.** Same κ_w thresholds as in Branch A. Single-
annotator design imposes additional discussion-level caveats independent of
reliability outcome.

## 7. Rater-model agreement

### 7.1 Primary analysis

For each model (Swedish SBERT primary plus four others):

- **Spearman ρ** between human rating (or rater mean, in Branch A) and model
  cosine, computed on the full rated set.
- **Bootstrap 95% CI** with 1000 resamples (random seed 20260528).

### 7.2 Per-stratum analysis

The same Spearman ρ computed within each cosine quartile separately, to test
whether agreement is uniform across the cosine range or concentrated at
particular ends.

### 7.3 Per-category analysis

Spearman ρ within each rater-assigned primary category, with category-level
n reported. Categories with n < 15 are reported descriptively but not
inferentially.

The principal hypothesis to be tested in the per-category analysis is whether
rater-model agreement is lower in the `hypernym`, `circumlocution`, and
`compound`-flagged subsets than in the `coordinate` subset, consistent with
the suspicion stated in §5.4 of the thesis draft. The test is reported as
descriptive comparison of point estimates with 95% CIs; given the small
per-category n, no formal hypothesis test on the differences between
correlations is performed.

### 7.4 SALDO scorer comparison

For pairs where both target and response have a SALDO sense:

- **Path-length similarity** computed via `SaldoGraph.path_length`, reported
  as `1 / (1 + path_length)` for compatibility with the cosine range.
- **Wu-Palmer similarity** via `SaldoGraph.wu_palmer`.

Spearman ρ between each SALDO score and rater rating is reported alongside
the cosine-based numbers. The SALDO OOV rate on the rated pair set is
reported as a methodological observation.

## 8. Qualitative divergence analysis

### 8.1 Divergence definition

A pair is flagged as a "divergence case" if its rater rating and model cosine
disagree by more than the 90th percentile of |rater - cosine| across the
rated set, after both are rescaled to [0, 1]. The 90th-percentile threshold
is fixed a priori; the resulting count of divergence cases (~20-30 expected
on n=200) is data-determined.

### 8.2 Divergence catalog

For each divergence case, the thesis author records:

- The (target, response) pair
- Rater rating (mean, in Branch A) and model cosine
- Rater-assigned category and compound flag
- SALDO relation between target and response, if any (mother / sibling /
  more distant / OOV)
- A one-sentence description of the apparent source of disagreement

The catalog is committed to the repository as `sq3_divergence_catalog.csv`
and is the input material for the Elisabeth interview (§9).

### 8.3 Typology

After the catalog is complete, the thesis author groups divergence cases
into a small number of recurring patterns (typically 4-7) and labels each
with a descriptive name (e.g., "morphologically-transparent compound,
SALDO-OOV"). The typology is *emergent from the data*, not pre-registered;
this is explicitly flagged in the Results section as a non-pre-specified
descriptive output.

## 9. Interview with Elisabeth

### 9.1 Format and ethics

Semi-structured interview, approximately 60-90 minutes, with the practising
speech-language pathologist Elisabeth (also a fellow master's student;
informed consent procedure follows the standard outlined in Appendix A of
the thesis). The interview is audio-recorded with consent and transcribed
for analysis. Consent and any limitations on quotation are documented
before recording begins.

### 9.2 Materials

The divergence catalog (§8.2), the typology (§8.3), and — if the timeline
permits — the clinical interface mockup (§10).

### 9.3 Interview structure

A short interview guide is prepared in advance, covering:

1. Reaction to the rated pair set as a whole (does the model's behavior
   match clinical intuitions?)
2. Walk-through of representative divergence cases per typology bucket,
   eliciting the clinical interpretation of each
3. Discussion of which divergence patterns matter clinically and which are
   acceptable noise
4. Reaction to the interface mockup (if presented), focused on what
   information would or would not be useful in a clinical context

### 9.4 Reporting

A thematic summary is prepared from the transcript, organized by the
divergence typology buckets. Direct quotation is used sparingly and only
with explicit per-quotation consent. The summary is reported in §4.3.2 and
discussed in §5.4.

## 10. Stretch goal: clinical interface mockup

**Conditional execution.** The mockup is constructed only if the rating
phase finishes ahead of the interview-week deadline. It is not a thesis
deliverable in its own right; it is interview material.

**Minimal specification.**

- A single-page HTML or Streamlit display.
- Per-pair panel showing: target word, response word, model cosine, the
  SALDO relation between target and response (if any), and the divergence
  flag.
- Navigation through the divergence catalog only.
- No interactivity beyond next/previous and a free-text comment field for
  the interview note.

The mockup is presented to Elisabeth as a discussion artifact and is
described briefly in §5.4 as part of the interview framing. If
implemented, a screenshot is included as a figure; if not implemented, the
section drops the screenshot without further consequence.

## 11. Reporting plan

### 11.1 §4.3.1 Quantitative agreement (Results)

Three tables and one figure:

- **Table 4.X** Reliability statistics (Branch A or B as applicable).
- **Table 4.X** Rater-model Spearman ρ for all five models, with bootstrap
  95% CIs. Columns for full sample, per quartile, and per category.
- **Table 4.X** Rater-SALDO Spearman ρ alongside rater-cosine for the
  in-SALDO subset.
- **Figure 4.X** Scatter of rater rating against primary-model cosine,
  point colour by rater-assigned category, with a marginal histogram of
  the residual.

### 11.2 §4.3.2 Qualitative divergence (Results)

- **Table 4.X** Divergence typology with counts and one representative pair
  per bucket.
- **Excerpt summaries** from the Elisabeth interview, organized by typology
  bucket.

### 11.3 §5.4 Discussion of SQ3

Interpretation of the agreement statistics in light of the reliability
ceiling, discussion of which divergence patterns are clinically meaningful
versus acceptable noise, comparison to Pakhomov (2012)'s reported figures
where comparable, and identification of next steps for the H70 application.

## 12. Pre-specified vs. data-driven choices

**Pre-specified (committed in this document, not revised after data
inspection):**

- Eligibility rule (§3)
- Stratification scheme (§4)
- Sample size targets (§4)
- Random seeds (§4, §5.4, §6.2, §7.1)
- Rating scale and anchors (§5.1)
- Category list (§5.2)
- Reliability metric and thresholds (§6)
- Spearman ρ as primary rater-model metric, bootstrap CI specification (§7)
- Divergence threshold (§8.1)
- Interview structure topics (§9.3)
- Reporting tables and figures (§11)

**Data-driven (declared as such in the Results section):**

- Cosine quartile cutpoints (computed from eligibility-filtered data)
- Per-category sample sizes (depend on rater categorization)
- Divergence count (depends on the |rater - cosine| distribution)
- Typology buckets in §8.3
- Themes from the interview transcript

## 13. Limitations to acknowledge in §5.7

The Limitations section will explicitly note:

- The synthetic-data origin of the rated pairs (Limitation already in the
  draft; SQ3 inherits it).
- Whichever reliability branch was executed, with its specific implications
  for the strength of claims.
- The BNT-only scope of SQ3 (no parallel rater-model agreement is computed
  for SVF or FAS).
- The single thesis-author rater (Branch B) or the small rater pool (Branch
  A) — both inherit BNT-relevant linguistic-intuition expertise but not
  clinical practice expertise; the interview with Elisabeth partially but
  not fully compensates for this.
- The per-category sample sizes are small and per-category Spearman ρ
  estimates are noisy; the analysis is reported descriptively.

## 14. Sequence of execution

1. **Data probe.** Compute eligible-pair count, cosine quartile cutpoints,
   and (informally) check that each quartile contains a reasonable mix of
   diagnostic groups. Commit the resulting summary as
   `sq3_eligibility_probe.md`.
2. **Sample draw.** Run the deterministic sampling described in §4.
   Commit the sampled pair list as `sq3_sampled_pairs.csv`.
3. **Rater training.** §5.4. Commit `sq3_training_notes.md`.
4. **Live rating.** Branches A or B. Commit `sq3_ratings_<rater_id>.csv`
   files.
5. **Reliability analysis.** §6. Commit `sq3_reliability_report.md`.
6. **Rater-model and SALDO analysis.** §7. Commit
   `sq3_agreement_report.md`.
7. **Divergence catalog and typology.** §8. Commit
   `sq3_divergence_catalog.csv` and `sq3_typology.md`.
8. **(Optional) interface mockup.** §10. Commit if executed.
9. **Interview.** §9. Commit interview guide, consent record, transcript
   (with appropriate consent), and `sq3_interview_summary.md`.
10. **Thesis write-up.** §4.3 and §5.4 in the New26 draft.

## 15. Deviations log

> Any deviation from this protocol made after rating commences must be
> recorded here, with date and reason. Empty at time of pre-registration.
"2026-05-XX: Identified upstream normalization bug stripping va from gradskiva. Bug fixed in BNT preprocessor (issue #N); BNT scored-results regenerated. Affects ≤48 pairs in the original eligible pool."
"2026-05-XX: Added pair-level deduplication to sq3_sampling.py between eligibility filtering and quartile assignment. Eligibility unit changed from row to unique (target, normalized_response) pair. Reproducibility preserved via stable sort + keep-first semantics."
---

*Pre-registration date: 2026-05-07.*
*Author: Fredrik Boglind. Companion to `phase_4b_linz_methodology.md` and*
*`phase_4c_fas_troyer_methodology.md`.*
