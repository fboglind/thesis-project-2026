# Phase 4a: v3 Data Migration, MMSE Integration, and Modern Embedding Models (BNT)

## Context

Three threads come together in this phase, scoped to the **BNT pipeline only**:

1. **v3 data** is now available (`sweBNT-syntheticData_v3.xlsx`,
   `sweSVF-syntheticData_v3.xlsx`, `sweFAS-syntheticData_v3.xlsx`). The
   v3 files include an **MMSE** field in the participant metadata block.
   This phase updates the shared data loader to extract MMSE for all
   three tests, but only updates the BNT pipeline to consume it. The
   actual SQ4 regression analysis is deferred to a later phase; this
   phase only opens the data path.

2. **Harmonization audit complete (negative result).** The data is NOT
   harmonized across the three test files: `User-N` does not refer to
   the same individual in BNT, SVF, and FAS. Per-test metadata
   (diagnosis, MMSE, age, gender) must therefore be loaded **per test**
   and never inferred from one file when scoring another. This is a
   correctness requirement, not a stylistic preference. See
   `data/processed/harmonization_check_v3.csv` for the empirical
   verification.

3. **Three new comparison models** (Qwen3-Embedding-0.6B,
   EmbeddingGemma-300M, and Microsoft Harrier-OSS-v1-0.6B) extend the
   existing 3-model comparison (KB-BERT [CLS], Swedish SBERT,
   multilingual-e5-large) to six. All three are post-2024,
   instruction-aware, and currently competitive on multilingual
   embedding benchmarks. Adding them demonstrates methodological
   progression in the SQ2 design study and answers the obvious "why no
   SOTA?" defense question. Harrier-OSS-v1 is the newest of the three
   (released March 2026); document its release date when reporting
   results.

**Phase 4b** (subsequent phase) will mirror the pipeline-side work for
SVF and FAS once the BNT flow has been validated end-to-end. The
data-loader and embedder work in this phase is shared infrastructure
and benefits both phases.

GitHub issue: #29 (to be created).

## Critical Correctness Issue: Per-Test Metadata Only

**Do not propagate metadata across test files.** When scoring SVF, use
SVF's own metadata. When scoring BNT, use BNT's own metadata. When
scoring FAS, use FAS's own metadata. The same `User-N` ID can have
different diagnoses across files, so any cross-file metadata join is a
silent correctness bug.

The existing notebooks predate the harmonization check. Audit any
existing code that joins metadata across files and refactor it to use
per-test metadata. If unsure, the harmonization CSV in
`data/processed/harmonization_check_v3.csv` is the source of truth.

## Files to Modify

```
src/thesis_project/preprocessing/data_loader.py
src/thesis_project/embeddings/encoder.py        # only if needed (see §2)
configs/_default_configs.yaml                   # already updated; verify only
bnt_pipeline.py
tests/test_data_loaders.py
tests/test_embeddings.py
```

## Files to Create

```
scripts/run_model_comparison.py        # sweep all comparison_models on BNT
```

## Files NOT to Touch

- `svf_pipeline.py` — Phase 4b.
- `fas_pipeline.py` — Phase 4b.
- `bnt_pipeline_legacy.py` — legacy reference, frozen.
- `bnt_pipeline.py`'s internal duplicate `KBBertEmbedder` class
  (per Phase 1 directive). The data-loading code in `bnt_pipeline.py`
  IS in scope; the duplicate embedder class is NOT.
- `src/thesis_project/lexical/` — SALDO/Swesaurus code is untouched.
- `src/thesis_project/scoring/*.py` — scoring functions themselves do
  not change. Only the data flowing into them and the metadata flowing
  into output CSVs change.
- All notebooks. They will be re-run separately after pipeline updates
  pass tests.
- `phase_b_probe.py` — leftover scratch, frozen.

## Requirements

### 1. Update `data_loader.py` for v3 + MMSE (all three loaders)

Even though only the BNT pipeline is updated this phase, the data
loader's metadata extraction is shared infrastructure. Update all
three loaders (`load_bnt_data`, `load_svf_data`, `load_fas_data`) so
that Phase 4b's pipeline work can proceed without further data-loader
changes.

#### 1a. Detect metadata rows by label, not by index

The v1/v2 data loaders hardcoded row indices (e.g. rows 32–34 for
Gender/Age/Diagnosis). v3 adds MMSE and may have shifted rows. Replace
index-based detection with label-based:

```python
import re

META_PATTERN = re.compile(
    r"^(?:Gender|Age|Categor|Kategori|MMSE)", re.IGNORECASE
)

def _extract_metadata_rows(raw: pd.DataFrame) -> dict[str, pd.Series]:
    """Return {field_name: row_series} for all metadata rows found in raw.

    Field names are normalized: 'Kategori' and 'Category' both → 'Category'.
    """
    first_col = raw.iloc[:, 0].astype(str)
    meta_mask = first_col.str.match(META_PATTERN, na=False)
    meta_rows = raw[meta_mask]

    user_cols = [c for c in raw.columns if str(c).startswith("User")]
    out = {}
    for _, row in meta_rows.iterrows():
        key = str(row.iloc[0]).rstrip(":").strip()
        kl = key.lower()
        if kl.startswith("kategori") or kl.startswith("categor"):
            key = "Category"
        elif kl.startswith("gender"):
            key = "Gender"
        elif kl.startswith("age"):
            key = "Age"
        elif kl.startswith("mmse"):
            key = "MMSE"
        out[key] = row[user_cols]
    return out
```

#### 1b. Update `load_bnt_data`, `load_svf_data`, `load_fas_data`

Each loader should return a metadata DataFrame that includes MMSE when
present. The MMSE field is **optional**: if absent (e.g. v1/v2 files
loaded for legacy comparison), the loader returns `mmse=NaN` and emits
no error. Numeric coercion via `pd.to_numeric(..., errors='coerce')`.

Updated metadata schema (per test):

| Column      | Type      | Notes                                  |
|-------------|-----------|----------------------------------------|
| user        | str       | "User-1", "User-2", ...                |
| gender      | str       | "F" / "M"                              |
| age         | float     |                                        |
| diagnosis   | str       | "HC" / "MCI" / "non-AD" / "AD"         |
| mmse        | float     | NaN when not present in source file    |

Public function signatures (unchanged tuple shape, expanded metadata):

```python
def load_bnt_data(filepath: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load BNT spreadsheet.

    Returns:
        items_df:  columns ['gold'] + user_cols, one row per BNT item
        user_meta: columns ['user', 'gender', 'age', 'diagnosis', 'mmse']
    """

def load_svf_data(filepath: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SVF spreadsheet (Phase 4a: data loader updated, pipeline NOT yet)."""

def load_fas_data(filepath: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FAS spreadsheet (Phase 4a: data loader updated, pipeline NOT yet)."""
```

The internal column names in `user_meta` are lowercase (`gender`,
`age`, `diagnosis`, `mmse`) for consistency with existing pipeline
code. The metadata-row labels in the spreadsheet itself are
title-case (`Gender`, `Age`, `Category`, `MMSE`); normalize on load.

#### 1c. Do not infer cross-file metadata

Each `load_*_data` function reads only its own file. There is no
cross-file lookup, no merging, no fallback to BNT metadata when SVF
metadata is missing for a user. If a user appears in test responses
but not in its own metadata block (shouldn't happen in v3, but
defensively): emit a warning and set their metadata fields to NaN.

### 2. Verify and extend `SentenceTransformerEmbedder`

The existing class (Phase 2) already supports `prefix`, `model_kwargs`,
and `encode_kwargs`. Verify it works for the three new models without
modification, then add light support for instruction-aware models if
needed.

#### 2a. Qwen3-Embedding-0.6B

Should work via sentence-transformers, which handles last-token
pooling internally. Verify with:

```python
emb = SentenceTransformerEmbedder(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"trust_remote_code": True},
)
v = emb.embed("kamel")
# Confirm dimension; assertion value depends on model variant
```

If Qwen3 produces sensible cosine similarities (off-diagonal std > 0.05
on a small Swedish-word probe), no further changes needed. If not,
investigate prompt formatting (Qwen3 documentation recommends an
"Instruct: ... \nQuery: <text>" template for retrieval).

#### 2b. EmbeddingGemma-300M

**Verify the exact HuggingFace model identifier first.** Likely
candidates: `google/embeddinggemma-300m`, `google/embedding-gemma-300m`,
or under a different org. Check the HF model card before downloading.
Document the verified ID in a comment in `encoder.py` and update the
YAML config with the verified ID.

#### 2c. Harrier-OSS-v1 0.6B

**Verify the exact HuggingFace model identifier first.** The YAML
placeholder is `microsoft/Harrier-OSS-v1-0.6B` but the actual ID may
differ — Harrier was released in March 2026 in three sizes (270M,
0.6B, 27B); we want the 0.6B variant for parity with Qwen3-0.6B. Check
the HF model card, document the verified ID in a comment in
`encoder.py`, and update the YAML config with the verified ID.

If the model fails to load with the documented ID, do NOT attempt to
substitute a different size. Report the failure clearly and stop —
Fredrik will resolve the ID separately.

#### 2d. Optional: add `instruction_template` parameter

Both Qwen3 and EmbeddingGemma (and possibly Harrier) support
task-specific instructions. The existing `prefix` parameter handles
E5-style fixed prefixes (`"query: "`). For instruction-aware models,
the prompt format is sometimes a structured template like:

```
Instruct: <task description>
Query: <text>
```

If `prefix` is sufficient (i.e. you can encode the entire instruction
as a fixed string and concatenate the response word), do not add a new
parameter — keep the API minimal. If a richer template is genuinely
needed, add an optional `instruction_template` parameter:

```python
class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str,
        prefix: str | None = None,
        instruction_template: str | None = None,  # NEW, optional
        model_kwargs: dict | None = None,
        encode_kwargs: dict | None = None,
    ):
        ...

    def _format_input(self, text: str) -> str:
        if self.instruction_template:
            return self.instruction_template.format(text=text)
        if self.prefix:
            return self.prefix + text
        return text
```

**Decide based on what the models actually require.** Do not add
parameters speculatively.

### 3. Update BNT pipeline only

#### 3a. Update `bnt_pipeline.py` data-loading block

- Replace any v1/v2 file paths with the v3 path from
  `configs/_default_configs.yaml` under `data.tests` where `name == "bnt"`.
- Use the updated `load_bnt_data` function from
  `src/thesis_project/preprocessing/data_loader.py`.
- After scoring, merge each per-participant result with `user_meta`
  so the output CSV includes `gender`, `age`, `diagnosis`, and `mmse`
  alongside the BNT-specific metrics.

Output CSV schema:

| Column          | Source                               |
|-----------------|--------------------------------------|
| participant_id  | user_meta['user']                    |
| gender          | user_meta['gender']                  |
| age             | user_meta['age']                     |
| diagnosis       | user_meta['diagnosis']               |
| mmse            | user_meta['mmse']                    |
| ...metrics...   | scoring function output              |

#### 3b. Do not implement SQ4 analysis

This phase **does not** include:
- Regression of metrics on MMSE
- Linz-style feature selection
- Cross-validation or train/test splits
- Predicted-MMSE scatter plots

Those are scoped for a later phase. The only SQ4-relevant work here
is making sure the MMSE column is present in the BNT output CSV.

#### 3c. Do not modify `svf_pipeline.py` or `fas_pipeline.py`

Even though the data loader for SVF/FAS now returns expanded metadata,
do not modify the SVF/FAS pipeline scripts. Phase 4b will handle this.

If running an existing SVF or FAS pipeline produces extra (ignored)
metadata columns, that is acceptable. If it crashes due to the
expanded metadata, that is a sign the loaders are returning a
different shape than before — fix the loader, not the pipeline. The
loader's contract should be backwards compatible (just adds an `mmse`
column to the existing metadata structure).

### 4. Add `comparison_models` sweep script (BNT-only this phase)

Create `scripts/run_model_comparison.py`:

```python
"""Run BNT pipeline for every model in
configs/_default_configs.yaml::comparison_models.

Outputs are saved to data/processed/bnt_scored_results_<model_label>.csv,
where <model_label> matches the 'label' field in the YAML config.

This phase (4a): only --tests bnt is supported. Phase 4b will add svf
and fas.

Usage:
    python scripts/run_model_comparison.py
    python scripts/run_model_comparison.py --models Qwen3-0.6B Harrier-0.6B
    python scripts/run_model_comparison.py --dry-run
"""
```

The script should:

1. Load the `comparison_models` list from the YAML config.
2. For each model: instantiate the appropriate `Embedder` subclass.
   - `family: encoder` → `KBBertEmbedder` (with `pooling` from config)
   - `family: sbert` → `SentenceTransformerEmbedder`
   - `family: contrastive` or `instruction` → `SentenceTransformerEmbedder`
     with `prefix` / `instruction_template` / `model_kwargs` from config
3. For BNT only (this phase): load v3 data, run the pipeline, save
   CSV with the model label appended.
4. The `--tests` flag accepts `bnt` only this phase. If `svf` or `fas`
   is passed, exit with a clear message: "Phase 4a supports BNT only;
   use Phase 4b for SVF/FAS."
5. `--dry-run` prints what would be run without instantiating models or
   executing the pipeline.
6. Print per-model elapsed wall-clock time and rough VRAM use (if
   CUDA available) to inform later "is this practical for real H70
   data" discussions.

The script should be idempotent: re-running it should overwrite outputs
without crashing. Add a `--skip-existing` flag that skips already-
completed (model, test) pairs based on output-CSV presence.

### 5. Update tests

#### 5a. `tests/test_data_loaders.py`

Add tests:

```python
def test_load_bnt_v3_includes_mmse():
    """v3 BNT data: user_meta has an 'mmse' column with numeric values."""
    items_df, user_meta = load_bnt_data(V3_BNT_PATH)
    assert "mmse" in user_meta.columns
    assert pd.api.types.is_numeric_dtype(user_meta["mmse"])
    assert user_meta["mmse"].notna().sum() > 0
    assert user_meta["mmse"].dropna().between(0, 30).all()


def test_load_v2_data_returns_nan_mmse():
    """Legacy v2 data without MMSE row: 'mmse' column is all NaN."""
    items_df, user_meta = load_bnt_data(V2_BNT_PATH)
    assert "mmse" in user_meta.columns
    assert user_meta["mmse"].isna().all()


def test_load_svf_v3_includes_mmse():
    """Phase 4a sanity check: SVF loader also returns mmse column,
    even though the SVF pipeline isn't yet using it."""
    _, user_meta = load_svf_data(V3_SVF_PATH)
    assert "mmse" in user_meta.columns


def test_load_fas_v3_includes_mmse():
    """Phase 4a sanity check: FAS loader also returns mmse column."""
    _, user_meta = load_fas_data(V3_FAS_PATH)
    assert "mmse" in user_meta.columns


def test_metadata_row_detection_label_based():
    """Metadata rows detected by label, not by hardcoded index.
    Test against a fixture where metadata rows are at non-default positions."""
    ...


def test_per_test_metadata_independence():
    """SVF and BNT meta for the same User-N may differ (data is unharmonized).
    This test exists to document that no cross-file invariant should be
    expected; it does not assert metadata equality."""
    _, bnt_meta = load_bnt_data(V3_BNT_PATH)
    _, svf_meta = load_svf_data(V3_SVF_PATH)
    bnt_indexed = bnt_meta.set_index("user")
    svf_indexed = svf_meta.set_index("user")
    common = bnt_indexed.index.intersection(svf_indexed.index)
    n_mismatched = (
        (bnt_indexed.loc[common, "diagnosis"]
         != svf_indexed.loc[common, "diagnosis"])
        .sum()
    )
    print(f"Diagnosis mismatches across BNT/SVF: {n_mismatched} / {len(common)}")
```

#### 5b. `tests/test_embeddings.py`

Add smoke tests for the three new models — gated by an env var so CI
doesn't try to download multi-GB models:

```python
import os
import pytest
import numpy as np

@pytest.mark.skipif(
    os.environ.get("RUN_HEAVY_TESTS") != "1",
    reason="Heavy test, set RUN_HEAVY_TESTS=1 to run.",
)
def test_qwen3_embedding_smoke():
    emb = SentenceTransformerEmbedder(
        "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"trust_remote_code": True},
    )
    v_kamel = emb.embed("kamel")
    v_hast = emb.embed("häst")
    v_cykel = emb.embed("cykel")
    sim_animals = float(np.dot(v_kamel, v_hast)
                        / (np.linalg.norm(v_kamel) * np.linalg.norm(v_hast)))
    sim_unrelated = float(np.dot(v_kamel, v_cykel)
                          / (np.linalg.norm(v_kamel) * np.linalg.norm(v_cykel)))
    assert sim_animals > sim_unrelated  # sanity check
```

Same shape for EmbeddingGemma and Harrier once their model IDs are
verified.

## What NOT to Do

- Do **not** modify `svf_pipeline.py` or `fas_pipeline.py`. Phase 4b
  scope.
- Do **not** assume `User-N` is the same individual across BNT, SVF,
  and FAS. The data is unharmonized; per-test metadata only.
- Do **not** implement SQ4 regression analysis. This phase only opens
  the data path. Adding a regression now would lock in design choices
  prematurely.
- Do **not** modify scoring functions in
  `src/thesis_project/scoring/`. Their inputs and outputs are
  unchanged.
- Do **not** touch the SALDO/Swesaurus lexical code.
- Do **not** modify `bnt_pipeline.py`'s internal duplicate
  `KBBertEmbedder` class (per Phase 1 directive). The data-loading
  code in that file IS in scope; the embedder class is not.
- Do **not** modify any notebook. Notebooks are re-run separately.
- Do **not** auto-detect models from string patterns. All model
  configuration is explicit in YAML.
- Do **not** hardcode metadata row indices. Use label-based detection.
- Do **not** change the `Embedder` interface or `embed()` /
  `embed_batch()` signatures.
- Do **not** substitute a different Harrier model size if the 0.6B ID
  fails — report and stop.

## Summary of Changes

| File | Action |
|------|--------|
| `src/thesis_project/preprocessing/data_loader.py` | Label-based metadata extraction; MMSE column in user_meta for all three loaders |
| `src/thesis_project/embeddings/encoder.py` | Verify Qwen3 + EmbeddingGemma + Harrier compatibility; add `instruction_template` parameter only if needed |
| `bnt_pipeline.py` | Use v3 paths; merge MMSE into output CSV |
| `scripts/run_model_comparison.py` | NEW — sweep all comparison_models on BNT |
| `tests/test_data_loaders.py` | MMSE coverage for all three loaders; per-test independence test |
| `tests/test_embeddings.py` | Heavy-gated smoke tests for Qwen3 + EmbeddingGemma + Harrier |
| `configs/_default_configs.yaml` | Verify `comparison_models` includes 6 entries; update model IDs once verified |

## Acceptance Criteria

A successful run of this phase produces:

1. All existing tests pass.
2. New tests in `test_data_loaders.py` pass, including MMSE coverage
   for all three test files.
3. `python scripts/run_model_comparison.py --dry-run` lists 6 models ×
   1 test (BNT) = 6 (model, test) pairs.
4. `python scripts/run_model_comparison.py --models KB-BERT-CLS Swedish-SBERT`
   completes successfully and produces 2 output CSVs in
   `data/processed/`, each containing an `mmse` column.
5. `python scripts/run_model_comparison.py --models Qwen3-0.6B EmbGemma-300M Harrier-0.6B`
   either completes successfully OR fails with a clear, actionable
   error message identifying the model ID or trust_remote_code issue.
6. BNT output CSVs contain the columns:
   `participant_id, gender, age, diagnosis, mmse, ...metrics...`.
7. The harmonization warning is logged at pipeline startup as a
   reminder (e.g. INFO-level: "Per-test metadata only; cross-file
   joins are not valid on v3 data.").

## Followups (Phase 4b, not this phase)

Phase 4b will mirror sections 3 and 4 above for SVF and FAS:
- Update `svf_pipeline.py` and `fas_pipeline.py` to v3 + MMSE output.
- Extend `scripts/run_model_comparison.py` to default to all three
  tests.
- Phase 4b acceptance criteria: 6 models × 3 tests = 18 (model, test)
  output CSVs.

This split exists so the v3 + MMSE flow is validated end-to-end on
BNT (the simplest test, with a fixed target word and an existing
working pipeline) before being rolled out to the more complex fluency
tests.

## Git

Branch: `feature/v3-mmse-bnt-and-modern-models` (create from `main`).

Commits:

1. `feat(data): label-based metadata extraction for v3 spreadsheets`
   — `data_loader.py` refactor, MMSE field added for all three loaders.
2. `test(data): cover MMSE extraction and per-test metadata independence`
   — `test_data_loaders.py` updates.
3. `feat(pipelines): propagate MMSE to BNT output CSV`
   — `bnt_pipeline.py` updated to v3 + new output schema.
4. `feat(embeddings): support Qwen3, EmbeddingGemma, Harrier`
   — `encoder.py` changes (only if needed beyond existing
   `SentenceTransformerEmbedder`).
5. `test(embeddings): heavy-gated smoke tests for new models`
   — `test_embeddings.py` updates.
6. `feat(scripts): add run_model_comparison.py for SQ2 model sweep (BNT)`
   — new script, BNT-only this phase.
