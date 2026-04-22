# Phase 3: SVF Threshold-Based Cluster/Switch Metrics

## Context

With Swedish SBERT now producing meaningful cosine similarity scores (K-W
p=0.011 on pairwise similarity, off-diagonal std 0.138 vs KB-BERT's 0.013),
the pipeline can support threshold-based clustering following Pakhomov et al.
(2016) and Troyer et al. (1997). This task implements the `detect_clusters`
function (stub already exists in `svf_scorer.py`), integrates cluster/switch
metrics into the SVF scoring output, and replaces the coarse temporal gradient
with a regression-based slope.

GitHub issue: #26.

## Methodological Background

### The chain method (Pakhomov 2016)

Walk the response sequence left to right. If the cosine similarity between
consecutive words exceeds a threshold, those words belong to the same cluster.
When similarity drops below threshold, a switch has occurred and a new cluster
begins. Words that are not similar to either neighbour form singleton clusters
(size = 1).

Example with threshold = 0.50:

```
Responses:  hund    katt    ko     elefant   kamel    cykel
ConsecSim:     0.64    0.52   0.48     0.60    0.34
              above   above  BELOW    above   BELOW
Clusters:   [hund katt ko] [elefant kamel] [cykel]
Sizes:            3              2           1
```

Result: cluster_count = 3, switch_count = 2, mean_cluster_size = 2.0

### Cluster size convention

**Use Pakhomov (2016) convention, not Troyer (1997).** Pakhomov counts
singletons as clusters of size 1. Troyer subtracts 1 from each cluster size
(so singletons become size 0 and are sometimes excluded). Pakhomov's
convention is simpler, avoids special-casing singletons, and is what VFClust
uses. Document this choice in the docstring.

### Threshold

Pakhomov calibrated their threshold against manual cluster ratings from human
raters. We do not have manual annotations for Swedish animal fluency, so we
take a data-driven approach:

1. Run a threshold sweep from 0.30 to 0.70 in steps of 0.025.
2. At each threshold, compute cluster metrics for all 100 participants.
3. Select the threshold that maximises the Kruskal-Wallis H statistic for
   `mean_cluster_size` across diagnostic groups.
4. Report the sweep results (H vs threshold) as a calibration figure.

This is defensible as an empirically optimised threshold but must be
documented as a limitation: the threshold is tuned on the same data it is
evaluated on. With real H70 data, split-half or cross-validation would be
appropriate.

The default threshold in the stub is 0.45. **Keep this as the default
parameter value** but expect the sweep to identify a different optimal value
for Swedish SBERT (likely in the 0.40–0.55 range given the similarity
distribution we've observed).

### Regression-based temporal gradient

The current temporal gradient (first-half mean minus second-half mean) is too
coarse for short sequences. Replace with the **OLS regression slope of
consecutive similarity against position index**:

```python
positions = np.arange(len(consec_sims))
slope, intercept = np.polyfit(positions, consec_sims, 1)
```

Negative slope = similarity decreases across the trial (less clustering over
time, consistent with semantic store depletion). This uses every datapoint
and degrades gracefully even for sequences of 4–5 words (where halving
produces groups of 2).

Keep the old `temporal_gradient` key for backwards compatibility, but also add
`similarity_slope` as the new metric.

## File to Modify

```
src/thesis_project/scoring/svf_scorer.py
```

### Files to Create

```
tests/test_svf_scoring.py       — currently exists but may be empty; add tests
scripts/calibrate_threshold.py   — threshold sweep script
```

### Files NOT to Touch

- `src/thesis_project/embeddings/` — no changes.
- `src/thesis_project/scoring/graded_scorer.py` — unchanged.
- `src/thesis_project/scoring/fas_scorer.py` — unchanged.
- Pipeline scripts and notebooks.

## Current State of svf_scorer.py

`score_svf()` returns a dict with:
- `total_words`, `unique_words`, `repetitions` (count-based)
- `consecutive_similarities` (list of floats, one per adjacent pair)
- `mean_consecutive_similarity`, `pairwise_similarity_mean`
- `temporal_gradient` (first-half minus second-half)

`detect_clusters()` exists as a stub that raises `NotImplementedError`.
Its current signature:

```python
def detect_clusters(
    responses: list[str],
    consecutive_similarities: list[float],
    threshold: float = 0.45,
    method: str = "chain",
) -> dict:
```

## Requirements

### 1. Implement `detect_clusters()` — chain method

Implement the chain method as described above. The function already takes
`responses` and `consecutive_similarities` as inputs (computed by
`score_svf()`), so it does not need the embedder.

```python
def detect_clusters(
    responses: list[str],
    consecutive_similarities: list[float],
    threshold: float = 0.45,
    method: str = "chain",
) -> dict:
```

**Return dict keys:**

```python
{
    "clusters": list[list[str]],    # e.g. [["hund","katt","ko"], ["elefant","kamel"], ["cykel"]]
    "cluster_sizes": list[int],      # e.g. [3, 2, 1]
    "cluster_count": int,            # number of clusters (including singletons)
    "switch_count": int,             # number of switches = cluster_count - 1
    "mean_cluster_size": float,      # sum(cluster_sizes) / cluster_count
    "max_cluster_size": int,         # largest cluster
}
```

**Algorithm (chain method):**

```python
if len(responses) == 0:
    return {"clusters": [], "cluster_sizes": [], "cluster_count": 0,
            "switch_count": 0, "mean_cluster_size": float("nan"),
            "max_cluster_size": 0}

if len(responses) == 1:
    return {"clusters": [responses], "cluster_sizes": [1],
            "cluster_count": 1, "switch_count": 0,
            "mean_cluster_size": 1.0, "max_cluster_size": 1}

clusters = []
current_cluster = [responses[0]]

for i, sim in enumerate(consecutive_similarities):
    if sim >= threshold:
        current_cluster.append(responses[i + 1])
    else:
        clusters.append(current_cluster)
        current_cluster = [responses[i + 1]]

clusters.append(current_cluster)  # don't forget the last cluster

cluster_sizes = [len(c) for c in clusters]
cluster_count = len(clusters)
```

**The `method` parameter:** Only implement `"chain"` for now. If
`method != "chain"`, raise `ValueError`. The Troyer "cluster" method
(requiring all pairwise similarities within a candidate group to exceed
threshold) is a documented future extension.

**Edge cases:**
- 0 responses: return empty/nan as shown above
- 1 response: single cluster of size 1, 0 switches
- All similarities above threshold: one big cluster, 0 switches
- All similarities below threshold: N singleton clusters, N-1 switches

### 2. Add `similarity_slope` to `score_svf()`

After computing `consec_sims`, add regression-based slope:

```python
# Regression-based temporal slope
if len(consec_sims) >= 2:
    positions = np.arange(len(consec_sims), dtype=float)
    slope = float(np.polyfit(positions, consec_sims, 1)[0])
    result["similarity_slope"] = slope
else:
    result["similarity_slope"] = float("nan")
```

**Keep the old `temporal_gradient` computation.** Don't remove it — existing
results depend on it. Add `similarity_slope` as a new key alongside it.

### 3. Integrate `detect_clusters` into `score_svf()`

After computing consecutive similarities, call `detect_clusters` and merge
the results into the output dict. Add a `threshold` parameter to `score_svf`:

```python
def score_svf(
    responses: list[str],
    encoder,
    semantic_resource=None,
    threshold: float = 0.45,
) -> dict:
```

After the consecutive similarities section, add:

```python
    # Cluster/switch analysis
    cluster_result = detect_clusters(
        responses=responses,
        consecutive_similarities=consec_sims,
        threshold=threshold,
        method="chain",
    )
    result["cluster_count"] = cluster_result["cluster_count"]
    result["switch_count"] = cluster_result["switch_count"]
    result["mean_cluster_size"] = cluster_result["mean_cluster_size"]
    result["max_cluster_size"] = cluster_result["max_cluster_size"]
    result["clusters"] = cluster_result["clusters"]
```

For the edge case `n < 2`: set `cluster_count`, `switch_count`,
`max_cluster_size` to 0, and `mean_cluster_size` to `float("nan")`,
`clusters` to `[]`. Handle this in the early return block.

### 4. Create threshold calibration script

Create `scripts/calibrate_threshold.py`:

```python
"""Threshold calibration for SVF cluster metrics.

Sweeps similarity thresholds and finds the value that maximises
Kruskal-Wallis H for group discrimination on mean_cluster_size.

Usage:
    python scripts/calibrate_threshold.py
    python scripts/calibrate_threshold.py --model sbert
    python scripts/calibrate_threshold.py --mock
"""
```

The script should:

1. Load SVF data using `load_svf_data`.
2. Initialise embedder (Swedish SBERT by default, with `--mock` flag for
   testing; accept `--model` flag with values `kbbert`, `sbert`).
3. For each participant, compute `score_svf` with a range of thresholds
   (0.30 to 0.70, step 0.025).
4. At each threshold, compute Kruskal-Wallis H for `mean_cluster_size`,
   `switch_count`, and `cluster_count` across diagnostic groups.
5. Print a table of threshold → H → p for each metric.
6. Identify and print the optimal threshold for each metric.
7. Save a CSV of results to `data/processed/threshold_calibration.csv`.

**Implementation note:** To avoid re-embedding for every threshold, compute
`score_svf` once (which returns `consecutive_similarities`), then call
`detect_clusters` repeatedly with different thresholds on the pre-computed
similarities. This is efficient because the embedding step is the expensive
part and threshold variation only affects the cluster-assignment logic.

```python
# Pseudo-structure
for participant in participants:
    metrics_base = score_svf(responses, encoder, threshold=0.45)
    consec_sims = metrics_base["consecutive_similarities"]

    for threshold in thresholds:
        cluster_result = detect_clusters(responses, consec_sims, threshold)
        # store cluster_result per participant per threshold
```

### 5. Write tests in `tests/test_svf_scoring.py`

Add tests for `detect_clusters`. These tests are pure logic (no embedder
needed):

```python
from thesis_project.scoring.svf_scorer import detect_clusters


def test_chain_basic():
    """Basic chain clustering with clear clusters."""
    responses = ["hund", "katt", "ko", "elefant", "kamel", "cykel"]
    consec_sims = [0.64, 0.52, 0.38, 0.60, 0.34]
    result = detect_clusters(responses, consec_sims, threshold=0.50)

    assert result["cluster_count"] == 3
    assert result["switch_count"] == 2
    assert result["cluster_sizes"] == [3, 2, 1]
    assert result["mean_cluster_size"] == 2.0
    assert result["max_cluster_size"] == 3
    assert result["clusters"] == [["hund", "katt", "ko"],
                                   ["elefant", "kamel"],
                                   ["cykel"]]


def test_chain_all_above_threshold():
    """All consecutive similarities above threshold → one big cluster."""
    responses = ["hund", "katt", "ko"]
    consec_sims = [0.70, 0.65]
    result = detect_clusters(responses, consec_sims, threshold=0.50)

    assert result["cluster_count"] == 1
    assert result["switch_count"] == 0
    assert result["mean_cluster_size"] == 3.0


def test_chain_all_below_threshold():
    """All below threshold → all singletons."""
    responses = ["hund", "cykel", "bord"]
    consec_sims = [0.20, 0.15]
    result = detect_clusters(responses, consec_sims, threshold=0.50)

    assert result["cluster_count"] == 3
    assert result["switch_count"] == 2
    assert result["mean_cluster_size"] == 1.0


def test_chain_single_word():
    """Single word → one singleton cluster."""
    result = detect_clusters(["hund"], [], threshold=0.50)
    assert result["cluster_count"] == 1
    assert result["switch_count"] == 0
    assert result["mean_cluster_size"] == 1.0


def test_chain_empty():
    """Empty input."""
    result = detect_clusters([], [], threshold=0.50)
    assert result["cluster_count"] == 0
    assert result["switch_count"] == 0


def test_chain_threshold_boundary():
    """Similarity exactly at threshold should be included in cluster."""
    responses = ["a", "b", "c"]
    consec_sims = [0.50, 0.49]
    result = detect_clusters(responses, consec_sims, threshold=0.50)

    assert result["cluster_count"] == 2
    assert result["cluster_sizes"] == [2, 1]


def test_invalid_method():
    """Non-chain method raises ValueError."""
    import pytest
    with pytest.raises(ValueError):
        detect_clusters(["a", "b"], [0.5], threshold=0.5, method="cluster")
```

Also add a test for `similarity_slope` in `score_svf`:

```python
from thesis_project.scoring.svf_scorer import score_svf
from thesis_project.embeddings import MockEmbedder


def test_score_svf_includes_cluster_metrics():
    """score_svf output includes cluster/switch metrics."""
    encoder = MockEmbedder()
    result = score_svf(["hund", "katt", "ko", "elefant"], encoder)

    assert "cluster_count" in result
    assert "switch_count" in result
    assert "mean_cluster_size" in result
    assert "max_cluster_size" in result
    assert "similarity_slope" in result
    assert isinstance(result["cluster_count"], int)
    assert isinstance(result["similarity_slope"], float)


def test_score_svf_short_sequence():
    """Single-word sequence returns nan for cluster metrics."""
    encoder = MockEmbedder()
    result = score_svf(["hund"], encoder)

    assert result["cluster_count"] == 0
    assert np.isnan(result["mean_cluster_size"])
    assert np.isnan(result["similarity_slope"])
```

## What NOT to Do

- Do NOT implement the Troyer "cluster" method (all-pairwise). Chain only.
- Do NOT remove the existing `temporal_gradient` key from the output.
- Do NOT modify the embedder or graded_scorer.
- Do NOT change the default `threshold=0.45` in the function signature (the
  calibration script will find the optimal value, which gets passed in at
  call time).
- Do NOT hardcode a specific model in the calibration script — accept it as
  a parameter.
- Do NOT modify pipeline scripts or notebooks.

## Summary of Changes

| File | Action |
|------|--------|
| `src/thesis_project/scoring/svf_scorer.py` | Implement `detect_clusters` (chain method). Add `threshold` parameter and cluster metrics to `score_svf`. Add `similarity_slope`. |
| `tests/test_svf_scoring.py` | Add tests for `detect_clusters` and updated `score_svf` output |
| `scripts/calibrate_threshold.py` | Create threshold sweep script |

## Git

Branch: `feature/svf-cluster-metrics` (create from `main` — merge Phase 2
first).

Commits:
1. `feat(scoring): implement chain-based cluster detection for SVF`
   — `detect_clusters` implementation + integration into `score_svf`
2. `feat(scoring): add regression-based similarity slope to SVF metrics`
   — `similarity_slope` addition
3. `test(scoring): add SVF cluster detection tests`
   — tests/test_svf_scoring.py
4. `feat(scripts): add threshold calibration script`
   — scripts/calibrate_threshold.py
