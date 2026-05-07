# Phase A: SALDO Graph Module

## Context

This thesis project scores neuropsychological language test responses (BNT, SVF,
FAS) using Swedish embeddings and cosine similarity. Prior phases established
that Swedish SBERT (`KBLab/sentence-bert-swedish-cased`) produces a workable
cosine range, but a distinct failure mode remains: hypernym over-scoring. On
BNT, the hypernym `djur` scores higher against `kamel` (0.623) than coordinate
animals like `åsna` (0.619), `häst` (0.615), or `lama` (0.547), and even higher
than the morphologically transparent `puckelkamel` (0.521). This is the
distributional inclusion phenomenon (Geffet & Dagan 2005) — symmetric cosine
similarity cannot encode the direction of hypernymy.

Phase A introduces a taxonomic distance metric computed from SALDO's lexical
hierarchy, which *can* encode that direction. Phase A builds the infrastructure
only; Phase B will apply it to the BNT/SVF data.

This phase is **purely additive**. No existing file is modified, no existing
pipeline changes behavior, no existing result is invalidated.

## Data Source

**SALDO v2.3** (LMF XML), available in
data/lexical/saldo.xml

- Size: ~71 MB
- Entries: 131,020 sense-level entries
- Updated: 2017-09-19
- Licence: CC-BY-4.0

The XML uses the Lexical Markup Framework (ISO 24613). The structure has been
verified and is documented in the "SALDO XML Schema" section below.

**Swesaurus was evaluated and rejected** as a supplementary resource. Coverage
testing on BNT target vocabulary showed 0 hits for 4/6 probe words and only
sparse hits for the remaining two, because Swesaurus stores synonymy/hyponymy
assertions over SALDO sense IDs (not independent lexical content) and has been
curated primarily for common vocabulary where synonymy is frequent, not the
specialized vocabulary that BNT probes. Phase A uses SALDO alone.

## Files to Create

```
src/thesis_project/lexical/
  __init__.py
  saldo.py

scripts/
  build_saldo_graph.py

tests/
  test_saldo.py
  fixtures/
    saldo_mini.xml       # hand-crafted test fixture

data/lexical/             # gitignored
  .gitkeep
```

Update `.gitignore` to exclude `data/lexical/*.xml` and `data/lexical/*.pkl`.

Update `pyproject.toml` to add `networkx` and `lxml` as dependencies (both are
widely-used, low-risk additions).

## Files NOT to Touch

- Any file under `src/thesis_project/embeddings/`
- Any file under `src/thesis_project/scoring/`
- Any file under `src/thesis_project/preprocessing/`
- Any file under `src/thesis_project/evaluation/`
- `bnt_pipeline.py`, `svf_pipeline.py`, `fas_pipeline.py`
- Any notebook under `notebooks/`
- Any existing test file

Phase A delivers a new library module. Integration with existing scorers is
Phase B.

## SALDO XML Schema (verified)

### Overall structure

```xml
<?xml version="1.0" encoding="utf-8"?>
<LexicalResource dtdVersion="16">
  <GlobalInformation>
    <feat att="languageCoding" val="ISO 639-3" />
  </GlobalInformation>
  <Lexicon>
    <feat att="language" val="swe" />
    <LexicalEntry>...</LexicalEntry>
    <LexicalEntry>...</LexicalEntry>
    ...
  </Lexicon>
</LexicalResource>
```

### Typical LexicalEntry

```xml
<LexicalEntry>
  <Lemma>
    <FormRepresentation>
      <feat att="writtenForm" val="hund" />
      <feat att="partOfSpeech" val="nn" />
      <feat att="lemgram" val="hund..nn.1" />
      <feat att="paradigm" val="nn_2u_sten" />
    </FormRepresentation>
  </Lemma>
  <Sense id="hund..1">
    <SenseRelation targets="djur..1">
      <feat att="label" val="primary" />
    </SenseRelation>
    <SenseRelation targets="sällskap..1">
      <feat att="label" val="secondary" />
    </SenseRelation>
  </Sense>
</LexicalEntry>
```

Notes on the schema:

- Every `LexicalEntry` contains exactly one `<Lemma>` and one or more `<Sense>`
  elements. `writtenForm` lives on the Lemma and is shared by all Senses in
  that entry (polysemy: `hund..1` = animal, `hund..2` = scoundrel).
- A `<Sense>` has an `id` attribute (the SALDO sense ID).
- A `<Sense>` contains zero or more `<SenseRelation>` elements with a
  `targets` attribute (another Sense ID) and a `label` feat of either
  `"primary"` or `"secondary"`.
- Direction is implicit: relations point *upward* — the source sense's primary
  descriptor is one step closer to PRIM than the source itself.

### PRIM (the root)

```xml
<LexicalEntry>
  <Lemma />
  <Sense id="PRIM..1" />
</LexicalEntry>
```

PRIM is an explicit self-closing Sense element with no relations. It has no
lemma and no writtenForm. It is the target of many relations from top-level
lexemes but has no outgoing edges. Depth of PRIM is 0 by definition.

### Swesaurus-style empty Lemma

You may encounter `<Lemma />` (self-closing) inside some SALDO entries. PRIM
is one such case. The parser should not crash on these.

### Verified global stats

- `<Sense id=` element count: 131,020
- Unique targets of `<SenseRelation>`: 44,793
- No dangling targets (every target resolves to a defined Sense)
- PRIM appears as the target of many SenseRelation elements in the full file

## Requirements

### 1. Data directory and resource download

The `scripts/build_saldo_graph.py` script is the single point of entry for
building the graph. It must:

1. Check for `data/lexical/saldo.xml`. If absent, download from
   `https://svn.spraakbanken.gu.se/sb-arkiv/pub/lmf/saldo/saldo.xml`
2. Compute and record SHA256 to `data/lexical/saldo.xml.sha256`
3. Parse XML and build the graph (see requirements below)
4. Pre-compute depths for all nodes (see §3)
5. Pickle to `data/lexical/saldo.pkl`
6. Print diagnostic stats (node count, edge count by type, depth distribution,
   count of primary-less non-PRIM senses)

The script must be idempotent: re-running it with a valid pickle present
should load and print stats without rebuilding, unless `--rebuild` is passed.

### 2. `SaldoGraph` class — public API

Located in `src/thesis_project/lexical/saldo.py`. The class must expose
exactly the following public methods. Signatures are prescriptive; internal
implementation is free.

```python
class SaldoGraph:
    @classmethod
    def from_xml(cls, xml_path: Path) -> "SaldoGraph":
        """Parse SALDO LMF XML and construct the graph. Slow (~30s)."""

    @classmethod
    def from_pickle(cls, pickle_path: Path) -> "SaldoGraph":
        """Load a previously-built graph. Fast."""

    def save_pickle(self, pickle_path: Path) -> None:
        """Persist the graph to disk."""

    # --- Lookup: writtenForm → sense_ids ---

    def lookup(self, written_form: str) -> list[str]:
        """
        Return all sense IDs for a given written form.
        Lookup is case-sensitive and assumes input is already normalized.
        Returns empty list for OOV.
        """

    # --- Sense-level attributes ---

    def written_form(self, sense_id: str) -> str | None:
        """Return the writtenForm for a sense, or None if unknown."""

    def part_of_speech(self, sense_id: str) -> str | None:
        """Return the partOfSpeech for a sense, or None if unknown."""

    def depth(self, sense_id: str) -> int | None:
        """
        Depth from PRIM via primary edges only.
        PRIM..1 has depth 0. A sense with no primary descriptor (and which
        is not PRIM itself) has depth None.
        """

    def primary_descriptor(self, sense_id: str) -> str | None:
        """Return the primary descriptor sense ID, or None."""

    def secondary_descriptor(self, sense_id: str) -> str | None:
        """Return the secondary descriptor sense ID, or None."""

    def children(self, sense_id: str) -> set[str]:
        """
        Return the set of sense IDs that have `sense_id` as their
        primary descriptor. Computed from inverse index, O(1) lookup.
        """

    # --- Pairwise similarity (on sense IDs, undirected graph) ---

    def path_length(self, s1: str, s2: str) -> int | None:
        """
        Shortest path length between two sense IDs on the undirected
        graph (primary + secondary edges combined, unit-weight).
        Returns None if either sense is unknown or the nodes are disconnected.
        """

    def wu_palmer(self, s1: str, s2: str) -> float | None:
        """
        Wu-Palmer similarity: 2 * depth(LCS) / (depth(s1) + depth(s2))
        where LCS is the lowest common subsumer in the primary-edge DAG.
        Returns a value in [0, 1], or None if either sense has undefined
        depth or no common subsumer exists.
        """

    # --- Diagnostics ---

    def coverage(self, written_forms: list[str]) -> dict:
        """
        Report coverage on a list of written forms.
        Returns: {"covered": [...], "oov": [...], "rate": float}
        """

    def stats(self) -> dict:
        """
        Report graph statistics.
        Returns a dict including at minimum:
        - n_senses: int
        - n_primary_edges: int
        - n_secondary_edges: int
        - n_senses_with_primary: int
        - n_senses_without_primary_excluding_prim: int
        - max_depth: int
        - mean_depth: float
        - depth_distribution: dict[int, int]  # histogram
        """
```

### 3. Graph construction rules

Use `networkx.DiGraph` for the internal representation.

**Nodes.** One per `<Sense id="...">` element, including PRIM. Node attributes:
- `written_form: str | None` — from the parent LexicalEntry's Lemma
- `part_of_speech: str | None` — likewise
- `lemgram: str | None` — likewise (optional, store if present)

**Edges.** One edge per `<SenseRelation>` element, directed from the Sense
containing the relation to the Sense named in `targets`. Edge attributes:
- `label: "primary" | "secondary"`

**Inverse index.** Maintain a separate `dict[str, set[str]]` mapping each
sense ID to the set of senses that have it as their *primary* descriptor. This
powers `children()` in O(1). Compute during build, not on demand.

**writtenForm index.** Maintain a `dict[str, list[str]]` mapping each
writtenForm to the list of sense IDs it names. Multi-valued because of
polysemy (`hund` → `[hund..1, hund..2]`). Compute during build.

**Depth computation.** After the graph is built:
- `depth["PRIM..1"] = 0`
- For every other sense `s`: follow the primary edge upward; depth is
  `1 + depth(parent)`. Memoize.
- If a sense has no primary edge and is not PRIM, its depth is `None`
  (top-level non-root sense — expect a small count; report in stats).

Pre-compute all depths during build and store in a flat `dict[str, int | None]`
on the graph object. Do not compute depth on demand during queries.

### 4. Wu-Palmer implementation notes

The lowest common subsumer (LCS) is the node with the greatest depth that
appears in both `ancestors(s1)` and `ancestors(s2)`, where `ancestors` follows
primary edges upward. For performance, pre-compute the primary-edge ancestor
set (as an ordered list from sense to PRIM) and cache it lazily as needed.

If either s1 or s2 has depth `None` (no path to PRIM), Wu-Palmer returns
`None`. If the only common ancestor is... — there is no such case if both
senses are connected to PRIM, but defensive coding is fine.

### 5. Tests (pytest)

Create `tests/test_saldo.py` and `tests/fixtures/saldo_mini.xml`.

**Test fixture** (`saldo_mini.xml`): a hand-crafted SALDO snippet that
exercises the full parser. Must include:
- PRIM..1 (self-closing)
- A direct child of PRIM (say, `djur..1`)
- A grandchild (say, `däggdjur..1` whose primary is `djur..1`)
- Two great-grandchildren (say, `hund..1` and `katt..1`, both primary
  `däggdjur..1`)
- `hund..1` also has a secondary descriptor (say, `sällskap..1`)
- A polysemous lemma: both `hund..1` and `hund..2` should exist, sharing
  the writtenForm "hund" but with different primary descriptors
- An entry with an empty `<Lemma />` that is NOT PRIM (edge case)

**Test cases** (minimum — more welcome):
- Parser round-trip: parse, check node count, check edge count, check a
  specific relation exists
- `lookup("hund")` returns both `hund..1` and `hund..2`
- `lookup("xyznonsense")` returns `[]`
- `depth("PRIM..1")` == 0
- `depth("djur..1")` == 1
- `depth("däggdjur..1")` == 2
- `depth("hund..1")` == 3
- `primary_descriptor("hund..1")` == `"däggdjur..1"`
- `secondary_descriptor("hund..1")` == `"sällskap..1"`
- `secondary_descriptor("katt..1")` is `None`
- `children("däggdjur..1")` == `{"hund..1", "katt..1"}`
- `path_length("hund..1", "katt..1")` == 2 (via shared parent)
- `path_length("hund..1", "PRIM..1")` == 3
- `wu_palmer("hund..1", "katt..1")` > `wu_palmer("hund..1", "djur..1")`
  — this is the key test: coordinates are more similar than a term and its
  hypernym
- `wu_palmer("hund..1", "hund..1")` == 1.0
- `coverage(["hund", "katt", "xyznonsense"])["rate"]` == 2/3
- `stats()` returns a dict with all required keys
- Pickle round-trip: save, load, verify identical stats

All tests must pass against `saldo_mini.xml`. Do **not** include any test
that requires the full 71 MB `saldo.xml` file — those would need GPU/CI
infrastructure we don't have.

### 6. Build-script acceptance

After `python scripts/build_saldo_graph.py` completes on the real saldo.xml,
the printed stats must satisfy:

- `n_senses` == 131,020
- `n_primary_edges` close to 131,019 (every sense except PRIM itself;
  minus whatever small number of top-level senses exist)
- `n_secondary_edges` is meaningfully non-zero (SALDO docs say ~50% of
  lexemes have a secondary descriptor, so expect ~60,000–70,000)
- `max_depth` between 10 and 20 (SALDO docs cite average ~6, max ~15 in an
  older version)
- `n_senses_without_primary_excluding_prim` small (likely single digits
  or low double digits)
- `depth("hund..1")` returns an integer (specific value depends on the full
  hierarchy; observe and record)
- `depth("djur..1")` returns an integer smaller than `depth("hund..1")`
- `wu_palmer("hund..1", "katt..1")` returns a value reasonably high
  (>0.7 is plausible; do not hard-code)

These are sanity checks on real data, not strict pass/fail. Include a short
section at the end of the build script that runs these checks and prints
`OK` or `WARN:` lines.

## What NOT to Do

- Do not add similarity functions that take written-form strings. All
  similarity methods operate on sense IDs. Polysemy resolution (how to pick
  which sense of `hund` when comparing to `katt`) is Phase B.
- Do not add compound decomposition. If `puckelkamel` is not in SALDO,
  `lookup("puckelkamel")` returns `[]` and callers handle it. Do not
  attempt to split the string and recurse.
- Do not integrate with any existing scorer. No import of `SaldoGraph` from
  anywhere outside `scripts/build_saldo_graph.py` and `tests/test_saldo.py`.
- Do not compute Resnik-style information-content similarity. That requires
  corpus frequencies, which is Phase C territory at earliest.
- Do not fetch the Karp API or any other remote service at query time. The
  graph is built from the local XML and queried from memory.
- Do not add caching decorators beyond the pre-computed depths and inverse
  index. No `functools.lru_cache` on query methods.
- Do not use pandas. The graph is a networkx object; coverage and stats
  return plain dicts.
- Do not add dependencies other than `networkx` and `lxml`. `lxml` is
  strictly optional — `xml.etree.ElementTree` from the stdlib will work,
  though `lxml.etree.iterparse` is faster on 71 MB files. Prefer stdlib
  unless parse time is a real problem; benchmark both if curious.
- Do not persist anything to disk other than `saldo.pkl` and the sha256 file.
- Do not touch any notebook.

## Git

Branch: `feature/saldo-graph` (create from `main`).

Commits (approximate):

1. `chore(lexical): add lexical module skeleton and dependencies`
   — create directory structure, update pyproject.toml, update .gitignore
2. `feat(lexical): parse SALDO LMF XML and build directed graph`
   — saldo.py with from_xml, node/edge construction, writtenForm/inverse
   indices
3. `feat(lexical): add depth and pairwise similarity metrics to SaldoGraph`
   — depth pre-computation, path_length, wu_palmer
4. `feat(lexical): add pickle persistence and build script`
   — from_pickle, save_pickle, scripts/build_saldo_graph.py
5. `test(lexical): add SALDO graph tests with minimal XML fixture`
   — tests/test_saldo.py and tests/fixtures/saldo_mini.xml

Squash-merge to main is fine; the commit history above is for the branch,
not for the final PR.

## Summary of Changes

| File | Action |
|------|--------|
| `src/thesis_project/lexical/__init__.py` | Create (exports SaldoGraph) |
| `src/thesis_project/lexical/saldo.py` | Create |
| `scripts/build_saldo_graph.py` | Create |
| `tests/test_saldo.py` | Create |
| `tests/fixtures/saldo_mini.xml` | Create |
| `data/lexical/.gitkeep` | Create |
| `.gitignore` | Modify (add lexical data patterns) |
| `pyproject.toml` | Modify (add networkx, lxml) |

Everything else: untouched.

## Out of Scope / Deferred to Phase B

For reference, so Claude Code doesn't start scope-creeping into these:

- Similarity between written-form strings (requires polysemy resolution)
- Integration with `bnt_scorer.py` or `svf_scorer.py`
- Compound decomposition for OOV responses
- Corpus-frequency-based information content (Resnik)
- Visualization of a response subgraph
- Analysis of coverage or depth statistics on the BNT/SVF data

All of the above are welcome in Phase B. Phase A's success criterion is:
*the graph loads, queries work, tests pass, build script produces sensible
stats on the real SALDO data.*
