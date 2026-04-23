"""Tests for the SALDO graph module."""

from pathlib import Path

import pytest

from thesis_project.lexical import SaldoGraph

FIXTURE = Path(__file__).parent / "fixtures" / "saldo_mini.xml"


@pytest.fixture(scope="module")
def graph() -> SaldoGraph:
    return SaldoGraph.from_xml(FIXTURE)


# --- Parsing ---------------------------------------------------------------


def test_node_count(graph: SaldoGraph) -> None:
    # PRIM, djur..1, däggdjur..1, sällskap..1, usling..1, hund..1, hund..2,
    # katt..1, mysterium..1 => 9
    assert graph.stats()["n_senses"] == 9


def test_edge_counts(graph: SaldoGraph) -> None:
    stats = graph.stats()
    # Primary edges: djur, däggdjur, sällskap, usling, hund..1, hund..2,
    # katt..1, mysterium..1 → PRIM/parent (8 edges)
    assert stats["n_primary_edges"] == 8
    # Secondary edges: hund..1 → sällskap..1
    assert stats["n_secondary_edges"] == 1


def test_relation_exists(graph: SaldoGraph) -> None:
    assert graph.primary_descriptor("hund..1") == "däggdjur..1"
    assert graph.secondary_descriptor("hund..1") == "sällskap..1"


# --- Lookup ---------------------------------------------------------------


def test_lookup_polysemy(graph: SaldoGraph) -> None:
    assert set(graph.lookup("hund")) == {"hund..1", "hund..2"}


def test_lookup_oov(graph: SaldoGraph) -> None:
    assert graph.lookup("xyznonsense") == []


def test_lookup_single(graph: SaldoGraph) -> None:
    assert graph.lookup("katt") == ["katt..1"]


# --- Attributes ----------------------------------------------------------


def test_written_form(graph: SaldoGraph) -> None:
    assert graph.written_form("hund..1") == "hund"
    assert graph.written_form("PRIM..1") is None


def test_part_of_speech(graph: SaldoGraph) -> None:
    assert graph.part_of_speech("hund..1") == "nn"
    assert graph.part_of_speech("nonexistent..1") is None


# --- Depth ---------------------------------------------------------------


def test_depth_prim(graph: SaldoGraph) -> None:
    assert graph.depth("PRIM..1") == 0


def test_depth_djur(graph: SaldoGraph) -> None:
    assert graph.depth("djur..1") == 1


def test_depth_daggdjur(graph: SaldoGraph) -> None:
    assert graph.depth("däggdjur..1") == 2


def test_depth_hund(graph: SaldoGraph) -> None:
    assert graph.depth("hund..1") == 3


def test_depth_unknown(graph: SaldoGraph) -> None:
    assert graph.depth("nonexistent..1") is None


# --- Descriptors ---------------------------------------------------------


def test_primary_descriptor(graph: SaldoGraph) -> None:
    assert graph.primary_descriptor("hund..1") == "däggdjur..1"
    assert graph.primary_descriptor("hund..2") == "usling..1"


def test_secondary_descriptor_present(graph: SaldoGraph) -> None:
    assert graph.secondary_descriptor("hund..1") == "sällskap..1"


def test_secondary_descriptor_absent(graph: SaldoGraph) -> None:
    assert graph.secondary_descriptor("katt..1") is None


def test_children(graph: SaldoGraph) -> None:
    assert graph.children("däggdjur..1") == {"hund..1", "katt..1"}


def test_children_leaf(graph: SaldoGraph) -> None:
    assert graph.children("hund..1") == set()


# --- Similarity ----------------------------------------------------------


def test_path_length_siblings(graph: SaldoGraph) -> None:
    # hund..1 → däggdjur..1 ← katt..1
    assert graph.path_length("hund..1", "katt..1") == 2


def test_path_length_to_prim(graph: SaldoGraph) -> None:
    # hund..1 → däggdjur..1 → djur..1 → PRIM..1
    assert graph.path_length("hund..1", "PRIM..1") == 3


def test_path_length_self(graph: SaldoGraph) -> None:
    assert graph.path_length("hund..1", "hund..1") == 0


def test_path_length_unknown(graph: SaldoGraph) -> None:
    assert graph.path_length("hund..1", "nonexistent..1") is None


def test_wu_palmer_coordinates_beat_hypernym(graph: SaldoGraph) -> None:
    # Coordinates (hund vs katt) share däggdjur..1 at depth 2.
    # Hypernym pair (hund vs djur) share djur..1 at depth 1.
    coord = graph.wu_palmer("hund..1", "katt..1")
    hyper = graph.wu_palmer("hund..1", "djur..1")
    assert coord is not None and hyper is not None
    assert coord > hyper


def test_wu_palmer_self(graph: SaldoGraph) -> None:
    assert graph.wu_palmer("hund..1", "hund..1") == 1.0


def test_wu_palmer_values(graph: SaldoGraph) -> None:
    # hund(3) vs katt(3), LCS=däggdjur..1 (depth 2) → 2*2/(3+3) = 0.6666...
    assert graph.wu_palmer("hund..1", "katt..1") == pytest.approx(4 / 6)
    # hund(3) vs djur(1), LCS=djur..1 (depth 1) → 2*1/(3+1) = 0.5
    assert graph.wu_palmer("hund..1", "djur..1") == pytest.approx(0.5)


def test_wu_palmer_unknown(graph: SaldoGraph) -> None:
    assert graph.wu_palmer("hund..1", "nonexistent..1") is None


# --- Coverage / stats -----------------------------------------------------


def test_coverage(graph: SaldoGraph) -> None:
    result = graph.coverage(["hund", "katt", "xyznonsense"])
    assert result["rate"] == pytest.approx(2 / 3)
    assert set(result["covered"]) == {"hund", "katt"}
    assert result["oov"] == ["xyznonsense"]


def test_stats_has_required_keys(graph: SaldoGraph) -> None:
    stats = graph.stats()
    for key in (
        "n_senses",
        "n_primary_edges",
        "n_secondary_edges",
        "n_senses_with_primary",
        "n_senses_without_primary_excluding_prim",
        "max_depth",
        "mean_depth",
        "depth_distribution",
    ):
        assert key in stats


def test_empty_lemma_non_prim(graph: SaldoGraph) -> None:
    # mysterium..1 has an empty <Lemma /> (not PRIM). It should still parse
    # and have depth 1, with None writtenForm.
    assert graph.depth("mysterium..1") == 1
    assert graph.written_form("mysterium..1") is None


# --- Pickle round-trip ----------------------------------------------------


def test_pickle_roundtrip(graph: SaldoGraph, tmp_path: Path) -> None:
    pkl = tmp_path / "mini.pkl"
    graph.save_pickle(pkl)
    loaded = SaldoGraph.from_pickle(pkl)
    assert loaded.stats() == graph.stats()
    assert loaded.depth("hund..1") == graph.depth("hund..1")
    assert loaded.wu_palmer("hund..1", "katt..1") == graph.wu_palmer(
        "hund..1", "katt..1"
    )
    assert set(loaded.lookup("hund")) == set(graph.lookup("hund"))
