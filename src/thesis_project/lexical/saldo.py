"""SALDO lexical graph.

Parses SALDO LMF XML into a directed graph keyed by Sense IDs, and exposes
taxonomic-distance queries (depth from PRIM, path length, Wu-Palmer
similarity).

The graph is a ``networkx.DiGraph``; edges point from a Sense to its
descriptor (primary or secondary), following the direction encoded in the
XML. PRIM..1 is the root and has depth 0.
"""

from __future__ import annotations

import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

import networkx as nx

PRIM = "PRIM..1"


class SaldoGraph:
    def __init__(
        self,
        graph: nx.DiGraph,
        form_index: dict[str, list[str]],
        children_index: dict[str, set[str]],
        depths: dict[str, int | None],
    ) -> None:
        self._g = graph
        self._form_index = form_index
        self._children_index = children_index
        self._depths = depths
        # Cache for ancestor chains along primary edges (lazy).
        self._ancestor_cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_xml(cls, xml_path: Path) -> "SaldoGraph":
        """Parse SALDO LMF XML and construct the graph. Slow (~30s)."""
        xml_path = Path(xml_path)
        graph: nx.DiGraph = nx.DiGraph()
        form_index: dict[str, list[str]] = defaultdict(list)
        children_index: dict[str, set[str]] = defaultdict(set)

        # iterparse keeps memory bounded on 70+ MB files.
        context = ET.iterparse(str(xml_path), events=("end",))
        for event, elem in context:
            if elem.tag != "LexicalEntry":
                continue

            written_forms, part_of_speech, lemgram = _parse_lemma(elem)
            primary_wf = written_forms[0] if written_forms else None

            for sense in elem.findall("Sense"):
                sense_id = sense.get("id")
                if sense_id is None:
                    continue

                graph.add_node(
                    sense_id,
                    written_form=primary_wf,
                    part_of_speech=part_of_speech,
                    lemgram=lemgram,
                )
                for wf in written_forms:
                    form_index[wf].append(sense_id)

                for relation in sense.findall("SenseRelation"):
                    target = relation.get("targets")
                    if target is None:
                        continue
                    label = _feat(relation, "label")
                    graph.add_edge(sense_id, target, label=label)
                    if label == "primary":
                        children_index[target].add(sense_id)

            elem.clear()

        # Ensure PRIM exists even if somehow missing from the input.
        if PRIM not in graph:
            graph.add_node(PRIM, written_form=None, part_of_speech=None, lemgram=None)

        depths = _compute_depths(graph)

        return cls(
            graph=graph,
            form_index=dict(form_index),
            children_index=dict(children_index),
            depths=depths,
        )

    @classmethod
    def from_pickle(cls, pickle_path: Path) -> "SaldoGraph":
        """Load a previously-built graph. Fast."""
        with open(pickle_path, "rb") as f:
            state = pickle.load(f)
        return cls(
            graph=state["graph"],
            form_index=state["form_index"],
            children_index=state["children_index"],
            depths=state["depths"],
        )

    def save_pickle(self, pickle_path: Path) -> None:
        """Persist the graph to disk."""
        state = {
            "graph": self._g,
            "form_index": self._form_index,
            "children_index": self._children_index,
            "depths": self._depths,
        }
        with open(pickle_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, written_form: str) -> list[str]:
        """Return all sense IDs for a given written form (case-sensitive)."""
        return list(self._form_index.get(written_form, []))

    # ------------------------------------------------------------------
    # Sense attributes
    # ------------------------------------------------------------------

    def written_form(self, sense_id: str) -> str | None:
        node = self._g.nodes.get(sense_id)
        return None if node is None else node.get("written_form")

    def part_of_speech(self, sense_id: str) -> str | None:
        node = self._g.nodes.get(sense_id)
        return None if node is None else node.get("part_of_speech")

    def depth(self, sense_id: str) -> int | None:
        return self._depths.get(sense_id)

    def primary_descriptor(self, sense_id: str) -> str | None:
        return self._descriptor(sense_id, "primary")

    def secondary_descriptor(self, sense_id: str) -> str | None:
        return self._descriptor(sense_id, "secondary")

    def children(self, sense_id: str) -> set[str]:
        """Senses that have ``sense_id`` as their primary descriptor."""
        return set(self._children_index.get(sense_id, set()))

    # ------------------------------------------------------------------
    # Pairwise similarity
    # ------------------------------------------------------------------

    def path_length(self, s1: str, s2: str) -> int | None:
        """Undirected shortest path length on primary+secondary edges."""
        if s1 not in self._g or s2 not in self._g:
            return None
        if s1 == s2:
            return 0
        # BFS on the undirected view.
        undirected = self._g.to_undirected(as_view=True)
        visited = {s1: 0}
        q: deque[str] = deque([s1])
        while q:
            current = q.popleft()
            d = visited[current]
            for nbr in undirected.neighbors(current):
                if nbr in visited:
                    continue
                visited[nbr] = d + 1
                if nbr == s2:
                    return d + 1
                q.append(nbr)
        return None

    def wu_palmer(self, s1: str, s2: str) -> float | None:
        """Wu-Palmer similarity on the primary-edge DAG."""
        if s1 not in self._g or s2 not in self._g:
            return None
        d1 = self._depths.get(s1)
        d2 = self._depths.get(s2)
        if d1 is None or d2 is None:
            return None
        if s1 == s2:
            return 1.0

        anc1 = self._primary_ancestors(s1)
        anc2_set = set(self._primary_ancestors(s2))
        # ``anc1`` is ordered from self → PRIM; the first ancestor also
        # appearing in anc2_set has the greatest depth among common ancestors.
        lcs = next((a for a in anc1 if a in anc2_set), None)
        if lcs is None:
            return None
        lcs_depth = self._depths.get(lcs)
        if lcs_depth is None:
            return None
        denom = d1 + d2
        if denom == 0:
            # Both are PRIM; handled above, but guard anyway.
            return 1.0
        return (2 * lcs_depth) / denom

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def coverage(self, written_forms: list[str]) -> dict:
        covered: list[str] = []
        oov: list[str] = []
        for wf in written_forms:
            if self._form_index.get(wf):
                covered.append(wf)
            else:
                oov.append(wf)
        total = len(written_forms)
        rate = (len(covered) / total) if total else 0.0
        return {"covered": covered, "oov": oov, "rate": rate}

    def stats(self) -> dict:
        n_primary = 0
        n_secondary = 0
        nodes_with_primary: set[str] = set()
        for src, _, data in self._g.edges(data=True):
            label = data.get("label")
            if label == "primary":
                n_primary += 1
                nodes_with_primary.add(src)
            elif label == "secondary":
                n_secondary += 1

        n_with_primary = len(nodes_with_primary)
        no_primary_non_prim = sum(
            1
            for node in self._g.nodes
            if node != PRIM and node not in nodes_with_primary
        )

        numeric_depths = [d for d in self._depths.values() if d is not None]
        max_depth = max(numeric_depths) if numeric_depths else 0
        mean_depth = (sum(numeric_depths) / len(numeric_depths)) if numeric_depths else 0.0
        histogram: dict[int, int] = defaultdict(int)
        for d in numeric_depths:
            histogram[d] += 1

        return {
            "n_senses": self._g.number_of_nodes(),
            "n_primary_edges": n_primary,
            "n_secondary_edges": n_secondary,
            "n_senses_with_primary": n_with_primary,
            "n_senses_without_primary_excluding_prim": no_primary_non_prim,
            "max_depth": max_depth,
            "mean_depth": mean_depth,
            "depth_distribution": dict(histogram),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _descriptor(self, sense_id: str, label: str) -> str | None:
        if sense_id not in self._g:
            return None
        for _, target, data in self._g.out_edges(sense_id, data=True):
            if data.get("label") == label:
                return target
        return None

    def _primary_ancestors(self, sense_id: str) -> list[str]:
        """Chain from sense_id up to PRIM via primary edges (inclusive)."""
        cached = self._ancestor_cache.get(sense_id)
        if cached is not None:
            return cached
        chain: list[str] = []
        current: str | None = sense_id
        seen: set[str] = set()
        while current is not None and current not in seen:
            chain.append(current)
            seen.add(current)
            current = self._descriptor(current, "primary")
        self._ancestor_cache[sense_id] = chain
        return chain


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _feat(elem: ET.Element, attribute: str) -> str | None:
    """Return the ``val`` of the child ``<feat att="{attribute}">``."""
    for child in elem.findall("feat"):
        if child.get("att") == attribute:
            return child.get("val")
    return None


def _parse_lemma(
    entry: ET.Element,
) -> tuple[list[str], str | None, str | None]:
    """Extract (written_forms, part_of_speech, lemgram) from a LexicalEntry.

    Written forms are collected across all FormRepresentation children so the
    form index can record every spelling variant. The primary POS / lemgram
    are taken from the first FormRepresentation.
    """
    lemma = entry.find("Lemma")
    if lemma is None:
        return [], None, None

    written_forms: list[str] = []
    part_of_speech: str | None = None
    lemgram: str | None = None
    seen_forms: set[str] = set()

    for fr in lemma.findall("FormRepresentation"):
        wf = _feat(fr, "writtenForm")
        if wf and wf not in seen_forms:
            written_forms.append(wf)
            seen_forms.add(wf)
        if part_of_speech is None:
            part_of_speech = _feat(fr, "partOfSpeech")
        if lemgram is None:
            lemgram = _feat(fr, "lemgram")

    return written_forms, part_of_speech, lemgram


def _compute_depths(graph: nx.DiGraph) -> dict[str, int | None]:
    """Depth from PRIM via primary edges; None if disconnected."""
    depths: dict[str, int | None] = {}

    def _depth(node: str, stack: set[str]) -> int | None:
        if node == PRIM:
            return 0
        if node in depths:
            return depths[node]
        if node in stack:
            # Cycle in primary edges; treat as disconnected.
            return None
        parent = None
        for _, target, data in graph.out_edges(node, data=True):
            if data.get("label") == "primary":
                parent = target
                break
        if parent is None:
            depths[node] = None
            return None
        stack.add(node)
        parent_depth = _depth(parent, stack)
        stack.discard(node)
        if parent_depth is None:
            depths[node] = None
            return None
        depths[node] = parent_depth + 1
        return depths[node]

    # Seed PRIM, then iterate.
    if PRIM in graph:
        depths[PRIM] = 0
    for node in graph.nodes:
        if node not in depths:
            _depth(node, set())

    return depths


__all__ = ["SaldoGraph", "PRIM"]
