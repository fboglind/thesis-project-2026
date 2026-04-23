"""Build the SALDO graph pickle from the LMF XML.

Usage:
    python scripts/build_saldo_graph.py            # build if missing
    python scripts/build_saldo_graph.py --rebuild  # force rebuild

If ``data/lexical/saldo.xml`` is missing, it is downloaded from Språkbanken.
The resulting graph is pickled to ``data/lexical/saldo.pkl`` and the XML's
SHA256 is written to ``data/lexical/saldo.xml.sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from pathlib import Path

# Make the src layout importable when running the script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_project.lexical import SaldoGraph  # noqa: E402

SALDO_URL = "https://svn.spraakbanken.gu.se/sb-arkiv/pub/lmf/saldo/saldo.xml"
DATA_DIR = ROOT / "data" / "lexical"
XML_PATH = DATA_DIR / "saldo.xml"
PICKLE_PATH = DATA_DIR / "saldo.pkl"
SHA256_PATH = DATA_DIR / "saldo.xml.sha256"


def download_saldo(target: Path) -> None:
    print(f"Downloading SALDO XML from {SALDO_URL} ...")
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SALDO_URL, target)
    print(f"  → saved to {target} ({target.stat().st_size / 1e6:.1f} MB)")


def write_sha256(xml_path: Path, sha_path: Path) -> str:
    h = hashlib.sha256()
    with open(xml_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    sha_path.write_text(f"{digest}  {xml_path.name}\n", encoding="utf-8")
    return digest


def build(xml_path: Path, pickle_path: Path) -> SaldoGraph:
    print(f"Parsing {xml_path} ...")
    t0 = time.perf_counter()
    graph = SaldoGraph.from_xml(xml_path)
    dt = time.perf_counter() - t0
    print(f"  → parsed in {dt:.1f}s")
    print(f"Pickling to {pickle_path} ...")
    graph.save_pickle(pickle_path)
    print(f"  → wrote {pickle_path.stat().st_size / 1e6:.1f} MB")
    return graph


def print_stats(graph: SaldoGraph) -> None:
    stats = graph.stats()
    print("---- Graph stats ----")
    print(f"n_senses:                                   {stats['n_senses']}")
    print(f"n_primary_edges:                            {stats['n_primary_edges']}")
    print(f"n_secondary_edges:                          {stats['n_secondary_edges']}")
    print(f"n_senses_with_primary:                      {stats['n_senses_with_primary']}")
    print(
        "n_senses_without_primary_excluding_prim:    "
        f"{stats['n_senses_without_primary_excluding_prim']}"
    )
    print(f"max_depth:                                  {stats['max_depth']}")
    print(f"mean_depth:                                 {stats['mean_depth']:.2f}")
    print("depth_distribution:")
    for depth in sorted(stats["depth_distribution"]):
        count = stats["depth_distribution"][depth]
        print(f"  depth {depth:>2}: {count}")


def sanity_checks(graph: SaldoGraph) -> None:
    print("---- Sanity checks ----")

    def check(label: str, condition: bool, detail: str = "") -> None:
        tag = "OK  " if condition else "WARN"
        extra = f" — {detail}" if detail else ""
        print(f"{tag}: {label}{extra}")

    stats = graph.stats()
    n = stats["n_senses"]
    check("n_senses == 131020", n == 131020, f"got {n}")
    check(
        "n_primary_edges ≥ 130000",
        stats["n_primary_edges"] >= 130000,
        f"got {stats['n_primary_edges']}",
    )
    check(
        "n_secondary_edges in [40000, 100000]",
        40000 <= stats["n_secondary_edges"] <= 100000,
        f"got {stats['n_secondary_edges']}",
    )
    check(
        "max_depth in [10, 25]",
        10 <= stats["max_depth"] <= 25,
        f"got {stats['max_depth']}",
    )
    check(
        "n_senses_without_primary_excluding_prim small (< 100)",
        stats["n_senses_without_primary_excluding_prim"] < 100,
        f"got {stats['n_senses_without_primary_excluding_prim']}",
    )

    d_hund = graph.depth("hund..1")
    d_djur = graph.depth("djur..1")
    check("depth('hund..1') is int", isinstance(d_hund, int), f"got {d_hund}")
    check("depth('djur..1') is int", isinstance(d_djur, int), f"got {d_djur}")
    if isinstance(d_hund, int) and isinstance(d_djur, int):
        check(
            "depth('djur..1') < depth('hund..1')",
            d_djur < d_hund,
            f"djur={d_djur} hund={d_hund}",
        )

    wp = graph.wu_palmer("hund..1", "katt..1")
    check(
        "wu_palmer('hund..1','katt..1') > 0.7",
        wp is not None and wp > 0.7,
        f"got {wp}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if the pickle already exists.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not XML_PATH.exists():
        download_saldo(XML_PATH)

    digest = write_sha256(XML_PATH, SHA256_PATH)
    print(f"SHA256({XML_PATH.name}) = {digest}")

    if PICKLE_PATH.exists() and not args.rebuild:
        print(f"Loading existing pickle at {PICKLE_PATH} (use --rebuild to overwrite)")
        graph = SaldoGraph.from_pickle(PICKLE_PATH)
    else:
        graph = build(XML_PATH, PICKLE_PATH)

    print_stats(graph)
    sanity_checks(graph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
