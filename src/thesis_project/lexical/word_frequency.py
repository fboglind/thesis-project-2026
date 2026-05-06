"""word_frequency.py

Pluggable Swedish word-frequency provider with three backends.

Backends
--------
- ``wordfreq``: the wordfreq Python package, ``language='sv'``. Always
  available. Used by Linz et al. (2017) for the Mean Word Frequency
  feature, so this is the directly-comparable choice.
- ``kelly``: the Swedish Kelly list (LMF XML, ``data/lexical/kelly.xml``).
  CEFR-tagged frequency list of ~8,400 lemmas derived from the SweWAC
  web corpus (Volodina & Johansson Kokkinakis 2012, CC-BY-4.0).
- ``sucx``: a SUCX-3.0-derived frequency list loaded from a CSV with
  columns ``lemma`` and ``frequency_per_million``. The Språkbanken-
  aligned alternative; preprocessing is a separate manual task and is
  out of scope for the Phase 4b code.

All backends return Zipf scores (``log10(frequency_per_billion) + 3``)
so they are directly comparable. Unknown words return 0.0 — the floor
on the Zipf scale, which matches the wordfreq package convention and
avoids NaN propagation through MWF averages.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

FrequencySource = Literal["wordfreq", "sucx", "kelly"]

DEFAULT_KELLY_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "lexical" / "kelly.xml"
)


class WordFrequencyProvider:
    """Pluggable word-frequency lookup with two backends.

    Backend choice:
        - ``"wordfreq"``: uses the wordfreq Python package, ``language='sv'``.
          Always available. Used by Linz et al. (2017) for MWF.
        - ``"sucx"``: uses a SUCX-3.0-derived frequency list loaded from
          ``data/external/sucx_frequencies.csv``. The file must have
          columns ``lemma`` and ``frequency_per_million``. If the file
          is absent, instantiation raises ``FileNotFoundError`` with a
          message pointing to the preprocessing instructions.

    Frequencies are returned as Zipf scores (log10(freq_per_billion) + 3).
    This matches the wordfreq convention.
    """

    def __init__(
        self,
        source: FrequencySource = "wordfreq",
        sucx_path: Path | None = None,
        kelly_path: Path | None = None,
        kelly_lenient: bool = True,
    ) -> None:
        self.source: FrequencySource = source
        self._sucx_table: dict[str, float] | None = None
        self._kelly_table: dict[str, float] | None = None
        self._kelly_lenient: bool = bool(kelly_lenient)

        if source == "wordfreq":
            try:
                import wordfreq  # noqa: F401  (import for availability check)
            except ImportError as exc:  # pragma: no cover — dependency declared in pyproject
                raise ImportError(
                    "The wordfreq package is required for source='wordfreq'. "
                    "Install with `pip install wordfreq`."
                ) from exc
        elif source == "sucx":
            path = Path(sucx_path) if sucx_path is not None else (
                Path(__file__).resolve().parents[3]
                / "data" / "external" / "sucx_frequencies.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"sucx_frequencies CSV not found at {path}. "
                    "Generate it via the SUCX preprocessing step (see Phase 4b "
                    "follow-ups in PHASE_4B_LINZ_INSTRUCTIONS.md), or use "
                    "source='wordfreq' instead."
                )
            self._sucx_table = self._load_sucx_table(path)
        elif source == "kelly":
            path = Path(kelly_path) if kelly_path is not None else DEFAULT_KELLY_PATH
            if not path.is_file():
                raise FileNotFoundError(
                    f"Kelly XML not found at {path}. "
                    "Download from "
                    "https://svn.spraakbanken.gu.se/sb-arkiv/pub/lmf/kelly/kelly.xml "
                    "(CC-BY-4.0; cite Volodina & Johansson Kokkinakis 2012) "
                    "or use source='wordfreq' instead."
                )
            self._kelly_table = self._load_kelly_table(path)
        else:
            raise ValueError(
                f"Unknown frequency source {source!r}. "
                "Expected 'wordfreq', 'kelly', or 'sucx'."
            )

        self._empty_warning_emitted = False

    @staticmethod
    def _clean_kelly_lemma(raw: str | None) -> str | None:
        """Kelly lemmas may include parenthetical variants ('vara (vardagl. va)')
        or slash-separated alternatives. Take the canonical (first) form,
        lowercased."""
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        if "(" in s:
            s = s.split("(")[0].strip()
        if "/" in s:
            s = s.split("/")[0].strip()
        return s.lower() if s else None

    @staticmethod
    def _load_kelly_table(path: Path) -> dict[str, dict]:
        """Load Kelly LMF XML into ``{lemma: {"zipf": float, "cefr": str|None}}``.

        Schema: LexicalEntry / Lemma / FormRepresentation / feat[@att,@val]
        with att ∈ {writtenForm, partOfSpeech, kellyPartOfSpeech, wpm,
        cefr, source, ...}. WPM uses Swedish decimal comma; convert to
        Zipf via ``log10(wpm) + 3``. CEFR is numeric 1–6 in the XML and is
        normalised to the ``A1``–``C2`` strings. "Manual" placeholder
        entries (empty rawFreq, wpm=1000000) carry no corpus-derived
        frequency and are skipped to avoid inflating Zipf scores. Within
        duplicates, prefer noun POS (animal vocabulary skews noun-heavy)
        and the higher-WPM entry within a POS class.
        """
        import xml.etree.ElementTree as ET

        cefr_map = {"1": "A1", "2": "A2", "3": "B1", "4": "B2", "5": "C1", "6": "C2"}

        table: dict[str, dict] = {}
        n_entries = 0
        n_skipped_manual = 0
        n_skipped_other = 0
        duplicates = 0

        for _event, elem in ET.iterparse(path, events=("end",)):
            if elem.tag != "LexicalEntry":
                continue
            n_entries += 1

            fr = elem.find("Lemma/FormRepresentation")
            if fr is None:
                n_skipped_other += 1
                elem.clear()
                continue

            feats = {f.get("att"): f.get("val") for f in fr.findall("feat")}

            source = (feats.get("source") or "").strip().lower()
            raw_freq_str = (feats.get("rawFreq") or "").strip()
            if source == "manual" and not raw_freq_str:
                n_skipped_manual += 1
                elem.clear()
                continue

            lemma = WordFrequencyProvider._clean_kelly_lemma(feats.get("writtenForm"))
            if not lemma:
                n_skipped_other += 1
                elem.clear()
                continue

            wpm_str = (feats.get("wpm") or "").replace(",", ".").strip()
            try:
                wpm = float(wpm_str)
            except (ValueError, TypeError):
                n_skipped_other += 1
                elem.clear()
                continue
            if math.isnan(wpm) or wpm <= 0:
                n_skipped_other += 1
                elem.clear()
                continue

            kelly_pos = (feats.get("kellyPartOfSpeech") or "").strip().lower()
            is_noun = kelly_pos.startswith("noun")
            zipf = math.log10(wpm) + 3.0

            cefr_raw = (feats.get("cefr") or "").strip()
            cefr = cefr_map.get(cefr_raw, cefr_raw if cefr_raw else None)

            entry = {"zipf": zipf, "cefr": cefr, "is_noun": is_noun}

            if lemma in table:
                duplicates += 1
                existing = table[lemma]
                if is_noun and not existing["is_noun"]:
                    table[lemma] = entry
                elif is_noun == existing["is_noun"] and zipf > existing["zipf"]:
                    table[lemma] = entry
            else:
                table[lemma] = entry

            elem.clear()

        logger.info(
            "Kelly XML: %d entries → %d unique lemmas "
            "(%d duplicates resolved, %d manual placeholders skipped, %d other skips)",
            n_entries, len(table), duplicates, n_skipped_manual, n_skipped_other,
        )
        return table

    @staticmethod
    def _lenient_candidates(word: str) -> list[str]:
        """Strip common Swedish definite-form suffixes for a fallback lookup.

        Conservative — only the most frequent inflections, applied only
        when the surface form misses, to avoid spurious hits.
        """
        cands = [word]
        for suf in ("en", "et", "n", "t"):
            if word.endswith(suf) and len(word) > len(suf) + 2:
                cands.append(word[: -len(suf)])
        return cands

    @staticmethod
    def _load_sucx_table(path: Path) -> dict[str, float]:
        df = pd.read_csv(path)
        missing = {"lemma", "frequency_per_million"} - set(df.columns)
        if missing:
            raise ValueError(
                f"SUCX CSV at {path} is missing required columns: {sorted(missing)}. "
                "Expected 'lemma' and 'frequency_per_million'."
            )
        out: dict[str, float] = {}
        for lemma, fpm in zip(df["lemma"], df["frequency_per_million"]):
            if lemma is None:
                continue
            try:
                fpm_f = float(fpm)
            except (TypeError, ValueError):
                continue
            if fpm_f <= 0 or math.isnan(fpm_f):
                continue
            # Zipf = log10(freq_per_billion) + 3 = log10(freq_per_million) + 6 - 3
            zipf = math.log10(fpm_f) + 3.0
            out[str(lemma).strip().lower()] = zipf
        return out

    def zipf_frequency(self, word: str) -> float:
        """Return the Zipf frequency of ``word`` (lowercased).

        Returns 0.0 for unknown words — the wordfreq convention for
        words not in the underlying resource.
        """
        if word is None:
            return 0.0
        token = str(word).strip().lower()
        if not token:
            return 0.0

        if self.source == "wordfreq":
            from wordfreq import zipf_frequency
            return float(zipf_frequency(token, "sv"))

        if self.source == "kelly":
            entry = self._kelly_lookup(token)
            if entry is not None:
                return float(entry["zipf"])
            return 0.0

        assert self._sucx_table is not None
        return float(self._sucx_table.get(token, 0.0))

    def _kelly_lookup(self, token: str) -> dict | None:
        """Return the Kelly entry for ``token`` (lowercased) or None.

        Tries the surface form first, then — if ``kelly_lenient`` was
        set at construction — the suffix-stripped Swedish definite-form
        candidates (``-en``, ``-et``, ``-n``, ``-t``). Returns None when
        nothing matches; callers map that to 0.0 / None as appropriate.
        """
        if self.source != "kelly":
            return None
        assert self._kelly_table is not None
        hit = self._kelly_table.get(token)
        if hit is not None:
            return hit
        if self._kelly_lenient:
            for cand in self._lenient_candidates(token)[1:]:
                hit = self._kelly_table.get(cand)
                if hit is not None:
                    return hit
        return None

    def cefr_level(self, word: str) -> str | None:
        """Return the Kelly CEFR level (``A1``..``C2``) of ``word``.

        Returns None for words not in Kelly, or when the backend is not
        ``'kelly'`` (CEFR is a Kelly-specific tag). Lookup is
        case-insensitive and uses the same lenient suffix-strip
        fallback as ``zipf_frequency``.
        """
        if word is None or self.source != "kelly":
            return None
        token = str(word).strip().lower()
        if not token:
            return None
        entry = self._kelly_lookup(token)
        return entry["cefr"] if entry is not None else None

    def mean_word_frequency(self, words: list[str]) -> float:
        """Mean Zipf frequency over ``words``.

        Whitespace and empty strings are filtered out before lookup so
        the metric is robust to upstream preprocessing artefacts.
        Returns NaN for empty input (after filtering), with a one-shot
        warning per provider instance.
        """
        cleaned = [
            str(w).strip().lower()
            for w in (words or [])
            if w is not None and str(w).strip() != ""
        ]
        if not cleaned:
            if not self._empty_warning_emitted:
                logger.warning(
                    "mean_word_frequency called with empty (or fully-filtered) "
                    "word list; returning NaN."
                )
                self._empty_warning_emitted = True
            return float("nan")
        scores = [self.zipf_frequency(w) for w in cleaned]
        return float(sum(scores) / len(scores))
