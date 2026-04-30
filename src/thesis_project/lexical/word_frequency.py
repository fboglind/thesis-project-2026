"""word_frequency.py

Pluggable Swedish word-frequency provider with two backends.

Backends
--------
- ``wordfreq``: the wordfreq Python package, ``language='sv'``. Always
  available. Used by Linz et al. (2017) for the Mean Word Frequency
  feature, so this is the directly-comparable choice.
- ``sucx``: a SUCX-3.0-derived frequency list loaded from a CSV with
  columns ``lemma`` and ``frequency_per_million``. The Språkbanken-
  aligned alternative; preprocessing is a separate manual task and is
  out of scope for the Phase 4b code.

Both backends return Zipf scores (``log10(frequency_per_billion) + 3``)
so the two are directly comparable. Unknown words return 0.0 — the
floor on the Zipf scale, which matches the wordfreq package convention
and avoids NaN propagation through MWF averages.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

FrequencySource = Literal["wordfreq", "sucx"]


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
    ) -> None:
        self.source: FrequencySource = source
        self._sucx_table: dict[str, float] | None = None

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
        else:
            raise ValueError(
                f"Unknown frequency source {source!r}. "
                "Expected 'wordfreq' or 'sucx'."
            )

        self._empty_warning_emitted = False

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

        assert self._sucx_table is not None
        return float(self._sucx_table.get(token, 0.0))

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
