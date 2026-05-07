"""Tests for thesis_project.lexical.word_frequency.WordFrequencyProvider."""

import math
from pathlib import Path

import pytest

from thesis_project.lexical.word_frequency import WordFrequencyProvider


def test_wordfreq_backend_returns_zipf_for_known_swedish_word():
    """Common Swedish word ('hund') should have Zipf > 4."""
    provider = WordFrequencyProvider(source="wordfreq")
    assert provider.zipf_frequency("hund") > 4.0


def test_unknown_word_returns_zero():
    """A made-up word returns 0.0, not NaN, and does not crash."""
    provider = WordFrequencyProvider(source="wordfreq")
    assert provider.zipf_frequency("xkqzj") == 0.0


def test_mwf_handles_empty_list():
    """Empty word list returns NaN."""
    provider = WordFrequencyProvider(source="wordfreq")
    assert math.isnan(provider.mean_word_frequency([]))


def test_mwf_filters_whitespace_and_empty():
    """Whitespace and empty strings are filtered before lookup."""
    provider = WordFrequencyProvider(source="wordfreq")
    mwf_clean = provider.mean_word_frequency(["hund", "katt"])
    mwf_dirty = provider.mean_word_frequency(["hund", "", "  ", "katt"])
    assert abs(mwf_clean - mwf_dirty) < 1e-9


def test_mwf_lowercases_inputs():
    """Lookups are case-insensitive (surface forms only, no lemmatisation)."""
    provider = WordFrequencyProvider(source="wordfreq")
    assert provider.zipf_frequency("HUND") == provider.zipf_frequency("hund")


def test_mwf_finite_for_animal_words():
    """Common animal words give a finite, plausible Zipf mean."""
    provider = WordFrequencyProvider(source="wordfreq")
    mwf = provider.mean_word_frequency(["hund", "katt", "fågel"])
    assert math.isfinite(mwf)
    assert mwf > 3.0


def test_sucx_backend_raises_on_missing_file():
    """source='sucx' without the data file gives a clear error."""
    with pytest.raises(FileNotFoundError, match="sucx_frequencies"):
        WordFrequencyProvider(source="sucx", sucx_path=Path("/nonexistent.csv"))


def test_sucx_backend_loads_from_csv(tmp_path):
    """source='sucx' with a fixture CSV returns Zipf frequencies."""
    fixture = tmp_path / "freq.csv"
    fixture.write_text("lemma,frequency_per_million\nhund,500\nkatt,300\n")
    provider = WordFrequencyProvider(source="sucx", sucx_path=fixture)
    # 500 per million → log10(500) + 3 ≈ 5.7
    z = provider.zipf_frequency("hund")
    assert 5.0 < z < 6.0
    assert provider.zipf_frequency("not_in_list") == 0.0


def test_sucx_backend_lowercases_csv_lemmas(tmp_path):
    fixture = tmp_path / "freq.csv"
    fixture.write_text("lemma,frequency_per_million\nHund,500\n")
    provider = WordFrequencyProvider(source="sucx", sucx_path=fixture)
    assert provider.zipf_frequency("hund") > 0.0


def test_sucx_backend_rejects_csv_missing_columns(tmp_path):
    fixture = tmp_path / "bad.csv"
    fixture.write_text("word,count\nhund,500\n")
    with pytest.raises(ValueError, match="missing required columns"):
        WordFrequencyProvider(source="sucx", sucx_path=fixture)


def test_unknown_source_rejected():
    with pytest.raises(ValueError, match="Unknown frequency source"):
        WordFrequencyProvider(source="other")  # type: ignore[arg-type]
