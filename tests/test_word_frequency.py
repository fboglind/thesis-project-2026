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


# ──────────────────────────────────────────────────────
# Kelly backend
# ──────────────────────────────────────────────────────

KELLY_FIXTURE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<LexicalResource>
  <Lexicon>
    <LexicalEntry>
      <Lemma>
        <FormRepresentation>
          <feat att="writtenForm" val="hund"/>
          <feat att="kellyPartOfSpeech" val="noun-en"/>
          <feat att="rawFreq" val="1234"/>
          <feat att="wpm" val="500,00"/>
          <feat att="cefr" val="1"/>
          <feat att="source" val="corpus"/>
        </FormRepresentation>
      </Lemma>
    </LexicalEntry>
    <LexicalEntry>
      <Lemma>
        <FormRepresentation>
          <feat att="writtenForm" val="katt"/>
          <feat att="kellyPartOfSpeech" val="noun-en"/>
          <feat att="rawFreq" val="800"/>
          <feat att="wpm" val="300,00"/>
          <feat att="cefr" val="1"/>
          <feat att="source" val="corpus"/>
        </FormRepresentation>
      </Lemma>
    </LexicalEntry>
    <LexicalEntry>
      <Lemma>
        <FormRepresentation>
          <feat att="writtenForm" val="myrslok"/>
          <feat att="kellyPartOfSpeech" val="noun-en"/>
          <feat att="rawFreq" val=""/>
          <feat att="wpm" val="1000000,00"/>
          <feat att="cefr" val="6"/>
          <feat att="source" val="manual"/>
        </FormRepresentation>
      </Lemma>
    </LexicalEntry>
    <LexicalEntry>
      <Lemma>
        <FormRepresentation>
          <feat att="writtenForm" val="vara (vardagl. va)"/>
          <feat att="kellyPartOfSpeech" val="verb"/>
          <feat att="rawFreq" val="9999"/>
          <feat att="wpm" val="9000,00"/>
          <feat att="cefr" val="1"/>
          <feat att="source" val="corpus"/>
        </FormRepresentation>
      </Lemma>
    </LexicalEntry>
  </Lexicon>
</LexicalResource>
"""


@pytest.fixture
def kelly_fixture(tmp_path):
    path = tmp_path / "kelly_fixture.xml"
    path.write_text(KELLY_FIXTURE_XML, encoding="utf-8")
    return path


def test_kelly_backend_returns_zipf_for_known_lemma(kelly_fixture):
    """500 WPM → log10(500)+3 ≈ 5.70 Zipf."""
    provider = WordFrequencyProvider(source="kelly", kelly_path=kelly_fixture)
    z = provider.zipf_frequency("hund")
    assert 5.6 < z < 5.8


def test_kelly_backend_unknown_word_returns_zero(kelly_fixture):
    provider = WordFrequencyProvider(source="kelly", kelly_path=kelly_fixture)
    assert provider.zipf_frequency("xkqzj") == 0.0


def test_kelly_backend_skips_manual_placeholder(kelly_fixture):
    """Manual placeholder entries (empty rawFreq) must not inflate Zipf to 6.

    Without the placeholder filter, 'myrslok' would resolve to wpm=1e6
    → Zipf=9, which is nonsensical for a rare animal noun.
    """
    provider = WordFrequencyProvider(source="kelly", kelly_path=kelly_fixture)
    assert provider.zipf_frequency("myrslok") == 0.0


def test_kelly_backend_strips_parenthetical_variants(kelly_fixture):
    """'vara (vardagl. va)' should be indexed as 'vara'."""
    provider = WordFrequencyProvider(source="kelly", kelly_path=kelly_fixture)
    assert provider.zipf_frequency("vara") > 6.0


def test_kelly_backend_lenient_strips_definite_suffix(kelly_fixture):
    """'hunden' (definite-en form) should fall back to 'hund' under lenient lookup."""
    provider = WordFrequencyProvider(
        source="kelly", kelly_path=kelly_fixture, kelly_lenient=True,
    )
    assert provider.zipf_frequency("hunden") == provider.zipf_frequency("hund")


def test_kelly_backend_strict_does_not_strip_suffix(kelly_fixture):
    """With kelly_lenient=False, 'hunden' is unknown."""
    provider = WordFrequencyProvider(
        source="kelly", kelly_path=kelly_fixture, kelly_lenient=False,
    )
    assert provider.zipf_frequency("hunden") == 0.0
    assert provider.zipf_frequency("hund") > 0.0


def test_kelly_backend_mwf_finite(kelly_fixture):
    provider = WordFrequencyProvider(source="kelly", kelly_path=kelly_fixture)
    mwf = provider.mean_word_frequency(["hund", "katt"])
    assert math.isfinite(mwf)
    assert mwf > 4.5


def test_kelly_backend_lowercases_inputs(kelly_fixture):
    provider = WordFrequencyProvider(source="kelly", kelly_path=kelly_fixture)
    assert provider.zipf_frequency("HUND") == provider.zipf_frequency("hund")


def test_kelly_backend_raises_on_missing_file():
    with pytest.raises(FileNotFoundError, match="Kelly XML"):
        WordFrequencyProvider(source="kelly", kelly_path=Path("/nonexistent.xml"))


# ──────────────────────────────────────────────────────
# Kelly cefr_level()
# ──────────────────────────────────────────────────────

KELLY_CEFR_FIXTURE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<LexicalResource>
  <Lexicon>
    <LexicalEntry><Lemma><FormRepresentation>
      <feat att="writtenForm" val="hund"/>
      <feat att="kellyPartOfSpeech" val="noun-en"/>
      <feat att="rawFreq" val="500"/><feat att="wpm" val="500,00"/>
      <feat att="cefr" val="1"/><feat att="source" val="corpus"/>
    </FormRepresentation></Lemma></LexicalEntry>
    <LexicalEntry><Lemma><FormRepresentation>
      <feat att="writtenForm" val="ödla"/>
      <feat att="kellyPartOfSpeech" val="noun-en"/>
      <feat att="rawFreq" val="50"/><feat att="wpm" val="50,00"/>
      <feat att="cefr" val="3"/><feat att="source" val="corpus"/>
    </FormRepresentation></Lemma></LexicalEntry>
    <LexicalEntry><Lemma><FormRepresentation>
      <feat att="writtenForm" val="vesslas"/>
      <feat att="kellyPartOfSpeech" val="noun-en"/>
      <feat att="rawFreq" val="5"/><feat att="wpm" val="5,00"/>
      <feat att="cefr" val="6"/><feat att="source" val="corpus"/>
    </FormRepresentation></Lemma></LexicalEntry>
    <LexicalEntry><Lemma><FormRepresentation>
      <feat att="writtenForm" val="okänd"/>
      <feat att="kellyPartOfSpeech" val="noun-en"/>
      <feat att="rawFreq" val="10"/><feat att="wpm" val="10,00"/>
      <feat att="source" val="corpus"/>
    </FormRepresentation></Lemma></LexicalEntry>
  </Lexicon>
</LexicalResource>
"""


@pytest.fixture
def kelly_cefr_fixture(tmp_path):
    path = tmp_path / "kelly_cefr_fixture.xml"
    path.write_text(KELLY_CEFR_FIXTURE_XML, encoding="utf-8")
    return path


def test_cefr_level_returns_a1_for_basic_lemma(kelly_cefr_fixture):
    p = WordFrequencyProvider(source="kelly", kelly_path=kelly_cefr_fixture)
    assert p.cefr_level("hund") == "A1"


def test_cefr_level_maps_numeric_to_named_levels(kelly_cefr_fixture):
    """Kelly XML stores cefr as 1..6; provider should expose A1..C2."""
    p = WordFrequencyProvider(source="kelly", kelly_path=kelly_cefr_fixture)
    assert p.cefr_level("ödla") == "B1"
    assert p.cefr_level("vesslas") == "C2"


def test_cefr_level_returns_none_for_unknown_word(kelly_cefr_fixture):
    p = WordFrequencyProvider(source="kelly", kelly_path=kelly_cefr_fixture)
    assert p.cefr_level("xkqzj") is None


def test_cefr_level_returns_none_when_entry_lacks_cefr(kelly_cefr_fixture):
    """Entries without a cefr feat should return None, not crash."""
    p = WordFrequencyProvider(source="kelly", kelly_path=kelly_cefr_fixture)
    assert p.cefr_level("okänd") is None
    # But Zipf still works for that entry.
    assert p.zipf_frequency("okänd") > 0


def test_cefr_level_lenient_strips_definite_suffix(kelly_cefr_fixture):
    """'hunden' (definite -en form) should fall through to 'hund'."""
    p = WordFrequencyProvider(
        source="kelly", kelly_path=kelly_cefr_fixture, kelly_lenient=True,
    )
    assert p.cefr_level("hunden") == "A1"


def test_cefr_level_strict_does_not_strip_suffix(kelly_cefr_fixture):
    p = WordFrequencyProvider(
        source="kelly", kelly_path=kelly_cefr_fixture, kelly_lenient=False,
    )
    assert p.cefr_level("hunden") is None
    assert p.cefr_level("hund") == "A1"


def test_cefr_level_lowercases_input(kelly_cefr_fixture):
    p = WordFrequencyProvider(source="kelly", kelly_path=kelly_cefr_fixture)
    assert p.cefr_level("HUND") == "A1"


def test_cefr_level_returns_none_for_wordfreq_backend():
    """CEFR is a Kelly-specific tag; other backends return None."""
    p = WordFrequencyProvider(source="wordfreq")
    assert p.cefr_level("hund") is None


def test_cefr_level_returns_none_for_empty_or_none_input(kelly_cefr_fixture):
    p = WordFrequencyProvider(source="kelly", kelly_path=kelly_cefr_fixture)
    assert p.cefr_level("") is None
    assert p.cefr_level("   ") is None
    assert p.cefr_level(None) is None  # type: ignore[arg-type]
