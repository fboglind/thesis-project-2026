"""data_loader.py

Data loading utilities for BNT, SVF, and FAS neuropsychological test data.

Phase 4a: metadata rows are detected by label (Gender / Age / Category /
Kategori / MMSE), not by hardcoded index, so v1, v2, and v3 spreadsheets
are all loadable. The MMSE field is added to per-participant metadata for
all three tests; it is `NaN` when the source file pre-dates v3.

Per-test metadata only — `User-N` does not refer to the same individual
across BNT, SVF, and FAS files (see
`data/processed/harmonization_check_v3.csv`). Each loader reads only its
own file; cross-file metadata joins are not valid.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import pandas as pd
import yaml

# Resolve project root relative to this file's location:
# data_loader.py → preprocessing/ → thesis_project/ → src/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

with open(_PROJECT_ROOT / "configs" / "_default_configs.yaml", "r") as _f:
    _config = yaml.safe_load(_f)

DATA_DIR = _PROJECT_ROOT / _config["paths"]["data_dir"]
PROCESSED_DATA_DIR = _PROJECT_ROOT / _config["paths"]["processed_data_dir"]


def _resolve_test_path(test_name: str) -> Path:
    for entry in _config.get("data", {}).get("tests", []):
        if entry.get("name") == test_name:
            return DATA_DIR / entry["file"]
    raise KeyError(f"No data.tests entry named {test_name!r} in config")


BNT_PATH = _resolve_test_path("bnt")
SVF_PATH = _resolve_test_path("svf")
FAS_PATH = _resolve_test_path("fas")

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────

META_PATTERN = re.compile(
    r"^(?:Gender|Age|Categor|Kategori|MMSE)", re.IGNORECASE
)


def _is_null(val) -> bool:
    """Return True if val is NaN, None, or the string 'nan'/'None'/''."""
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return str(val).strip().lower() in ("nan", "none", "")


def _to_float(val) -> float | None:
    """Convert a cell value to float, returning None on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _user_columns(raw: pd.DataFrame) -> list:
    return [c for c in raw.columns if str(c).startswith("User")]


def _normalize_meta_key(label: str) -> str | None:
    """Map raw spreadsheet labels to canonical names.

    'Kategori:' / 'Category:' → 'Category'. Returns None if the label
    does not match any known metadata field.
    """
    key = str(label).rstrip(":").strip()
    kl = key.lower()
    if kl.startswith("kategori") or kl.startswith("categor"):
        return "Category"
    if kl.startswith("gender"):
        return "Gender"
    if kl.startswith("age"):
        return "Age"
    if kl.startswith("mmse"):
        return "MMSE"
    return None


def _extract_metadata_rows(raw: pd.DataFrame) -> dict[str, pd.Series]:
    """Return ``{canonical_field: row_series_over_user_cols}``.

    Detects metadata rows by label (Gender/Age/Category/Kategori/MMSE),
    not by hardcoded index, and normalizes the label to canonical form.
    Only rows whose first column matches one of those labels are returned.
    """
    first_col = raw.iloc[:, 0].astype(str)
    meta_mask = first_col.str.match(META_PATTERN, na=False)
    meta_rows = raw[meta_mask]

    user_cols = _user_columns(raw)
    out: dict[str, pd.Series] = {}
    for _, row in meta_rows.iterrows():
        canonical = _normalize_meta_key(row.iloc[0])
        if canonical is not None:
            out[canonical] = row[user_cols]
    return out


def _first_metadata_row_index(raw: pd.DataFrame) -> int | None:
    """Index of the first row whose first cell matches the metadata pattern.

    Used to slice off the response/item rows above the metadata block.
    Returns None if no metadata row is present.
    """
    first_col = raw.iloc[:, 0].astype(str)
    matches = first_col.index[first_col.str.match(META_PATTERN, na=False)]
    return int(matches[0]) if len(matches) else None


def _build_user_meta(
    user_cols: list,
    meta: dict[str, pd.Series],
    *,
    source_label: str,
) -> pd.DataFrame:
    """Assemble a (user, gender, age, diagnosis, mmse) DataFrame.

    Missing fields are filled with NaN. Numeric fields use
    ``pd.to_numeric(errors='coerce')`` so non-numeric or missing cells
    become NaN rather than raising.
    """
    def _series(field: str, default=None) -> pd.Series:
        if field in meta:
            return meta[field]
        return pd.Series([default] * len(user_cols), index=user_cols)

    gender = _series("Gender").reindex(user_cols).astype(object)
    age = pd.to_numeric(_series("Age").reindex(user_cols).values, errors="coerce")
    diagnosis = _series("Category").reindex(user_cols).astype(object)
    mmse = pd.to_numeric(_series("MMSE").reindex(user_cols).values, errors="coerce")

    if "MMSE" not in meta:
        logger.info(
            "%s: MMSE row not present in source file; mmse will be NaN.",
            source_label,
        )

    return pd.DataFrame(
        {
            "user": user_cols,
            "gender": gender.values,
            "age": age,
            "diagnosis": diagnosis.values,
            "mmse": mmse,
        }
    )


# ──────────────────────────────────────────────────────
# BNT Loader
# ──────────────────────────────────────────────────────

def load_bnt_data(
    filepath: str | Path = BNT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load BNT spreadsheet and separate items from user metadata.

    Args:
        filepath: Path to the BNT XLSX file.

    Returns:
        (items_df, user_meta):

        items_df: ``['gold', <user cols>]``, one row per BNT item; gold
            words are stripped and lowercased.
        user_meta: ``['user', 'gender', 'age', 'diagnosis', 'mmse']``,
            one row per user. ``mmse`` is NaN for v1/v2 files.
    """
    raw = pd.read_excel(filepath)
    user_cols = _user_columns(raw)
    if not user_cols:
        raise ValueError(f"No User-* columns found in {filepath}")

    meta_start = _first_metadata_row_index(raw)
    if meta_start is None:
        raise ValueError(
            f"No metadata rows (Gender/Age/Category/MMSE) found in {filepath}"
        )

    items_df = raw.iloc[:meta_start].copy()
    items_df = items_df[items_df["Gold"].notna()]
    items_df = items_df[["Gold"] + user_cols].reset_index(drop=True)
    items_df.rename(columns={"Gold": "gold"}, inplace=True)
    items_df["gold"] = items_df["gold"].astype(str).str.strip().str.lower()

    meta = _extract_metadata_rows(raw)
    user_meta = _build_user_meta(user_cols, meta, source_label=f"BNT {Path(filepath).name}")

    return items_df, user_meta


# ──────────────────────────────────────────────────────
# SVF Loader
# ──────────────────────────────────────────────────────

def load_svf_data(
    filepath: str | Path = SVF_PATH,
    sheet_name: str | None = None,
) -> list[dict]:
    """Load SVF spreadsheet and return per-participant records.

    Args:
        filepath: Path to the SVF XLSX file.
        sheet_name: Sheet to read. Defaults to the first sheet (v3 uses
            ``SVF_djur_60s``; older files may differ).

    Returns:
        List of dicts, one per participant::

            {
                "participant_id": "User-1",
                "responses": ["hund", "kisse", ...],  # Nones stripped, lowercased
                "gender": "M",
                "age": 58.0,
                "diagnosis": "MCI",
                "mmse": 27.0,                          # NaN if not present in source
            }

    Phase 4a: data loader updated to extract MMSE; the SVF pipeline does
    not yet consume it.
    """
    df = pd.read_excel(
        filepath,
        sheet_name=sheet_name if sheet_name is not None else 0,
        header=0,
    )
    user_cols = _user_columns(df)
    if not user_cols:
        raise ValueError(f"No User-* columns found in {filepath}")

    meta_start = _first_metadata_row_index(df)
    if meta_start is None:
        raise ValueError(
            f"No metadata rows (Gender/Age/Category/MMSE) found in {filepath}"
        )

    response_rows = df.iloc[:meta_start][user_cols]
    meta = _extract_metadata_rows(df)
    user_meta = _build_user_meta(user_cols, meta, source_label=f"SVF {Path(filepath).name}")
    meta_lookup = user_meta.set_index("user")

    participants: list[dict] = []
    for user_id in user_cols:
        responses = [
            str(val).strip().lower()
            for val in response_rows[user_id]
            if not _is_null(val)
        ]
        row = meta_lookup.loc[user_id]
        gender_val = row["gender"]
        diagnosis_val = row["diagnosis"]
        participants.append(
            {
                "participant_id": user_id,
                "responses": responses,
                "gender": str(gender_val).strip() if not _is_null(gender_val) else None,
                "age": _to_float(row["age"]),
                "diagnosis": (
                    str(diagnosis_val).strip() if not _is_null(diagnosis_val) else None
                ),
                "mmse": _to_float(row["mmse"]),
            }
        )

    return participants


# ──────────────────────────────────────────────────────
# FAS Loader
# ──────────────────────────────────────────────────────

def _extract_flagged(word: str) -> tuple[str, list[str]]:
    """Strip angle brackets from a flagged word.

    Returns (cleaned_word, [original_flagged_form]) — cleaned word is
    lowercased; the original form (with brackets) is recorded for audit.
    """
    word = word.strip()
    if word.startswith("<") and word.endswith(">"):
        return word[1:-1].lower(), [word]
    return word.lower(), []


def load_fas_data(
    filepath: str | Path = FAS_PATH,
    sheet_name: str | None = None,
) -> list[dict]:
    """Load FAS spreadsheet and return per-participant records.

    Each data cell holds a comma-separated triplet 'F-word, A-word, S-word'.
    Empty positions (e.g. 'fart, , stjärna') mean no word for that letter
    at that time point and are preserved as empty strings.

    Returns:
        List of dicts, one per participant::

            {
                "participant_id": "User-1",
                "responses_f": ["fägring", "fart", ...],
                "responses_a": ["ansvar", "", ...],
                "responses_s": ["sjudning", "spår", ...],
                "flagged_errors": ["<Anna>", "<Stockholm>"],
                "gender": "M",
                "age": 51.2,
                "diagnosis": "HC",
                "mmse": 28.0,                          # NaN if not present
            }

    Phase 4a: data loader updated to extract MMSE; the FAS pipeline does
    not yet consume it.
    """
    df = pd.read_excel(
        filepath,
        sheet_name=sheet_name if sheet_name is not None else 0,
        header=0,
    )
    user_cols = _user_columns(df)
    if not user_cols:
        raise ValueError(f"No User-* columns found in {filepath}")

    meta_start = _first_metadata_row_index(df)
    if meta_start is None:
        raise ValueError(
            f"No metadata rows (Gender/Age/Category/MMSE) found in {filepath}"
        )

    response_rows = df.iloc[:meta_start][user_cols]
    meta = _extract_metadata_rows(df)
    user_meta = _build_user_meta(user_cols, meta, source_label=f"FAS {Path(filepath).name}")
    meta_lookup = user_meta.set_index("user")

    participants: list[dict] = []
    for user_id in user_cols:
        responses_f: list[str] = []
        responses_a: list[str] = []
        responses_s: list[str] = []
        flagged_errors: list[str] = []

        for val in response_rows[user_id]:
            if _is_null(val):
                break  # participant stopped producing words

            cell = str(val).strip()
            parts = cell.split(",")
            while len(parts) < 3:
                parts.append("")

            f_raw, a_raw, s_raw = parts[0], parts[1], parts[2]
            f_word, f_flags = _extract_flagged(f_raw)
            a_word, a_flags = _extract_flagged(a_raw)
            s_word, s_flags = _extract_flagged(s_raw)

            flagged_errors.extend(f_flags + a_flags + s_flags)
            responses_f.append(f_word)
            responses_a.append(a_word)
            responses_s.append(s_word)

        row = meta_lookup.loc[user_id]
        gender_val = row["gender"]
        diagnosis_val = row["diagnosis"]
        participants.append(
            {
                "participant_id": user_id,
                "responses_f": responses_f,
                "responses_a": responses_a,
                "responses_s": responses_s,
                "flagged_errors": flagged_errors,
                "gender": str(gender_val).strip() if not _is_null(gender_val) else None,
                "age": _to_float(row["age"]),
                "diagnosis": (
                    str(diagnosis_val).strip() if not _is_null(diagnosis_val) else None
                ),
                "mmse": _to_float(row["mmse"]),
            }
        )

    return participants


# ──────────────────────────────────────────────────────
# Per-test user_meta accessors
# ──────────────────────────────────────────────────────

def load_svf_user_meta(filepath: str | Path = SVF_PATH) -> pd.DataFrame:
    """Return SVF user_meta DataFrame (user, gender, age, diagnosis, mmse).

    Convenience for callers that only need participant metadata (e.g.,
    cross-test independence checks). Reads only the SVF file — no
    cross-file inference.
    """
    df = pd.read_excel(filepath, sheet_name=0, header=0)
    user_cols = _user_columns(df)
    meta = _extract_metadata_rows(df)
    return _build_user_meta(user_cols, meta, source_label=f"SVF {Path(filepath).name}")


def load_fas_user_meta(filepath: str | Path = FAS_PATH) -> pd.DataFrame:
    """Return FAS user_meta DataFrame (user, gender, age, diagnosis, mmse).

    See ``load_svf_user_meta``.
    """
    df = pd.read_excel(filepath, sheet_name=0, header=0)
    user_cols = _user_columns(df)
    meta = _extract_metadata_rows(df)
    return _build_user_meta(user_cols, meta, source_label=f"FAS {Path(filepath).name}")
