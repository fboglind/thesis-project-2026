"""data_loader.py

Data loading utilities for BNT, SVF, and FAS neuropsychological test data.
"""

from pathlib import Path

import pandas as pd
import yaml

# Resolve project root relative to this file's location:
# data_loader.py → preprocessing/ → thesis_project/ → src/ → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

with open(_PROJECT_ROOT / "configs" / "_default_configs.yaml", "r") as _f:
    _config = yaml.safe_load(_f)

DATA_DIR = _PROJECT_ROOT / _config["paths"]["data_dir"]
PROCESSED_DATA_DIR = _PROJECT_ROOT / _config["paths"]["processed_dir"]

BNT_PATH = DATA_DIR / "BNT-syntheticData_v2.xlsx"
FAS_PATH = DATA_DIR / "FAS-syntheticData_v1.xlsx"
SVF_PATH = DATA_DIR / "SVF-syntheticData_v1.xlsx"


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────

def _is_null(val) -> bool:
    """Return True if val is NaN, None, or the string 'nan'/'None'."""
    if val is None:
        return True
    try:
        import math
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return str(val).strip().lower() in ("nan", "none", "")


def _to_float(val) -> float | None:
    """Convert a cell value to float, returning None on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _find_metadata_row(df: pd.DataFrame, label: str) -> int:
    """Return the integer row index of the row whose first column starts with label."""
    first_col = df.columns[0]
    matches = df.index[df[first_col].astype(str).str.startswith(label)].tolist()
    if not matches:
        raise ValueError(f"Could not find metadata row starting with '{label}'")
    return matches[0]


# ──────────────────────────────────────────────────────
# BNT Loader
# ──────────────────────────────────────────────────────

def load_bnt_data(
    filepath: str | Path = BNT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load BNT spreadsheet and separate items from user metadata.

    The BNT sheet is laid out with one row per item (gold word + responses
    from each user), followed by metadata rows labelled ``Gender:``, ``Age:``,
    and ``Kategori:`` (diagnosis). This splits the sheet into an items frame
    and a user-metadata frame.

    Args:
        filepath: Path to the BNT XLSX file.

    Returns:
        (items_df, user_meta):

        items_df: DataFrame with columns ``['gold', <user cols>]``, one row
            per BNT item. Gold words are stripped and lowercased.
        user_meta: DataFrame with columns ``['user', 'gender', 'age',
            'diagnosis']``, one row per user.
    """
    raw = pd.read_excel(filepath)
    user_cols = [c for c in raw.columns if str(c).startswith("User")]

    gender_idx = _find_metadata_row(raw, "Gender")
    age_idx = _find_metadata_row(raw, "Age")
    cat_idx = _find_metadata_row(raw, "Kategori")

    # Items are the rows before the first metadata row; drop blank rows.
    items_df = raw.iloc[:gender_idx].copy()
    items_df = items_df[items_df["Gold"].notna()]
    items_df = items_df[["Gold"] + user_cols].reset_index(drop=True)
    items_df.rename(columns={"Gold": "gold"}, inplace=True)
    items_df["gold"] = items_df["gold"].astype(str).str.strip().str.lower()

    user_meta = pd.DataFrame(
        {
            "user": user_cols,
            "gender": raw.iloc[gender_idx][user_cols].values,
            "age": pd.to_numeric(raw.iloc[age_idx][user_cols].values, errors="coerce"),
            "diagnosis": raw.iloc[cat_idx][user_cols].values,
        }
    )

    return items_df, user_meta


# ──────────────────────────────────────────────────────
# SVF Loader
# ──────────────────────────────────────────────────────

def load_svf_data(
    filepath: str | Path = SVF_PATH,
    sheet_name: str = "SVF_djur_60s",
) -> list[dict]:
    """Load SVF spreadsheet and return per-participant records.

    Args:
        filepath: Path to the SVF XLSX file.
        sheet_name: Name of the sheet to read.

    Returns:
        List of dicts, one per participant::

            {
                "participant_id": "User-1",
                "responses": ["hund", "kisse", ...],  # ordered, Nones stripped
                "gender": "M",
                "age": 58.0,
                "diagnosis": "MCI",
            }
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=0)
    user_cols = [c for c in df.columns if str(c).startswith("User")]

    gender_idx = _find_metadata_row(df, "Gender")
    age_idx = _find_metadata_row(df, "Age")
    cat_idx = _find_metadata_row(df, "Category")

    gender_row = df.iloc[gender_idx]
    age_row = df.iloc[age_idx]
    cat_row = df.iloc[cat_idx]

    # Responses are all rows before the Gender metadata row
    response_rows = df.iloc[:gender_idx][user_cols]

    participants = []
    for user_id in user_cols:
        responses = [
            str(val).strip().lower()
            for val in response_rows[user_id]
            if not _is_null(val)
        ]
        participants.append(
            {
                "participant_id": user_id,
                "responses": responses,
                "gender": str(gender_row[user_id]).strip(),
                "age": _to_float(age_row[user_id]),
                "diagnosis": str(cat_row[user_id]).strip(),
            }
        )

    return participants


# ──────────────────────────────────────────────────────
# FAS Loader
# ──────────────────────────────────────────────────────

def _extract_flagged(word: str) -> tuple[str, list[str]]:
    """Strip angle brackets from a flagged word.

    Args:
        word: A word that may be wrapped in angle brackets, e.g. '<Anna>'.

    Returns:
        (cleaned_word, [original_flagged_form]) — the cleaned word is lowercased;
        the original form (with brackets) is returned in the list for recording.
    """
    word = word.strip()
    if word.startswith("<") and word.endswith(">"):
        return word[1:-1].lower(), [word]
    return word.lower(), []


def load_fas_data(
    filepath: str | Path = FAS_PATH,
    sheet_name: str = "FAS_simulation",
) -> list[dict]:
    """Load FAS spreadsheet and return per-participant records.

    Each data cell contains a comma-separated triplet: 'F-word, A-word, S-word'.
    Empty positions (e.g. 'fart, , stjärna') mean no word was produced for that
    letter at that time point; these empty strings are preserved in the output.

    Args:
        filepath: Path to the FAS XLSX file.
        sheet_name: Name of the sheet to read.

    Returns:
        List of dicts, one per participant::

            {
                "participant_id": "User-1",
                "responses_f": ["fägring", "fart", ...],
                "responses_a": ["ansvar", "", ...],  # "" = no word produced
                "responses_s": ["sjudning", "spår", ...],
                "flagged_errors": ["<Anna>", "<Stockholm>"],
                "gender": "M",
                "age": 51.2,
                "diagnosis": "HC",
            }
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name, header=0)
    user_cols = [c for c in df.columns if str(c).startswith("User")]

    gender_idx = _find_metadata_row(df, "Gender")
    age_idx = _find_metadata_row(df, "Age")
    cat_idx = _find_metadata_row(df, "Category")

    gender_row = df.iloc[gender_idx]
    age_row = df.iloc[age_idx]
    cat_row = df.iloc[cat_idx]

    response_rows = df.iloc[:gender_idx][user_cols]

    participants = []
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
            # Pad to exactly 3 parts (F, A, S)
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

        participants.append(
            {
                "participant_id": user_id,
                "responses_f": responses_f,
                "responses_a": responses_a,
                "responses_s": responses_s,
                "flagged_errors": flagged_errors,
                "gender": str(gender_row[user_id]).strip(),
                "age": _to_float(age_row[user_id]),
                "diagnosis": str(cat_row[user_id]).strip(),
            }
        )

    return participants
