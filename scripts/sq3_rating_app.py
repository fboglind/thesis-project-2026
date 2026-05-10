"""SQ3 Rating App — minimal Streamlit interface for completing
sq3_ratings_<rater>.csv.

Same data contract as the spreadsheet workflow described in
phase_5_sq3_methodology.md §5.3: reads the rater CSV, writes back to the
same path with the same column order. No internal storage format. The
identical rating set could be completed in any spreadsheet tool and
processed by sq3_analyze.py without modification.

Requires: streamlit >= 1.30 (for index=None on st.radio).

Usage:
    streamlit run scripts/sq3_rating_app.py

Configure the input/output CSV via environment variable:
    SQ3_RATINGS_CSV=data/processed/sq3/sq3_ratings_FB.csv \\
        streamlit run scripts/sq3_rating_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st


# --- Configuration ---------------------------------------------------------

RATINGS_PATH = Path(
    os.environ.get(
        "SQ3_RATINGS_CSV",
        "data/processed/sq3/sq3_ratings_FB.csv",
    )
)

EXPECTED_COLUMNS = [
    "pair_id", "target", "response",
    "rating", "category", "is_compound", "notes",
]

CATEGORIES = [
    "coordinate", "hypernym", "hyponym", "circumlocution",
    "phonological", "unrelated", "other",
]

RATING_DEFS: dict[int, tuple[str, str]] = {
    0: ("Unrelated",
        "Response and target share no recognizable semantic relation."),
    1: ("Distantly related",
        "Some thematic or contextual association, but distinct semantic "
        "categories."),
    2: ("Clearly related",
        "Same superordinate category and similar semantic profile, "
        "but a different word."),
    3: ("Synonymous",
        "Response is a near-identical or equivalent word for the target."),
}

# Anchor examples for target = kamel (BNT-aligned, purely illustrative).
KAMEL_ANCHORS = {0: "cykel", 1: "öken", 2: "häst, åsna", 3: "dromedar"}


# --- IO helpers ------------------------------------------------------------

def load_ratings(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"CSV not found: `{path}`. Set SQ3_RATINGS_CSV or place "
                 f"the file at the default path.")
        st.stop()

    df = pd.read_csv(path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        st.error(f"CSV at `{path}` is missing columns: {sorted(missing)}.")
        st.stop()

    df = df[EXPECTED_COLUMNS].copy()

    # Normalize dtypes for Streamlit editing / autosave.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")

    for col in ["pair_id", "target", "response", "category", "notes"]:
        df[col] = df[col].astype("string")

    df["is_compound"] = (
        df["is_compound"]
        .map(lambda x: False if pd.isna(x) or str(x).strip() == ""
             else str(x).strip().lower() in {"true", "1", "yes"})
        .astype("boolean")
    )

    return df


def save_ratings(df: pd.DataFrame, path: Path) -> None:
    """Atomic write via tempfile + rename — survives mid-write crashes."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def get_rating(row) -> int | None:
    val = row["rating"]
    if pd.isna(val) or str(val).strip() == "":
        return None
    return int(float(val))  # CSV round-trip may give "2" or "2.0"


def get_str(row, col) -> str | None:
    val = row[col]
    if pd.isna(val) or str(val).strip() == "":
        return None
    return str(val)


def get_bool(row, col) -> bool:
    val = row[col]
    if pd.isna(val) or str(val).strip() == "":
        return False
    return str(val).strip().lower() in {"true", "1", "yes"}


def is_rated(row) -> bool:
    return get_rating(row) is not None


# --- App -------------------------------------------------------------------

def init_state() -> None:
    if "df" not in st.session_state:
        st.session_state.df = load_ratings(RATINGS_PATH)
    if "idx" not in st.session_state:
        df = st.session_state.df
        unrated = [i for i in df.index if not is_rated(df.loc[i])]
        st.session_state.idx = int(unrated[0]) if unrated else 0


def main() -> None:
    st.set_page_config(page_title="SQ3 Rating", layout="centered")
    init_state()

    df = st.session_state.df
    n = len(df)
    idx = st.session_state.idx
    row = df.iloc[idx]
    pair_id = str(row["pair_id"])

    # Progress header
    rated_n = sum(is_rated(df.loc[i]) for i in df.index)
    st.markdown(f"### Pair {idx + 1} of {n}  ·  {rated_n}/{n} rated")
    st.progress(rated_n / n if n else 0)

    # The pair
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.markdown("**Target**")
    c1.markdown(f"## {row['target']}")
    c2.markdown("**Response**")
    c2.markdown(f"## {row['response']}")
    st.markdown("---")

    # Rating
    st.markdown("**How similar is the response to the target?**")
    cur_rating = get_rating(row)
    rating = st.radio(
        label="Rating",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"{x} — {RATING_DEFS[x][0]}",
        horizontal=True,
        index=cur_rating,  # options are [0,1,2,3] so value == index
        label_visibility="collapsed",
        key=f"rating_{pair_id}",
    )
    if rating is not None:
        st.caption(RATING_DEFS[rating][1])

    # Category
    st.markdown("**Category**")
    cur_category = get_str(row, "category")
    cat_options = ["(not yet selected)"] + CATEGORIES
    cat_idx = (
        cat_options.index(cur_category)
        if cur_category in CATEGORIES
        else 0
    )
    category_choice = st.selectbox(
        label="Category",
        options=cat_options,
        index=cat_idx,
        label_visibility="collapsed",
        key=f"cat_{pair_id}",
    )
    category = None if category_choice == "(not yet selected)" else category_choice

    # Compound flag
    is_compound = st.checkbox(
        "Response is a compound word",
        value=get_bool(row, "is_compound"),
        key=f"comp_{pair_id}",
    )

    # Notes
    notes = st.text_area(
        "Notes (optional)",
        value=get_str(row, "notes") or "",
        key=f"notes_{pair_id}",
        height=70,
    )

    # Persist any change for the current pair
    new_values = {
    "rating": rating if rating is not None else pd.NA,
    "category": category if category else pd.NA,
    "is_compound": bool(is_compound),
    "notes": notes if notes else pd.NA,
    }
    def values_differ(a, b) -> bool:
        """Safe comparison for pandas values, including pd.NA."""
        if pd.isna(a) and pd.isna(b):
            return False
        if pd.isna(a) or pd.isna(b):
            return True
        return a != b


    changed = False
    for col, val in new_values.items():
        existing = df.at[idx, col]

        if values_differ(existing, val):
            df.at[idx, col] = val
            changed = True

    # Navigation
    st.markdown("---")
    nav = st.columns([1, 1, 2, 1])
    if nav[0].button("← Previous", disabled=(idx == 0)):
        st.session_state.idx = idx - 1
        st.rerun()
    if nav[1].button("Next →", disabled=(idx == n - 1)):
        st.session_state.idx = idx + 1
        st.rerun()
    jump = nav[3].number_input(
        "Jump to pair",
        min_value=1, max_value=n, value=idx + 1, step=1,
        label_visibility="collapsed",
    )
    if jump != idx + 1:
        st.session_state.idx = int(jump) - 1
        st.rerun()

    if rated_n == n:
        st.success("All pairs rated. The CSV is ready for sq3_analyze.py.")

    # Rubric
    with st.expander("Rubric and anchor examples"):
        st.markdown("**Rating scale (0–3)**")
        for r in [0, 1, 2, 3]:
            st.markdown(
                f"- **{r} — {RATING_DEFS[r][0]}** · {RATING_DEFS[r][1]}  \n"
                f"  *Anchor for target = kamel*: `{KAMEL_ANCHORS[r]}`"
            )
        st.markdown("---")
        st.markdown("**Categories**  ")
        st.markdown(
            "- **coordinate** — same-level semantic neighbour (Tallberg's "
            "*semantic paraphasia*)  \n"
            "- **hypernym** — category label that includes the target  \n"
            "- **hyponym** — more specific instance subsumed by the target  \n"
            "- **circumlocution** — multi-word description of function or "
            "attributes  \n"
            "- **phonological** — sound-alike, different meaning  \n"
            "- **unrelated** — no apparent semantic or phonological relation  \n"
            "- **other** — anything else; add a note"
        )
        st.markdown("---")
        st.markdown("**Compound flag**  ")
        st.markdown(
            "Tick if the response is a Swedish noun compound or "
            "morphologically transparent multi-morpheme word "
            "(e.g. `puckelkamel`, `hårkam`, `dörrlås`). The flag is "
            "*independent* of the primary category — a compound can be of "
            "any category."
        )

    st.caption(f"Auto-saving to: `{RATINGS_PATH}`")


if __name__ == "__main__":
    main()
