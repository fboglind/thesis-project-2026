# Data Directory README

Synthetic Swedish neuropsychological language assessment data for embedding-based paraphasia scoring research.

## Overview

| File | Test | Participants | Items/Rows | Description |
|------|------|--------------|------------|-------------|
| `bnt_v2.xlsx` | Boston Naming Test | 100 | 30 items | Confrontation naming |
| `svf_v1.xlsx` | Semantic Verbal Fluency | 100 | ~25 responses | Animal naming |
| `fas_v1.xlsx` | Phonemic Verbal Fluency | 100 | ~15 responses | F-A-S letter fluency |

**Encoding:** UTF-8  
**Language:** Swedish  
**Data type:** Synthetic (LLM-generated)

---

## File Schemas

### bnt_v2.xlsx — Boston Naming Test

Participants name pictures; responses compared against target words.

| Column | Type | Description |
|--------|------|-------------|
| `Gold` | string | Target word (correct answer) |
| `User-1` ... `User-100` | string | Participant responses |

**Metadata rows:**
| Row | Field | Values |
|-----|-------|--------|
| 34 | `Gender:` | `M`, `F` |
| 35 | `Age:` | numeric (years) |
| 36 | `Kategori:` | `HC`, `MCI`, `AD`, `non-AD` |

**Target words (Gold):**
säng, penna, visselpipa, kam, såg, helikopter, bläckfisk, galge, kamel, kringla, racket, vulkan, pil, jordglob, bäver, noshörning, iglo, domino, rulltrappa, hängmatta, pelikan, pyramid, passare, dragspel, sparris, lås, ok, sfinx, spalje, gradskiva

---

### svf_v1.xlsx — Semantic Verbal Fluency

Participants name animals within time limit; one word per row.

| Column | Type | Description |
|--------|------|-------------|
| `Unnamed: 0` | — | Row index (empty) |
| `User-1` ... `User-100` | string | Animal names or empty |

**Metadata rows:**
| Row | Field | Values |
|-----|-------|--------|
| 28 | `Gender:` | `M`, `F` |
| 29 | `Age:` | numeric (years) |
| 30 | `Category:` | `HC`, `MCI`, `AD`, `non-AD` |

---

### fas_v1.xlsx — Phonemic/Letter Verbal Fluency

Participants produce words starting with F, A, S; stored as comma-separated triplets.

| Column | Type | Description |
|--------|------|-------------|
| `Unnamed: 0` | — | Row index (empty) |
| `User-1` ... `User-100` | string | Triplet: `"F-word, A-word, S-word"` |

**Example:** `"fisk, arbete, sol"`

**Metadata rows:**
| Row | Field | Values |
|-----|-------|--------|
| 19 | `Gender:` | `M`, `F` |
| 20 | `Age:` | numeric (years) |
| 21 | `Category:` | `HC`, `MCI`, `AD`, `non-AD` |

**Note:** Proper nouns/NERs marked with angle brackets: `<Stockholm>`, `<Anna>`

---

## Diagnostic Categories

| Code | Description |
|------|-------------|
| `HC` | Healthy Control |
| `MCI` | Mild Cognitive Impairment |
| `AD` | Alzheimer's Disease |
| `non-AD` | Non-Alzheimer's Dementia |

---

## Missing/Special Values

| Value | Meaning |
|-------|---------|
| *(empty)* | No response / end of list |
| `pass` | Participant skipped item |
| `hhhm jag vet inte` | "I don't know" |
| `vet inte` | "Don't know" |

---

## Directory Structure

```
data/
├── xlsx/          # Original source files
├── processed/     # Pipeline outputs
└── README.md
```

---

## Notes

- Column naming quirks in BNT: `User-2.1`, `User-34.1` exist due to Excel export (100 unique users total)
- FAS empty positions appear as missing elements: `", ansvar, sol"` or `"fisk, , stol"`
- Response data includes realistic errors: semantic paraphasias, circumlocutions, superordinates

---

## Source

Synthetic data generated January 2026 for thesis project "Embedding-Based Graded Scoring of Paraphasic Errors in Neuropsychological Language Tests."

PI: Dimitrios Kokkinakis (dimitrios.kokkinakis@svenska.gu.se)
