# Data README

**Title:** Synthetic Neuropsychological Language Assessment Data for Embedding-Based Paraphasia Scoring

------

## General Information

### Project Context

This dataset supports the Master's thesis project "Embedding-Based Graded Scoring of Paraphasic Errors in Neuropsychological Language Tests Using Synthetic Data," conducted at Stockholm University (AI and Language, Spring 2026). The project is associated with the research initiative "AI-driven Language Biomarkers for Early Detection and Progression of Cognitive Decline."

### Principal Investigators and Contacts

- **External Supervisor / PI:** Dimitrios Kokkinakis, University of Gothenburg (dimitrios.kokkinakis@svenska.gu.se)
- **Co-supervisor:** Charalambos Themistocleous, University of Oslo (charalampos.themistokleous@isp.uio.no)
- **SU Supervisor:** Dana Dannélls, University of Gothenburg / Stockholm University (dana.dannells@ling.su.se)
- **Student Researcher:** Fredrik Bogind

### Data Generation Date

2026-01 (January 2026)

### Geographic and Linguistic Context

The synthetic data simulates Swedish-language responses from a population resembling participants in the [Gothenburg H70 Birth Cohort Study](https://www.gu.se/en/research/the-gothenburg-h70-birth-cohort-study). All response data is in **Swedish**.

### Keywords

neuropsychological assessment, confrontation naming, verbal fluency, paraphasia, Boston Naming Test, semantic verbal fluency, phonemic fluency, FAS, synthetic data, Swedish NLP, cognitive decline, dementia, Alzheimer's disease, mild cognitive impairment

### Language

- Data content: Swedish
- Documentation: English

------

## Data and File Overview

### Directory Structure

```
data/
├── csv/                          # Processed CSV files (primary analysis format)
│   ├── bnt_v2.csv
│   ├── fas_v1.csv
│   └── svf_v1.csv
├── processed/                    # Output directory for pipeline-generated files
├── xlsx/                         # Original Excel files (source format)
│   ├── BNT-syntheticData_v1.xlsx
│   ├── BNT-syntheticData_v2.xlsx
│   ├── FAS-syntheticData_v1.xlsx
│   └── SVF-syntheticData_v1.xlsx
└── README.md                     # This file
```

### File Descriptions

| Filename     | Description                                 | Format      | Rows | Columns | Version |
| ------------ | ------------------------------------------- | ----------- | ---- | ------- | ------- |
| `bnt_v2.csv` | Boston Naming Test responses                | CSV (UTF-8) | 35   | 102     | v2      |
| `svf_v1.csv` | Semantic Verbal Fluency (animals) responses | CSV (UTF-8) | 29   | 101     | v1      |
| `fas_v1.csv` | Phonemic/Letter Verbal Fluency responses    | CSV (UTF-8) | 20   | 101     | v1      |

### Version History

- **2026-01:** Initial synthetic data generation (v1 for all tests; v2 for BNT with refinements)
- The `xlsx/` directory contains original source files; `csv/` contains converted files for analysis

------

## Methodological Information

### Data Generation Method

The data was synthetically generated using a large language model (LLM) to simulate realistic participant responses to standardized neuropsychological language assessments. The synthetic generation was designed to:

[Add description]

**Important:** This is entirely synthetic data created for methodology development and validation purposes. No real patient data is included.

### Source Population Model

The synthetic data is modeled after the Gothenburg H70 Birth Cohort Study population, a longitudinal study of aging and cognitive health in Swedish adults.

### Data Processing

1. Original data generated in XLSX format
2. Converted to CSV (UTF-8 encoding) for analysis compatibility
3. No additional preprocessing applied to raw responses in the archived files

## Data-Specific Information

### 1. Boston Naming Test (BNT) — `bnt_v2.csv`

#### Test Description

The Boston Naming Test is a standardized confrontation naming assessment where participants are shown pictures and asked to name them. The Swedish version used here contains 30 items of increasing difficulty. It is widely used in clinical and research settings to assess lexical retrieval abilities and detect language impairments associated with aphasia and neurodegenerative disorders.

#### File Structure

- **Rows 1–31:** Test items and participant responses
- **Row 34:** Gender metadata
- **Row 35:** Age metadata
- **Row 36:** Diagnostic category metadata

#### Variable Definitions

| Column                 | Description                    | Values/Format                                      |
| ---------------------- | ------------------------------ | -------------------------------------------------- |
| `Gold`                 | Target word (correct response) | Swedish noun (e.g., "säng", "penna", "helikopter") |
| `Unnamed: 1`           | Empty/unused                   | —                                                  |
| `User-1` to `User-100` | Participant responses          | Swedish text (see Response Types below)            |
| `Gender:` (row 34)     | Participant gender             | `M` = Male, `F` = Female                           |
| `Age:` (row 35)        | Participant age                | Numeric (years, e.g., 62.5)                        |
| `Kategori:` (row 36)   | Diagnostic category            | See Category Codes below                           |

#### Target Words (Gold Standard)

The 30 BNT items in this Swedish version (items visible in data): säng, penna, visselpipa, kam, såg, helikopter, bläckfisk, galge, kamel, kringla, trädgårdslang, mask, näbbdjur, harpun, bäverhona, pegasus, enhörning, sjöhäst, traktor, ankare, ekorre, sko, vulkan, pyramid, mumie, kikare, glob, harpa, lås, ok, sfinx, spalje, gradskiva

#### Response Types in BNT Data

Participant responses fall into several categories relevant for paraphasia analysis:

| Response Type            | Description                                 | Examples                                    |
| ------------------------ | ------------------------------------------- | ------------------------------------------- |
| **Correct**              | Exact match or acceptable variant of target | "säng", "en säng"                           |
| **Semantic paraphasia**  | Semantically related word substitution      | "bädd" for säng, "flöjt" for visselpipa     |
| **Circumlocution**       | Descriptive phrase instead of target word   | "något som flyger" for helikopter           |
| **Superordinate**        | Category-level response                     | "verktyg" for såg, "havsdjur" for bläckfisk |
| **Visual misperception** | Response based on visual confusion          | "häst kanske" for kamel                     |
| **Phonemic paraphasia**  | Sound-based error                           | (rare in this dataset)                      |
| **Unrelated error**      | Response with no clear relation to target   | "sak" for various items                     |
| **Don't know / Pass**    | Explicit non-response                       | "pass", "hhhm jag vet inte", "vet inte"     |
| **Compound response**    | Multiple attempts or elaborations           | "bläckfisk är det", "jag tror helikopter"   |
| **Determiner-prefixed**  | Response with article                       | "en säng", "ett havsdjur"                   |

------

### 2. Semantic Verbal Fluency (SVF) — `svf_v1.csv`

#### Test Description

Semantic Verbal Fluency (category fluency) assesses the ability to generate words within a semantic category under time constraints. In this version, participants are asked to name as many **animals** as possible within a fixed time period (typically 60 seconds). The test measures semantic memory, lexical retrieval, and executive function.

#### File Structure

- **Rows 1–26:** Participant responses (one word per row per participant)
- **Row 28:** Gender metadata
- **Row 29:** Age metadata
- **Row 30:** Category metadata

#### Variable Definitions

| Column                 | Description           | Values/Format                    |
| ---------------------- | --------------------- | -------------------------------- |
| `Unnamed: 0`           | Row index/empty       | —                                |
| `User-1` to `User-100` | Participant responses | Swedish animal names (or errors) |
| `Gender:` (row 28)     | Participant gender    | `M` = Male, `F` = Female         |
| `Age:` (row 29)        | Participant age       | Numeric (years)                  |
| `Category:` (row 30)   | Diagnostic category   | See Category Codes below         |

#### Response Types in SVF Data

| Response Type          | Description                   | Examples                                                 |
| ---------------------- | ----------------------------- | -------------------------------------------------------- |
| **Correct animal**     | Valid animal name             | "hund", "katt", "elefant"                                |
| **Diminutive/variant** | Variant forms                 | "hundvalp", "kisse", "vovve", "kattunge"                 |
| **Perseveration**      | Repeated response             | Same animal appearing multiple times for one participant |
| **Intrusion**          | Non-animal response           | "cykel", "stol"                                          |
| **Verbal comment**     | Meta-response or hesitation   | "nu blev det slut", "kan inga mer", "ja vet inte"        |
| **Phonemic cluster**   | Phonemically similar sequence | Multiple animals starting with same sound                |
| **Semantic cluster**   | Semantically grouped sequence | Farm animals, sea creatures in sequence                  |
| **Empty cell**         | No further responses          | (blank)                                                  |

#### Notes on SVF Scoring

- Traditional scoring counts unique correct animals
- Embedding-based analysis may examine: semantic clustering, intrusion similarity, response progression patterns

------

### 3. Phonemic/Letter Verbal Fluency (FAS) — `fas_v1.csv`

#### Test Description

Phonemic (or Letter) Verbal Fluency assesses the ability to generate words beginning with specific letters. The FAS test uses three letters: **F**, **A**, and **S**. Participants produce as many words as possible starting with each letter within a time limit. This test primarily measures executive function and phonemic retrieval strategies.

#### File Structure

- **Rows 1–17:** Participant responses (triplet format, one set per row)
- **Row 19:** Gender metadata
- **Row 20:** Age metadata
- **Row 21:** Category metadata

#### Variable Definitions

| Column                 | Description           | Values/Format                              |
| ---------------------- | --------------------- | ------------------------------------------ |
| `Unnamed: 0`           | Row index/empty       | —                                          |
| `User-1` to `User-100` | Participant responses | Comma-separated triplet (see format below) |
| `Gender:` (row 19)     | Participant gender    | `M` = Male, `F` = Female                   |
| `Age:` (row 20)        | Participant age       | Numeric (years)                            |
| `Category:` (row 21)   | Diagnostic category   | See Category Codes below                   |

#### Response Format

Each cell contains a triplet of words separated by commas, representing one word from each letter condition:

```
"F-word, A-word, S-word"
```

Example: `"fägring, ansvar, sjudning"` where:

- "fägring" = F-word
- "ansvar" = A-word
- "sjudning" = S-word

#### Response Types in FAS Data

| Response Type        | Description                                    | Examples                                  |
| -------------------- | ---------------------------------------------- | ----------------------------------------- |
| **Correct word**     | Valid Swedish word starting with target letter | "fisk", "arbete", "sol"                   |
| **Proper noun**      | Names (often marked with angle brackets)       | `<Stockholm>`, `<Anna>`, `<Fredrik>`      |
| **Rule violation**   | Word not starting with target letter           | (rare, would appear misplaced in triplet) |
| **Repetition**       | Same word repeated across rows                 | "frost" appearing multiple times          |
| **Missing response** | Empty position in triplet                      | ", ansvar, sol" or "fisk, , stol"         |
| **Partial response** | Incomplete triplet                             | Fewer than 3 words in later rows          |

#### Special Notation

- `<word>` — Angle brackets indicate proper nouns (names, places) which may be scored differently depending on protocol

------

## Participant Metadata

### Category Codes (Diagnostic Classification)

| Code     | Description                                                  |
| -------- | ------------------------------------------------------------ |
| `HC`     | Healthy Control — cognitively healthy participants           |
| `MCI`    | Mild Cognitive Impairment — early-stage cognitive decline    |
| `AD`     | Alzheimer's Disease — dementia of Alzheimer's type           |
| `non-AD` | Non-Alzheimer's Dementia — other dementia types (e.g., vascular, frontotemporal) |

### Demographic Summary

| Dataset | N    | Age Range | Gender Distribution |
| ------- | ---- | --------- | ------------------- |
| BNT     | 100  | 55.2–80.0 | ~50% M / ~50% F     |
| SVF     | 100  | 55.3–79.8 | ~50% M / ~50% F     |
| FAS     | 100  | 50.1–77.8 | ~50% M / ~50% F     |

------

## Missing Data Codes

| Representation                  | Meaning                                                      |
| ------------------------------- | ------------------------------------------------------------ |
| Empty cell / blank              | No response provided (end of fluency list, or unable to respond) |
| `pass`                          | Participant explicitly passed/skipped item (BNT)             |
| `hhhm jag vet inte`             | Participant expressed uncertainty ("hmm I don't know")       |
| `vet inte` / `vet inte riktigt` | "Don't know" responses                                       |
| Partial triplet in FAS          | Participant stopped producing words for one or more letters  |

------

## Sharing and Access Information

### License

This synthetic dataset is created for academic research purposes. Usage terms to be determined by the research project PIs.

### Restrictions

- Data is synthetic and contains no real patient information
- BNT test materials (images) are copyright-protected; this dataset contains only response data
- For questions about data use, contact the PI: Dimitrios Kokkinakis

### Related Publications

This dataset supports methodology development described in the thesis project. Key background references:

- Tallberg (2005). The Boston Naming Test in Swedish: Normative data. *Brain and Language, 94*, 19–31.
- Salem et al. (2022). Refining Semantic Similarity of Paraphasias Using a Contextual Language Model. *JSLHR, 66*, 206–220.
- Fergadiotis et al. (2016). Algorithmic Classification of Five Characteristic Types of Paraphasias. *AJSLP, 25*, S776–S787.

### Recommended Citation

```
Kokkinakis, D. (2026). Synthetic Neuropsychological Language Assessment 
Data for Embedding-Based Paraphasia Scoring [Data set]. University of Gothenburg / 
```

------

## Quality Assurance Notes

### Known Data Characteristics

1. **Column naming inconsistencies:** BNT file contains duplicate-style column names (e.g., "User-2.1", "User-34.1") likely from Excel conversion; actual user count is 100
2. **Proper noun marking:** FAS data uses `<brackets>` to mark proper nouns; preprocessing should handle these appropriately
3. **Response variation:** Intentionally includes realistic variation in response style (determiner usage, compound responses, hesitation markers)

### Validation Status

- Synthetic generation reviewed for linguistic plausibility
- Full validation against clinical scoring standards is part of thesis methodology development

------

## File Format Specifications

### CSV Files

- **Encoding:** UTF-8
- **Delimiter:** Comma (`,`)
- **Quote character:** Double quote (`"`) for fields containing commas
- **Line endings:** Unix-style (LF)

### Excel Files

- **Format:** XLSX (Office Open XML)
- **Preserved for:** Source reference and potential formatting information

------

## Change Log

| Date       | Version | Description                                  | Author        |
| ---------- | ------- | -------------------------------------------- | ------------- |
| 2026-01    | v1      | Initial synthetic data generation (SVF, FAS) | D. Kokkinakis |
| 2026-01-30 | v2      | Revised BNT data with refinements            | D. Kokkinakis |
| 2026-02-02 | —       | README documentation created                 | F. Boglind    |

------

## Contact

For questions about this dataset or the associated research project:

**Primary Contact:**
 Dimitrios Kokkinakis
 University of Gothenburg, Språkbanken Text
 dimitrios.kokkinakis@svenska.gu.se

------

*This README follows guidelines adapted from Cornell Data Services: [Guide to Writing READMEs](https://data.research.cornell.edu/data-management/sharing/readme/)*