## Embedding-Based Graded Scoring of Paraphasic Errors in Neuropsychological Language Tests
Thesis project, AI and Language, Stockholm University, Spring 2026

### Pipelines

**BNT** (confrontation naming):
```bash
python bnt_pipeline.py --data data/xlsx/BNT-syntheticData_v2.xlsx [--mock]
```

**SVF** (semantic verbal fluency — animal naming):
```bash
python svf_pipeline.py --data data/xlsx/SVF-syntheticData_v1.xlsx [--mock]
```
Computes: total/unique word counts, repetitions, consecutive cosine similarities
(KB-BERT), pairwise semantic diversity, and temporal gradient.

**FAS** (phonemic verbal fluency — words starting with F, A, S):
```bash
python fas_pipeline.py --data data/xlsx/FAS-syntheticData_v1.xlsx
```
Computes: per-letter total/valid/proper-noun/repetition counts, total FAS score,
letter asymmetry, and mean normalised Levenshtein distance between consecutive
responses per letter.

All pipelines save CSV results to `data/processed/` and print a per-diagnosis
summary to stdout.

