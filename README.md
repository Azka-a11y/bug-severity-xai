# Explainable Bug Severity Prediction Using DistilBERT

A Master's research project investigating multi-class bug severity classification on Bugzilla bug reports using fine-tuned transformer architectures (DistilBERT) and post-hoc model explainability (SHAP).

---

## Problem Statement

Accurate bug severity assessment is crucial for software triage, resource allocation, and timely bug resolution in large-scale software engineering projects. When bug reports arrive in high volumes, manual severity assignment can be inconsistent, slow, and subjective. Automated severity prediction using natural language processing helps triage teams prioritize defects according to their real impact.

## Base Research Paper

This work builds upon and extends the research conducted in:

- **Paper:** *SevPredict: Exploring the Potential of Large Language Models in Software Maintenance*
- **Authors:** Ali Arshad et al. (2024)
- **Publication:** *AI*, 5(4), 132
- **DOI:** [10.3390/ai5040132](https://doi.org/10.3390/ai5040132)

### Research Extension
The base paper focuses primarily on binary bug severity prediction (e.g., severe vs. non-severe). This project extends that formulation to the full **six-class severity taxonomy**:
1. `blocker`
2. `critical`
3. `major`
4. `normal`
5. `minor`
6. `trivial`

To balance classification performance with computational efficiency, we employ **DistilBERT**, a distilled and lightweight transformer model. Furthermore, we incorporate post-hoc explainable AI (XAI) using **SHAP (SHapley Additive exPlanations)** to interpret token-level feature attribution and understand which textual patterns influence severity decisions.

---

## Dataset

We use the public Bugzilla Eclipse bug reports dataset published by the base paper's authors:

- **Source:** [AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset](https://huggingface.co/datasets/AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset) on Hugging Face
- **Domain:** Eclipse open-source bug reports extracted from Bugzilla
- **Size:** Approximately 88,682 bug reports labeled across the six severity classes

Detailed empirical schema and distribution metrics derived from the raw data are tracked in [`data/dataset_summary.md`](data/dataset_summary.md).

---

## Current Project Status

- [x] Initial repository setup and dependency specification
- [x] Automated dataset download and inspection pipeline (`src/prepare_data.py`)
- [x] Initial dataset inspection and empirical summary (`data/dataset_summary.md`)
- [x] Detailed offline EDA, quality/leakage checks, tables, and Matplotlib plots (`src/explore_data.py`)
- [ ] Text cleaning, tokenization, and stratified train/validation/test split
- [ ] DistilBERT model fine-tuning for 6-class classification
- [ ] Baseline model evaluation (macro/weighted F1, precision, recall)
- [ ] SHAP attribution and explainability analysis

---

## Repository Structure

```text
bug-severity-xai/
├── data/
│   ├── .gitkeep
│   └── dataset_summary.md       # Empirical dataset statistics and schema
├── src/
│   ├── prepare_data.py          # Data download, inspection, and summary generator
│   └── explore_data.py          # Detailed offline EDA and Matplotlib plots
├── reports/eda/                 # Generated EDA report, tables, figures (git-ignored)
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Setup & Execution (Windows)

### 1. Prerequisites
- Python 3.10+ (tested with Python 3.13)
- Internet connection (to download dataset from Hugging Face)

### 2. Virtual Environment Setup

Open PowerShell or Command Prompt in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run Dataset Preparation & Inspection

```powershell
python src/prepare_data.py
```

This script will:
1. Download the raw dataset from Hugging Face into cache.
2. Inspect schema, record counts, and column types.
3. Verify class distribution across all six severity categories.
4. Export a local CSV copy to `data/eclipse_bug_reports.csv` (git-ignored).
5. Generate an updated empirical report at `data/dataset_summary.md` (git-tracked).

### 5. Run Detailed Exploratory Data Analysis

After preparing the CSV, run:

```powershell
python src/explore_data.py
```

The command reads the local CSV without network access or changes to the source.
It creates `reports/eda/report.md` with embedded Matplotlib figures, `metadata.json`
with the input hash and run configuration, `row_features.csv` for record-level
auditing, and complete CSV tables in `reports/eda/tables/`.

The analysis covers:

- All six severity counts, missing/unknown labels, class imbalance, exploratory
  class weights, and an always-majority reference baseline.
- Column completeness, exact duplicate rows/texts, normalized duplicate text,
  conflicting labels, and candidate `(Project, Bug ID)` collisions.
- Character and whitespace-word distributions, quantiles, outliers, and word
  budget thresholds overall and by severity.
- Project, resolution, and existing split class composition when present,
  plus normalized-text overlap across existing splits.
- Unigram/bigram document frequencies, frequent terms by severity, and
  smoothed class-versus-rest term enrichment.

Matplotlib PNGs are saved under `reports/eda/figures/`: severity distribution,
full-range text-length histograms, word-count boxplots by class, quality indicators,
severity mix by available metadata fields, frequent unigrams/bigrams, and term
prevalence by class. Empty or missing classes remain visible in the outputs.

Customize the input, output, column mapping, or lexical thresholds:

```powershell
python src/explore_data.py --input data/eclipse_bug_reports.csv --output-dir reports/eda --top-n 15 --min-df 10
python src/explore_data.py --input data/custom.csv --text-column "Summary" --label-column "Severity"
```

Blank text is excluded from length and duplicate-text statistics. Labels are
trimmed and lowercased for analysis; invalid labels remain visible in quality
and segment tables. Word counts are **not** DistilBERT tokenizer lengths, and
term associations are **not** SHAP attributions. Full-dataset EDA weights and
vocabulary statistics must be recomputed on training data before modeling.
No train/test partition or preprocessing changes are made by this command.

---

## Planned Next Steps

1. **Preprocessing Pipeline:** Address empty descriptions and evaluate duplicate bug reports before train/test partitioning.
2. **Class Imbalance Handling:** Explore class weights and focal loss to handle the predominance of normal severity reports.
3. **Model Fine-Tuning:** Fine-tune `distilbert-base-uncased` with sequence classification head for 6 classes.
4. **Explainability Experiments:** Integrate SHAP text explainer to compute token-level attribution scores on validation predictions.
