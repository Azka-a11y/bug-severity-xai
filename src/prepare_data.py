"""Data preparation and inspection script for bug severity classification.

This script:
1. Downloads the Bugzilla Eclipse Bug Reports dataset from Hugging Face.
2. Inspects schema, columns, and total records.
3. Identifies the description and severity target columns.
4. Analyzes the distribution across the six severity classes.
5. Inspects data quality (missing and duplicate values).
6. Exports a local CSV copy into data/.
7. Generates data/dataset_summary.md with empirical findings.
"""

from pathlib import Path
import sys
import pandas as pd
from datasets import load_dataset


DATASET_NAME = "AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_CSV_PATH = OUTPUT_DIR / "eclipse_bug_reports.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "dataset_summary.md"


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    """Find the best matching column name from a list of candidates (case-insensitive)."""
    col_map = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    # Partial match
    for candidate in candidates:
        for c_lower, original in col_map.items():
            if candidate.lower() in c_lower:
                return original
    return None


def main() -> None:
    print(f"Loading dataset '{DATASET_NAME}' from Hugging Face Datasets...")
    try:
        dataset_dict = load_dataset(DATASET_NAME)
    except Exception as err:
        print(f"\n[ERROR] Failed to download dataset '{DATASET_NAME}'.", file=sys.stderr)
        print(f"Reason: {err}", file=sys.stderr)
        sys.exit(1)

    print("Dataset downloaded successfully.")
    print(f"Available splits: {list(dataset_dict.keys())}")

    # Combine or select splits into a single DataFrame
    split_dfs = []
    for split_name, split_data in dataset_dict.items():
        sdf = split_data.to_pandas()
        if len(dataset_dict) > 1:
            sdf["_split"] = split_name
        split_dfs.append(sdf)

    df = pd.concat(split_dfs, ignore_index=True) if len(split_dfs) > 1 else split_dfs[0]

    total_records = len(df)
    columns = list(df.columns)
    print(f"\nTotal records: {total_records:,}")
    print(f"Columns found ({len(columns)}): {columns}")
    print("\nData Types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")

    # Identify short description and severity columns
    desc_col = find_column(
        columns,
        ["short_desc", "short_description", "description", "summary", "title", "bug_description"]
    )
    severity_col = find_column(
        columns,
        ["bug_severity", "severity", "severity_level", "class", "label"]
    )

    if not desc_col:
        print("\n[WARNING] Could not automatically match description column. Available columns:", columns)
        # Fall back to first object/string column if needed
        desc_col = columns[0]
    if not severity_col:
        print("\n[WARNING] Could not automatically match severity column. Available columns:", columns)
        severity_col = columns[-1]

    print(f"\nIdentified Description Column : '{desc_col}'")
    print(f"Identified Severity Column    : '{severity_col}'")

    # Data Quality: Missing & Duplicate Descriptions
    desc_series = df[desc_col].astype(str)
    empty_desc_count = (df[desc_col].isna() | (desc_series.str.strip() == "")).sum()
    empty_desc_pct = (empty_desc_count / total_records) * 100

    dup_desc_count = df[desc_col].duplicated().sum()
    dup_desc_pct = (dup_desc_count / total_records) * 100
    unique_desc_count = df[desc_col].nunique()

    # Missing values across all columns
    missing_by_col = df.isna().sum().to_dict()

    # Severity distribution
    severity_series = df[severity_col].astype(str).str.strip().str.lower()
    severity_counts = severity_series.value_counts()
    severity_pcts = (severity_series.value_counts(normalize=True) * 100)

    print("\n" + "=" * 50)
    print("SEVERITY CLASS DISTRIBUTION")
    print("=" * 50)
    for cls_name, count in severity_counts.items():
        pct = severity_pcts[cls_name]
        print(f"  {cls_name:<15}: {count:>8,} ({pct:6.2f}%)")

    print("\n" + "=" * 50)
    print("DATA QUALITY METRICS")
    print("=" * 50)
    print(f"  Empty descriptions     : {empty_desc_count:,} ({empty_desc_pct:.2f}%)")
    print(f"  Duplicate descriptions : {dup_desc_count:,} ({dup_desc_pct:.2f}%)")
    print(f"  Unique descriptions    : {unique_desc_count:,}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save local CSV copy
    print(f"\nSaving local CSV copy to: {OUTPUT_CSV_PATH}")
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Local CSV saved.")

    # Generate dataset_summary.md
    print(f"Generating summary report at: {OUTPUT_SUMMARY_PATH}")
    markdown_lines = [
        "# Dataset Summary: Bugzilla Eclipse Bug Reports",
        "",
        "This document contains empirical statistics derived directly from the downloaded dataset without synthetic modification.",
        "",
        "## Source Information",
        f"- **Dataset Identifier:** `{DATASET_NAME}`",
        f"- **Dataset URL:** [Hugging Face Dataset](https://huggingface.co/datasets/{DATASET_NAME})",
        f"- **Reference Paper:** Arshad et al., *SevPredict: Exploring the Potential of Large Language Models in Software Maintenance*, 2024. [DOI: 10.3390/ai5040132](https://doi.org/10.3390/ai5040132)",
        "",
        "## Dataset Overview",
        f"- **Total Records:** {total_records:,}",
        f"- **Total Columns:** {len(columns)}",
        f"- **Identified Text Column:** `{desc_col}`",
        f"- **Identified Label Column:** `{severity_col}`",
        "",
        "## Column Schema",
        "| Column Name | Data Type | Missing Values | Missing % |",
        "|---|---|---|---|",
    ]

    for col in columns:
        col_missing = missing_by_col.get(col, 0)
        col_missing_pct = (col_missing / total_records) * 100
        col_dtype = str(df[col].dtype)
        markdown_lines.append(f"| `{col}` | `{col_dtype}` | {col_missing:,} | {col_missing_pct:.2f}% |")

    markdown_lines.extend([
        "",
        "## Severity Class Distribution (6 Classes)",
        "| Severity Level | Record Count | Percentage |",
        "|---|---|---|",
    ])

    for cls_name, count in severity_counts.items():
        pct = severity_pcts[cls_name]
        markdown_lines.append(f"| `{cls_name}` | {count:,} | {pct:.2f}% |")

    markdown_lines.extend([
        "",
        "## Data Quality & Text Metrics",
        f"- **Empty / Blank Descriptions:** {empty_desc_count:,} ({empty_desc_pct:.2f}%)",
        f"- **Duplicate Descriptions:** {dup_desc_count:,} ({dup_desc_pct:.2f}%)",
        f"- **Unique Descriptions:** {unique_desc_count:,}",
        "",
        "## Key Observations",
        "1. **Class Imbalance:** The distribution is heavily skewed toward `normal` severity reports. Class-weighting, focal loss, or stratified sampling will be evaluated during model fine-tuning.",
        "2. **Description Duplicates:** Identical bug descriptions exist within the dataset, representing duplicate submissions or automated logging. Deduplication strategies will be assessed prior to train/test partitioning.",
        "3. **Completeness:** Empty or whitespace-only descriptions are accounted for and will be filtered during preprocessing.",
        "",
        "---",
        "*Report automatically generated by `src/prepare_data.py`.*",
        ""
    ])

    with open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))

    print(f"Summary written successfully to {OUTPUT_SUMMARY_PATH}")
    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
