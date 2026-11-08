"""Reproducible, offline EDA for the six-class bug severity dataset.

Run ``python src/explore_data.py --help`` for the command-line interface.
The input is never modified. Tables use full-data counts; no model is fitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")  # Also works on CI and machines without a display.
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEVERITIES = ("blocker", "critical", "major", "normal", "minor", "trivial")
COLORS = dict(zip(SEVERITIES, ("#73395D", "#A44864", "#C97835", "#356C9B", "#7B813D", "#677785")))
ALIASES = {
    "text": ("Short Description", "short_desc", "short_description", "description", "summary", "title", "bug_description"),
    "label": ("Severity Label", "bug_severity", "severity", "severity_level", "class", "label"),
    "project": ("Project", "product"),
    "bug_id": ("Bug ID", "bug_id", "id"),
    "resolution": ("Resolution Status", "resolution_status", "resolution"),
    "split": ("_split", "split"),
}
# A deliberately small, inspectable English stop list. This is EDA tokenization,
# not the training tokenizer; technical words and negation are retained.
STOP_WORDS = set("a an the and or but if then to of in on at by for from with as is are was were be been being it its this that these those i you he she we they my your our their me us them can could would should will may might do does did have has had".split())
WORD_RE = re.compile(r"[a-z][a-z0-9_]+")


@dataclass
class Analysis:
    metrics: dict
    tables: dict[str, pd.DataFrame]
    rows: pd.DataFrame


def canonical(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", value.strip().casefold())


def resolve_column(df: pd.DataFrame, role: str, explicit: str | None = None) -> str | None:
    """Resolve aliases without silently guessing the first/last input column."""
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f"Column {explicit!r} does not exist. Available: {list(df.columns)}")
        return explicit
    for alias in ALIASES[role]:
        matches = [c for c in df.columns if canonical(c) == canonical(alias)]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous {role} columns: {matches}. Specify a column explicitly.")
        if matches:
            return matches[0]
    if role in ("text", "label"):
        raise ValueError(f"Cannot identify {role} column; pass --{role}-column. Available: {list(df.columns)}")
    return None


def strings(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("")


def normalize_text(series: pd.Series) -> pd.Series:
    """Conservative duplicate key: Unicode NFKC, casefold, collapsed whitespace.

    Punctuation and numbers remain significant. This does not detect semantic
    near-duplicates and must not be treated as a model preprocessing recipe.
    """
    return strings(series).str.normalize("NFKC").str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()


def duplicate_groups(keys: pd.Series, labels: pd.Series) -> pd.DataFrame:
    data = pd.DataFrame({"key": keys, "label": labels})
    # Aggregate only repeated keys, avoiding Python callbacks for every unique
    # description in the corpus (the overwhelming majority of source rows).
    data = data.loc[data["key"].ne("") & data["key"].duplicated(keep=False)]
    grouped = data.groupby("key", sort=True)["label"].agg(
        rows="size", distinct_labels=lambda s: s.loc[s.ne("")].nunique(),
        labels=lambda s: ", ".join(sorted(s.loc[s.ne("")].unique())),
    ).reset_index()
    grouped = grouped.loc[grouped["rows"].gt(1)].copy()
    grouped["text_sha256"] = grouped["key"].map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    return grouped.drop(columns="key").sort_values(["rows", "text_sha256"], ascending=[False, True]).reset_index(drop=True)


def describe_lengths(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups = [("ALL_NONBLANK", rows.loc[~rows["blank_text"]])]
    groups += [(label, rows.loc[rows["label"].eq(label) & ~rows["blank_text"]]) for label in SEVERITIES]
    for label, group in groups:
        for feature in ("characters", "words"):
            values = group[feature]
            record = {"severity": label, "feature": feature, "count": len(values)}
            record.update({"mean": values.mean(), "std": values.std(), "min": values.min(),
                           "p25": values.quantile(.25), "median": values.median(),
                           "p75": values.quantile(.75), "p90": values.quantile(.90),
                           "p95": values.quantile(.95), "p99": values.quantile(.99), "max": values.max()})
            records.append(record)
    return pd.DataFrame(records)


def lexical_tables(text: pd.Series, labels: pd.Series, top_n: int, min_df: int) -> dict[str, pd.DataFrame]:
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()
    by_class = {label: Counter() for label in SEVERITIES}
    class_docs: Counter = Counter()
    valid_corpus: Counter = Counter()
    total_docs = 0
    for value, label in zip(text, labels):
        if not value.strip():
            continue
        total_docs += 1
        tokens = WORD_RE.findall(value.casefold())
        terms = {token for token in tokens if token not in STOP_WORDS}
        pairs = {f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a not in STOP_WORDS and b not in STOP_WORDS}
        unigrams.update(terms)
        bigrams.update(pairs)
        if label in by_class:
            by_class[label].update(terms)
            class_docs[label] += 1
            valid_corpus.update(terms)

    def ranked(counter: Counter) -> pd.DataFrame:
        values = sorted(((term, count) for term, count in counter.items() if count >= min_df), key=lambda x: (-x[1], x[0]))[:top_n]
        return pd.DataFrame([{"term": term, "documents": count, "documents_pct": 100 * count / total_docs} for term, count in values], columns=["term", "documents", "documents_pct"])

    records, distinct = [], []
    valid_docs = sum(class_docs.values())
    for label in SEVERITIES:
        n = class_docs[label]
        for term, count in sorted(by_class[label].items(), key=lambda x: (-x[1], x[0])):
            if count >= min_df:
                records.append({"severity": label, "term": term, "documents": count,
                                "class_documents": n, "within_class_pct": 100 * count / n})
        candidates = []
        rest_n = valid_docs - n
        if n and rest_n:
            for term, count in by_class[label].items():
                if count < min_df:
                    continue
                rest_count = valid_corpus[term] - count
                score = math.log2(((count + 1) / (n + 2)) / ((rest_count + 1) / (rest_n + 2)))
                candidates.append({"severity": label, "term": term, "documents": count,
                                   "within_class_pct": 100 * count / n, "rest_documents": rest_count,
                                   "rest_pct": 100 * rest_count / rest_n, "log2_smoothed_prevalence_ratio": score})
        distinct.extend(sorted(candidates, key=lambda r: (-r["log2_smoothed_prevalence_ratio"], r["term"]))[:top_n])
    per_class = pd.DataFrame(records, columns=["severity", "term", "documents", "class_documents", "within_class_pct"])
    if not per_class.empty:
        per_class = per_class.groupby("severity", sort=False).head(top_n).reset_index(drop=True)
    return {"top_unigrams": ranked(unigrams), "top_bigrams": ranked(bigrams), "terms_by_severity": per_class,
            "distinctive_terms": pd.DataFrame(distinct, columns=["severity", "term", "documents", "within_class_pct", "rest_documents", "rest_pct", "log2_smoothed_prevalence_ratio"])}


def analyze(df: pd.DataFrame, *, text_column: str | None = None, label_column: str | None = None,
            top_n: int = 20, min_df: int = 5) -> Analysis:
    if df.empty:
        raise ValueError("Input contains no records.")
    if top_n < 1 or min_df < 1:
        raise ValueError("top_n and min_df must be positive integers.")
    df = df.reset_index(drop=True)
    columns = {role: resolve_column(df, role, explicit) for role, explicit in
               (("text", text_column), ("label", label_column), ("project", None),
                ("bug_id", None), ("resolution", None), ("split", None))}
    if columns["text"] == columns["label"]:
        raise ValueError("Text and label columns must be different.")
    text = strings(df[columns["text"]])
    labels = strings(df[columns["label"]]).str.strip().str.casefold()
    normalized = normalize_text(text)
    blank = text.str.strip().eq("")
    valid = labels.isin(SEVERITIES)
    rows = pd.DataFrame({"source_record": df.index + 1, "label": labels, "valid_label": valid,
                         "blank_text": blank, "characters": text.str.len(), "words": text.str.split().str.len(),
                         "exact_text_sha256": text.where(~blank, "").map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest() if s else ""),
                         "normalized_text_sha256": normalized.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest() if s else "")})
    quality = []
    for column in df:
        series = strings(df[column])
        missing = int(df[column].isna().sum())
        blanks = int((df[column].notna() & series.str.strip().eq("")).sum())
        quality.append({"column": column, "dtype": str(df[column].dtype), "nulls": missing,
                        "blanks": blanks, "missing_or_blank_pct": 100 * (missing + blanks) / len(df),
                        "distinct_nonblank": series.loc[series.str.strip().ne("")].nunique()})
    counts = labels.loc[valid].value_counts().reindex(SEVERITIES, fill_value=0)
    valid_n = int(valid.sum())
    class_counts = pd.DataFrame({"severity": SEVERITIES, "records": counts.values})
    class_counts["all_rows_pct"] = 100 * class_counts["records"] / len(df)
    class_counts["valid_labels_pct"] = 100 * class_counts["records"] / valid_n if valid_n else 0.0
    # Descriptive weights only. Recompute on training data after the split.
    class_counts["exploratory_balanced_weight"] = [valid_n / (len(SEVERITIES) * n) if n else float("nan") for n in counts]
    observed = labels.value_counts(dropna=False).rename_axis("normalized_label").reset_index(name="records")
    observed["valid"] = observed["normalized_label"].isin(SEVERITIES)
    observed["all_rows_pct"] = 100 * observed["records"] / len(df)
    exact = duplicate_groups(text.where(~blank, ""), labels)
    normalized_duplicates = duplicate_groups(normalized, labels)
    conflicts = normalized_duplicates.loc[normalized_duplicates["distinct_labels"].gt(1)].copy()
    positive = counts[counts.gt(0)]
    majority = str(counts.idxmax()) if valid_n else None
    majority_share = float(counts.max() / valid_n) if valid_n else None
    metrics = {"records": len(df), "columns": columns, "valid_label_rows": valid_n,
               "missing_label_rows": int(labels.eq("").sum()), "unknown_label_rows": int((~valid & labels.ne("")).sum()),
               "blank_text_rows": int(blank.sum()), "exact_duplicate_rows_excess": int(df.duplicated().sum()),
               "exact_text_duplicate_groups": len(exact), "exact_text_duplicate_rows_excess": int((exact["rows"] - 1).sum()),
               "normalized_duplicate_groups": len(normalized_duplicates),
               "normalized_duplicate_rows_affected": int(normalized_duplicates["rows"].sum()),
               "normalized_duplicate_rows_excess": int((normalized_duplicates["rows"] - 1).sum()),
               "conflicting_text_groups": len(conflicts), "conflicting_text_rows": int(conflicts["rows"].sum()),
               "absent_classes": counts.index[counts.eq(0)].tolist(), "majority_class": majority,
               "majority_accuracy": majority_share,
               "majority_macro_f1_six_classes": (2 * majority_share / (1 + majority_share) / 6) if valid_n else None,
               "largest_to_smallest_observed_class_ratio": float(positive.max() / positive.min()) if len(positive) else None,
               "top_n": top_n, "min_document_frequency": min_df,
               "nonblank_text_rows": int((~blank).sum())}
    thresholds = []
    for label in ("ALL_NONBLANK", *SEVERITIES):
        subset = rows.loc[~blank & (labels.eq(label) if label != "ALL_NONBLANK" else True)]
        for threshold in (32, 64, 128, 256, 512):
            count = int(subset["words"].gt(threshold).sum())
            thresholds.append({"severity": label, "word_threshold": threshold, "nonblank_rows": len(subset),
                               "rows_above": count, "pct_above": 100 * count / len(subset) if len(subset) else float("nan")})
    quality_flags = pd.DataFrame({"issue": ["Blank text", "Missing label", "Unknown label", "Exact duplicate rows (excess)",
                                           "Normalized duplicate text (affected)", "Conflicting text labels (affected)",
                                           "Nonblank text under 3 words", "Text contains non-ASCII"],
                                 "rows": [int(blank.sum()), metrics["missing_label_rows"], metrics["unknown_label_rows"],
                                          metrics["exact_duplicate_rows_excess"], metrics["normalized_duplicate_rows_affected"],
                                          metrics["conflicting_text_rows"], int((~blank & rows["words"].lt(3)).sum()),
                                          int(text.str.contains(r"[^\x00-\x7f]", regex=True).sum())]})
    quality_flags["all_rows_pct"] = 100 * quality_flags["rows"] / len(df)
    tables = {"schema_quality": pd.DataFrame(quality), "class_distribution": class_counts, "observed_labels": observed,
              "quality_flags": quality_flags, "text_length_summary": describe_lengths(rows),
              "word_length_thresholds": pd.DataFrame(thresholds), "exact_text_duplicates": exact,
              "normalized_text_duplicates": normalized_duplicates, "conflicting_text_labels": conflicts}
    for role in ("project", "resolution", "split"):
        column = columns[role]
        if column is None:
            continue
        group = strings(df[column]).str.strip().replace("", "<missing>")
        rows[role] = group
        # Include invalid/missing labels in counts so segment totals reconcile.
        cross = pd.crosstab(group.rename(role), labels.rename("severity"))
        label_order = list(SEVERITIES) + sorted(set(cross.columns) - set(SEVERITIES))
        cross = cross.reindex(columns=label_order, fill_value=0)
        totals = cross.sum(axis=1)
        order = totals.sort_values(ascending=False, kind="stable").index
        cross = cross.loc[order]
        tables[f"{role}_severity_counts"] = cross.reset_index()
        tables[f"{role}_severity_pct"] = cross.div(cross.sum(axis=1), axis=0).mul(100).reset_index()
    if columns["bug_id"] is not None:
        ids = strings(df[columns["bug_id"]]).str.strip()
        present = ids.ne("")
        metrics["missing_bug_id_rows"] = int((~present).sum())
        metrics["malformed_bug_id_rows"] = int((present & ~ids.str.fullmatch(r"[1-9]\d*")).sum())
        metrics["repeated_bug_id_rows_affected"] = int((present & ids.duplicated(keep=False)).sum())
        keys = pd.DataFrame({"bug_id": ids, "label": labels})
        key_columns = ["bug_id"]
        if "project" in rows:
            keys["project"] = rows["project"]
            present &= strings(df[columns["project"]]).str.strip().ne("")
            key_columns.insert(0, "project")
        repeated_keys = keys.loc[present]
        repeated_keys = repeated_keys.loc[repeated_keys.duplicated(key_columns, keep=False)]
        key_groups = repeated_keys.groupby(key_columns, sort=True).agg(
            rows=("label", "size"), distinct_labels=("label", lambda s: s.loc[s.ne("")].nunique()))
        key_groups = key_groups.loc[key_groups["rows"].gt(1)].reset_index()
        tables["duplicate_bug_keys"] = key_groups
        metrics["bug_key_columns"] = key_columns
        metrics["duplicate_bug_key_groups"] = len(key_groups)
        metrics["duplicate_bug_key_rows_affected"] = int(key_groups["rows"].sum())
    if "split" in rows:
        overlaps = []
        split_names = sorted(rows["split"].unique())
        split_keys = {name: set(normalized.loc[rows["split"].eq(name) & normalized.ne("")]) for name in split_names}
        for left, right in itertools.combinations(split_names, 2):
            shared = split_keys[left] & split_keys[right]
            overlaps.append({"split_a": left, "split_b": right, "shared_normalized_texts": len(shared),
                             "rows_in_a": int((rows["split"].eq(left) & normalized.isin(shared)).sum()),
                             "rows_in_b": int((rows["split"].eq(right) & normalized.isin(shared)).sum())})
        tables["split_text_overlap"] = pd.DataFrame(overlaps, columns=["split_a", "split_b", "shared_normalized_texts", "rows_in_a", "rows_in_b"])
    tables.update(lexical_tables(text, labels, top_n, min_df))
    # Longest observations can be located in the unchanged source by record number.
    tables["longest_texts"] = rows.sort_values(["words", "characters", "source_record"], ascending=[False, False, True]).head(top_n)
    return Analysis(metrics, tables, rows)


def save_plots(analysis: Analysis, directory: Path, top_n: int) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    saved = []

    def save(fig, name):
        fig.savefig(directory / name, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        saved.append(name)

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.constrained_layout.use": True}):
        counts = analysis.tables["class_distribution"]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(counts["severity"], counts["records"], color=[COLORS[s] for s in SEVERITIES])
        ax.bar_label(bars, labels=[f"{n:,}\n({p:.2f}%)" for n, p in zip(counts["records"], counts["valid_labels_pct"])], padding=4)
        ax.set(title="Severity distribution", ylabel="Reports (valid labels only)", ylim=(0, max(1, counts["records"].max()) * 1.20))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        save(fig, "severity_distribution.png")

        nonblank = analysis.rows.loc[~analysis.rows["blank_text"]]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, feature in zip(axes, ("words", "characters")):
            if not nonblank.empty:
                # Full range included; log counts reveal the sparse long tail.
                ax.hist(nonblank[feature], bins=min(60, max(1, nonblank[feature].nunique())), color="#356C9B", log=True)
            else:
                ax.text(.5, .5, "No nonblank text", ha="center", transform=ax.transAxes)
            ax.set(title=f"Text length: {feature}", xlabel=f"{feature.capitalize()} per report", ylabel="Reports (log scale)")
        save(fig, "text_length_distributions.png")

        fig, ax = plt.subplots(figsize=(10, 5))
        present = [s for s in SEVERITIES if ((nonblank["label"] == s).any())]
        if present:
            groups = [nonblank.loc[nonblank["label"].eq(s), "words"] for s in present]
            boxes = ax.boxplot(groups, tick_labels=present, patch_artist=True, showfliers=False)
            for patch, label in zip(boxes["boxes"], present):
                patch.set_facecolor(COLORS[label])
            for line in boxes["medians"]:
                line.set_color("black")
        else:
            ax.text(.5, .5, "No nonblank text with valid labels", ha="center", transform=ax.transAxes)
        ax.set(title="Word counts by severity (outliers hidden; full summaries in tables)", ylabel="Whitespace-separated words per report", ylim=(0, None))
        save(fig, "word_lengths_by_severity.png")

        flags = analysis.tables["quality_flags"].iloc[::-1]
        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.barh(flags["issue"], flags["rows"], color="#356C9B")
        ax.bar_label(bars, labels=[f"{n:,} ({p:.2f}%)" for n, p in zip(flags["rows"], flags["all_rows_pct"])], padding=4)
        ax.set(title="Data-quality indicators (categories can overlap)", xlabel="Rows", xlim=(0, max(1, flags["rows"].max()) * 1.45))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        save(fig, "data_quality.png")

        for role in ("project", "resolution", "split"):
            name = f"{role}_severity_counts"
            if name not in analysis.tables:
                continue
            table = analysis.tables[name].head(top_n).set_index(role)
            totals = table.sum(axis=1)
            proportions = table.div(totals, axis=0).mul(100)
            fig, ax = plt.subplots(figsize=(11, max(4, .4 * len(table) + 1.8)))
            left = pd.Series(0., index=table.index)
            for label in table.columns:
                ax.barh(range(len(table)), proportions[label], left=left, label=label or "<missing label>", color=COLORS.get(label, "#AAAAAA"))
                left += proportions[label]
            ax.set_yticks(range(len(table)), [f"{value} (n={total:,})" for value, total in zip(table.index, totals)])
            ax.invert_yaxis()
            ax.set(title=f"Severity mix by {role} (largest {min(top_n, len(table))} groups)", xlabel="Share of all reports in group", xlim=(0, 100))
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=min(4, len(table.columns)), frameon=False)
            save(fig, f"{role}_severity_mix.png")

        for name, title in (("top_unigrams", "Frequent terms"), ("top_bigrams", "Frequent adjacent word pairs")):
            table = analysis.tables[name].head(top_n).iloc[::-1]
            fig, ax = plt.subplots(figsize=(10, max(4, .32 * len(table) + 1.4)))
            if table.empty:
                ax.text(.5, .5, "No terms meet minimum document frequency", ha="center", transform=ax.transAxes)
            else:
                bars = ax.barh(table["term"], table["documents"], color="#356C9B")
                ax.bar_label(bars, padding=3, fmt="%.0f")
                ax.set_xlim(0, table["documents"].max() * 1.18)
            ax.set(title=title, xlabel="Nonblank reports containing term (document frequency)")
            save(fig, f"{name}.png")

        terms = analysis.tables["terms_by_severity"]
        fig, axes = plt.subplots(3, 2, figsize=(13, 12))
        for ax, label in zip(axes.flat, SEVERITIES):
            table = terms.loc[terms["severity"].eq(label)].head(min(10, top_n)).iloc[::-1]
            if table.empty:
                ax.text(.5, .5, "No eligible terms", ha="center", transform=ax.transAxes)
            else:
                bars = ax.barh(table["term"], table["within_class_pct"], color=COLORS[label])
                ax.bar_label(bars, fmt="%.1f%%", padding=3)
            ax.set(title=label, xlabel="Share of nonblank reports in class", xlim=(0, 100))
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        fig.suptitle("Frequent terms within each severity class")
        save(fig, "terms_by_severity.png")
    return saved


def markdown_table(table: pd.DataFrame) -> str:
    """Avoid the optional tabulate dependency; escape data used as Markdown."""
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:,.2f}"
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ") or "<blank>"
    return "\n".join(["| " + " | ".join(map(str, table.columns)) + " |",
                      "| " + " | ".join(["---"] * len(table.columns)) + " |",
                      *["| " + " | ".join(cell(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)]])


def write_report(analysis: Analysis, output: Path, figures: list[str]) -> None:
    m, t = analysis.metrics, analysis.tables
    lines = ["# Exploratory Data Analysis: Bug Severity", "",
             f"Analyzed **{m['records']:,} source records** without removing or modifying input rows.", "",
             "## Findings and modeling implications", ""]
    if m["valid_label_rows"]:
        lines += [f"- `{m['majority_class']}` accounts for **{100*m['majority_accuracy']:.2f}%** of valid labels. "
                  f"An always-majority predictor has {100*m['majority_accuracy']:.2f}% descriptive accuracy and "
                  f"{m['majority_macro_f1_six_classes']:.4f} macro F1 across the fixed six labels. These are full-dataset reference values, not held-out scores.",
                  f"- Largest/smallest observed class ratio: **{m['largest_to_smallest_observed_class_ratio']:.2f}:1**. "
                  "Report per-class recall and macro F1; recompute any class weights using training labels only."]
    else:
        lines += ["- No valid severity labels were found; class-based modeling is blocked until labels are reviewed."]
    lines += [f"- Completeness and label checks found **{m['blank_text_rows']:,}** blank texts, "
              f"**{m['missing_label_rows']:,}** missing labels, and **{m['unknown_label_rows']:,}** unknown labels.",
              f"- **{m['normalized_duplicate_groups']:,}** normalized duplicate-text groups affect "
              f"**{m['normalized_duplicate_rows_affected']:,}** rows; **{m['normalized_duplicate_rows_excess']:,}** are repeats after the first occurrence. "
              "Keep each group in a single partition to reduce text leakage.",
              f"- **{m['conflicting_text_groups']:,}** duplicate-text groups contain multiple nonmissing labels "
              f"(**{m['conflicting_text_rows']:,}** affected rows). Review them; identical summaries can describe distinct bugs, so automatic relabeling is unsafe.",
              f"- Absent expected classes: {', '.join(m['absent_classes']) or 'none'}.", "",
              ]
    if "project_severity_counts" in t:
        projects = t["project_severity_counts"]
        largest = projects.iloc[0]
        project_n = int(largest.drop(labels="project").sum())
        lines += [f"- The corpus covers **{len(projects)}** project groups. `{largest['project']}` is largest "
                  f"with **{project_n:,}** reports (**{100*project_n/m['records']:.2f}%** of all rows). "
                  "Pooled results may be dominated by this project's language and class mix.", ""]
    if "resolution_severity_counts" in t and len(t["resolution_severity_counts"]) == 1:
        resolution = t["resolution_severity_counts"].iloc[0]["resolution"]
        lines += [f"- All observed resolution values are `{resolution}`. Performance on other resolution populations "
                  "cannot be assessed from this sample; review corpus selection before generalizing to newly filed reports.", ""]
    lines += ["## Schema and completeness", "", markdown_table(t["schema_quality"]), "",
              "Nulls and whitespace-only strings are counted separately. CSV fields are loaded as strings and literal values such as `NA` are preserved.", "",
              "## Labels and imbalance", "", markdown_table(t["class_distribution"]), "",
              "Percentages use either all input rows or only the six valid normalized labels, as named. "
              "Exploratory balanced weight = N_valid / (6 × class_count); absent-class weights are undefined. "
              "These full-data weights must not be used for training.", "", "### All observed labels", "", markdown_table(t["observed_labels"]), "",
              "## Quality and duplicate checks", "", markdown_table(t["quality_flags"]), "",
              f"Exact duplicate rows after the first occurrence: **{m['exact_duplicate_rows_excess']:,}**. "
              f"Exact nonblank duplicate texts: **{m['exact_text_duplicate_groups']:,}** groups, "
              f"**{m['exact_text_duplicate_rows_excess']:,}** repeated rows after the first occurrence.", "",
              "Quality categories overlap and must not be summed. Non-ASCII text and short summaries are review indicators, not proven defects. "
              "Normalized text uses Unicode NFKC, casefolding, whitespace collapsing and trimming; punctuation and numbers remain significant. "
              "Blank texts are excluded from duplicate groups. Conflicts count all distinct nonmissing normalized labels, including unknown labels. "
              "SHA-256 keys in duplicate tables join to the corresponding exact/normalized hash column in `row_features.csv`; "
              "source_record is the one-based parsed data-record position, excluding the header and skipped empty lines.", ""]
    if "duplicate_bug_keys" in t:
        lines += ["### Identifier checks", "",
                  f"Candidate key: `{', '.join(m['bug_key_columns'])}`. Duplicate key groups: **{m['duplicate_bug_key_groups']:,}**, "
                  f"affected rows: **{m['duplicate_bug_key_rows_affected']:,}**. "
                  f"Missing Bug IDs: **{m['missing_bug_id_rows']:,}**; IDs failing positive-integer format: **{m['malformed_bug_id_rows']:,}**. "
                  f"Rows with repeated Bug ID alone: **{m['repeated_bug_id_rows_affected']:,}**. IDs may belong to different project namespaces. "
                  "Candidate-key duplicates exclude incomplete keys; no uniqueness guarantee is assumed.", ""]
    lines += ["## Text lengths and input budgets", "", markdown_table(t["text_length_summary"]), "",
              "Lengths exclude blank text. Character counts include whitespace; words are whitespace-separated units. "
              "The boxplot hides outlier marks for readability; histograms and quantiles retain the full range. "
              "Word thresholds below are screening proxies, not WordPiece token counts or measured truncation. "
              "Measure the actual DistilBERT tokenizer, including special tokens, on training data before choosing max_length.", "",
              markdown_table(t["word_length_thresholds"].loc[t["word_length_thresholds"]["severity"].eq("ALL_NONBLANK")]), ""]
    for role in ("project", "resolution", "split"):
        if f"{role}_severity_counts" in t:
            lines += [f"## Severity by {role}", "", markdown_table(t[f"{role}_severity_counts"].head(m["top_n"])), "",
                      "Tables include invalid/missing labels so totals reconcile. Plots show row percentages, with group sizes labeled; "
                      "all groups are retained in exported tables. Group differences describe associations and may reflect sampling or label practices.", ""]
    if "split_text_overlap" in t:
        lines += ["### Existing split text overlap", "", markdown_table(t["split_text_overlap"]), "",
                  "Shared normalized nonblank texts are a leakage risk requiring review. Text overlap does not establish identical underlying bug reports.", ""]
    else:
        lines += ["## Partition limitations", "", "No source split column is available, so cross-partition overlap cannot be measured yet. "
                  "Group duplicate texts and repeated bug identities before constructing partitions; verify class coverage after splitting.", ""]
    lines += ["## Vocabulary and severity language", "", "### Frequent terms", "", markdown_table(t["top_unigrams"]), "",
              "### Frequent adjacent word pairs", "", markdown_table(t["top_bigrams"]), "",
              "### Terms enriched within a class", "", markdown_table(t["distinctive_terms"].groupby("severity", sort=False).head(5)), "",
              "Counts are document frequencies (each term counts at most once per report), not raw token occurrences. "
              f"Terms require at least {m['min_document_frequency']} documents globally or within the relevant class. "
              "Lexical tokens match `[a-z][a-z0-9_]+` after casefolding and use the small stop list in the script; "
              "non-Latin language coverage is limited. Bigrams retain original adjacency and require both words to survive the stop list. "
              "Enrichment compares a class with all other valid classes using log2 of add-one-smoothed document prevalence: "
              "((class_df+1)/(class_docs+2)) / ((rest_df+1)/(rest_docs+2)). "
              "These exploratory associations are neither significance tests nor SHAP attributions. "
              "Fit any learned vocabulary and feature selection on training data only.", "",
              "## Matplotlib figures", ""]
    for figure in figures:
        lines += [f"![{figure.removesuffix('.png').replace('_', ' ')}](figures/{figure})", ""]
    lines += ["## Next preprocessing and evaluation checks", "",
              "1. Resolve missing/unknown labels and blank texts using documented inclusion rules; retain an audit of exclusions.",
              "2. Review conflicting labels and candidate-key collisions; group repeated identities and normalized texts before splitting.",
              "3. Preserve six-class coverage in partitions where feasible. Compare project-specific and pooled performance to expose domain differences.",
              "4. Measure training-tokenizer lengths, choose an input budget and record actual truncation rates by class.",
              "5. Use training-only weights or resampling, then evaluate macro/weighted F1 and per-class precision/recall on held-out data.",
              "6. Confirm feature availability at triage time. Resolution status may be assigned after triage and should not become a predictor without a timing audit.", "",
              "## Scope and reproducibility", "",
              "The source has no configured event timestamp; no time trends, freshness guarantees, or temporal generalization claims are made. "
              "Whole-corpus EDA can influence modeling decisions; reserve an untouched final test set for confirmatory evaluation. "
              "Duplicate checks do not detect semantic near-duplicates. No rows are cleaned, dropped, relabeled, split, or sampled by this command.", "",
              "See `metadata.json` for the input SHA-256, execution timestamp, column mapping, package versions, configuration and artifact inventory. "
              "All calculations are implemented in `src/explore_data.py`; full tables and row-level audit features accompany this report.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(input_path: Path, output: Path, *, text_column: str | None = None, label_column: str | None = None,
        top_n: int = 20, min_df: int = 5) -> Analysis:
    input_path, output = input_path.resolve(), output.resolve()
    # Read bytes once: metadata hashes exactly the input used for analysis.
    import io
    data = input_path.read_bytes()
    df = pd.read_csv(io.BytesIO(data), dtype="string", keep_default_na=False)
    analysis = analyze(df, text_column=text_column, label_column=label_column, top_n=top_n, min_df=min_df)
    # Refuse accidental overwrite of the input, including a previous output CSV.
    targets = [output / "tables" / f"{name}.csv" for name in analysis.tables]
    targets += [output / "row_features.csv", output / "report.md", output / "metadata.json"]
    if input_path in targets:
        raise ValueError("Output would overwrite the source input. Choose a different --output-dir.")
    (output / "tables").mkdir(parents=True, exist_ok=True)
    for name, table in analysis.tables.items():
        table.to_csv(output / "tables" / f"{name}.csv", index=False)
    analysis.rows.to_csv(output / "row_features.csv", index=False)
    figures = save_plots(analysis, output / "figures", top_n)
    write_report(analysis, output, figures)
    metadata = {"input_path": str(input_path), "input_sha256": hashlib.sha256(data).hexdigest(),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version.split()[0], "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                "metrics": analysis.metrics,
                "artifacts": ["report.md", "row_features.csv", "metadata.json"] +
                             [f"tables/{name}.csv" for name in analysis.tables] + [f"figures/{name}" for name in figures]}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "eclipse_bug_reports.csv", help="Local source CSV (default: data/eclipse_bug_reports.csv)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "eda", help="Output directory (default: reports/eda)")
    parser.add_argument("--text-column", help="Override automatic text-column detection")
    parser.add_argument("--label-column", help="Override automatic severity-column detection")
    parser.add_argument("--top-n", type=int, default=20, help="Maximum terms and groups shown (default: 20)")
    parser.add_argument("--min-df", type=int, default=5, help="Minimum lexical document frequency (default: 5)")
    args = parser.parse_args()
    try:
        result = run(args.input, args.output_dir, text_column=args.text_column, label_column=args.label_column,
                     top_n=args.top_n, min_df=args.min_df)
    except (OSError, ValueError, pd.errors.ParserError) as err:
        parser.exit(1, f"EDA failed: {err}\n")
    print(f"Analyzed {result.metrics['records']:,} records. Report: {(args.output_dir / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
