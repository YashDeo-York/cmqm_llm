"""
Analysis and comparison of LLM judge results.

Produces:
- Per-model summary stats
- Cross-model agreement metrics
- Language-wise breakdown
- CMQM category distribution per model
- Clinical harm distribution per model
- Human vs LLM agreement (when human labels are available)
"""

import json
import os
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from .config import RESULTS_DIR, ALL_CMQM_IDS, CMQM_ID_TO_NAME, HARM_LEVELS, LANGUAGES


def load_all_results(results_dir: str | None = None) -> pd.DataFrame:
    """Load all JSONL result files into a single DataFrame."""
    rdir = results_dir or RESULTS_DIR
    frames = []
    for fname in os.listdir(rdir):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(rdir, fname)
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if records:
            frames.append(pd.DataFrame(records))

    if not frames:
        raise FileNotFoundError(f"No .jsonl result files found in {rdir}")

    df = pd.concat(frames, ignore_index=True)
    return df


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model summary: total judged, edit rate, error rate, parse errors."""
    rows = []
    for model, mdf in df.groupby("model"):
        total = len(mdf)
        edit_yes = (mdf["edit_required"] == "yes").sum()
        edit_no = (mdf["edit_required"] == "no").sum()
        errors = (mdf["edit_required"].isin(["error", "unknown"])).sum()
        parse_errs = mdf.get("_parse_error", pd.Series([False]*len(mdf))).sum()

        rows.append({
            "model": model,
            "total": total,
            "edit_yes": edit_yes,
            "edit_no": edit_no,
            "edit_rate": edit_yes / max(total - errors, 1),
            "api_errors": errors,
            "parse_errors": int(parse_errs),
        })

    return pd.DataFrame(rows).sort_values("model")


def edit_rate_by_language(df: pd.DataFrame) -> pd.DataFrame:
    """Edit rate per (model, language) — shows how models differ across languages."""
    rows = []
    for (model, lang), gdf in df.groupby(["model", "language"]):
        valid = gdf[gdf["edit_required"].isin(["yes", "no"])]
        total = len(valid)
        edits = (valid["edit_required"] == "yes").sum()
        rows.append({
            "model": model,
            "language": lang,
            "total": total,
            "edits": edits,
            "edit_rate": edits / max(total, 1),
        })

    result = pd.DataFrame(rows)
    return result.pivot_table(
        index="language", columns="model", values="edit_rate"
    ).round(3)


def cmqm_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count of each CMQM category per model.
    Since categories is a list, we explode it.
    """
    # Ensure cmqm_categories is a list
    df = df.copy()
    df["cmqm_categories"] = df["cmqm_categories"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    rows = []
    for model, mdf in df.groupby("model"):
        counter = Counter()
        for cats in mdf["cmqm_categories"]:
            for c in cats:
                counter[c] += 1
        total_items = len(mdf[mdf["edit_required"] == "yes"])
        for cat_id in ALL_CMQM_IDS:
            rows.append({
                "model": model,
                "category": CMQM_ID_TO_NAME.get(cat_id, cat_id),
                "count": counter.get(cat_id, 0),
                "rate": counter.get(cat_id, 0) / max(total_items, 1),
            })

    result = pd.DataFrame(rows)
    return result.pivot_table(
        index="category", columns="model", values="count", fill_value=0
    )


def cmqm_by_language(df: pd.DataFrame) -> pd.DataFrame:
    """CMQM category counts per (model, language)."""
    df = df.copy()
    df["cmqm_categories"] = df["cmqm_categories"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    rows = []
    for (model, lang), gdf in df.groupby(["model", "language"]):
        counter = Counter()
        for cats in gdf["cmqm_categories"]:
            for c in cats:
                counter[c] += 1
        for cat_id in ALL_CMQM_IDS:
            rows.append({
                "model": model,
                "language": lang,
                "category": CMQM_ID_TO_NAME.get(cat_id, cat_id),
                "count": counter.get(cat_id, 0),
            })

    return pd.DataFrame(rows)


def harm_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Clinical harm potential distribution per model."""
    rows = []
    for model, mdf in df.groupby("model"):
        total = len(mdf)
        for level in HARM_LEVELS + ["unknown", "error"]:
            count = (mdf["harm_potential"] == level).sum()
            rows.append({
                "model": model,
                "harm_level": level,
                "count": count,
                "rate": count / max(total, 1),
            })

    result = pd.DataFrame(rows)
    return result.pivot_table(
        index="harm_level", columns="model", values="count", fill_value=0
    )


def harm_by_language(df: pd.DataFrame) -> pd.DataFrame:
    """Harm distribution per (model, language)."""
    rows = []
    for (model, lang), gdf in df.groupby(["model", "language"]):
        total = len(gdf)
        for level in HARM_LEVELS:
            count = (gdf["harm_potential"] == level).sum()
            rows.append({
                "model": model,
                "language": lang,
                "harm_level": level,
                "count": count,
                "rate": count / max(total, 1),
            })
    return pd.DataFrame(rows)


def cross_model_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise agreement on edit_required between models.
    Returns a model x model matrix of agreement rates.
    """
    models = sorted(df["model"].unique())
    # Build a dict: (language, identifier) -> {model: edit_required}
    lookup: dict[str, dict[str, str]] = defaultdict(dict)
    for _, row in df.iterrows():
        key = f"{row['language']}|{row['identifier']}"
        lookup[key][row["model"]] = row["edit_required"]

    # Compute pairwise agreement
    matrix = {}
    for m1 in models:
        matrix[m1] = {}
        for m2 in models:
            agree = 0
            total = 0
            for key, verdicts in lookup.items():
                if m1 in verdicts and m2 in verdicts:
                    v1, v2 = verdicts[m1], verdicts[m2]
                    if v1 in ("yes", "no") and v2 in ("yes", "no"):
                        total += 1
                        if v1 == v2:
                            agree += 1
            matrix[m1][m2] = agree / max(total, 1)

    return pd.DataFrame(matrix).round(3)


def human_vs_llm_agreement(df: pd.DataFrame,
                           human_labels: dict[str, dict]) -> pd.DataFrame:
    """
    Compare LLM judgments against human annotations.

    human_labels: {(language, identifier): {"edit_required": "yes"/"no", ...}}

    Returns per-model agreement metrics.
    """
    rows = []
    for model, mdf in df.groupby("model"):
        agree_edit = 0
        agree_harm = 0
        agree_cmqm = 0
        total = 0

        for _, row in mdf.iterrows():
            key = (row["language"], row["identifier"])
            if key not in human_labels:
                continue
            human = human_labels[key]
            total += 1

            # Edit required agreement
            if row["edit_required"] == human.get("edit_required", "").lower():
                agree_edit += 1

            # Harm level agreement
            if row["harm_potential"] == human.get("harm_potential", "").lower():
                agree_harm += 1

            # CMQM category overlap (Jaccard)
            llm_cats = set(row.get("cmqm_categories", []) or [])
            human_cats = set(human.get("cmqm_categories", []) or [])
            if llm_cats or human_cats:
                jaccard = len(llm_cats & human_cats) / max(len(llm_cats | human_cats), 1)
                agree_cmqm += jaccard
            else:
                agree_cmqm += 1  # both empty = agree

        rows.append({
            "model": model,
            "items_with_human_labels": total,
            "edit_agreement": agree_edit / max(total, 1),
            "harm_agreement": agree_harm / max(total, 1),
            "cmqm_jaccard": agree_cmqm / max(total, 1),
        })

    return pd.DataFrame(rows)


def generate_report(results_dir: str | None = None,
                    output_path: str | None = None) -> str:
    """
    Generate a comprehensive analysis report.

    Returns the report as a string and optionally saves to file.
    """
    df = load_all_results(results_dir)
    lines = []
    lines.append("=" * 70)
    lines.append("LLM TRANSLATION QUALITY JUDGE - ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"\nTotal judgments: {len(df)}")
    lines.append(f"Models: {sorted(df['model'].unique())}")
    lines.append(f"Languages: {sorted(df['language'].unique())}")

    # 1. Summary stats
    lines.append("\n" + "-" * 50)
    lines.append("1. PER-MODEL SUMMARY")
    lines.append("-" * 50)
    stats = summary_stats(df)
    lines.append(stats.to_string(index=False))

    # 2. Edit rate by language
    lines.append("\n" + "-" * 50)
    lines.append("2. EDIT RATE BY LANGUAGE (rows=language, cols=model)")
    lines.append("-" * 50)
    er_lang = edit_rate_by_language(df)
    lines.append(er_lang.to_string())

    # 3. CMQM distribution
    lines.append("\n" + "-" * 50)
    lines.append("3. CMQM CATEGORY COUNTS (rows=category, cols=model)")
    lines.append("-" * 50)
    cmqm = cmqm_distribution(df)
    lines.append(cmqm.to_string())

    # 4. Harm distribution
    lines.append("\n" + "-" * 50)
    lines.append("4. CLINICAL HARM POTENTIAL (rows=level, cols=model)")
    lines.append("-" * 50)
    harm = harm_distribution(df)
    lines.append(harm.to_string())

    # 5. Cross-model agreement
    lines.append("\n" + "-" * 50)
    lines.append("5. CROSS-MODEL AGREEMENT ON edit_required")
    lines.append("-" * 50)
    agree = cross_model_agreement(df)
    lines.append(agree.to_string())

    report = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {output_path}")

    return report


def export_results_xlsx(results_dir: str | None = None,
                        output_path: str | None = None):
    """Export all results to an Excel workbook with analysis sheets."""
    df = load_all_results(results_dir)
    out = output_path or os.path.join(RESULTS_DIR, "analysis.xlsx")

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        # Raw results
        df.to_excel(writer, sheet_name="Raw Results", index=False)

        # Summary
        summary_stats(df).to_excel(writer, sheet_name="Summary", index=False)

        # Edit rate by language
        edit_rate_by_language(df).to_excel(writer, sheet_name="Edit Rate by Lang")

        # CMQM distribution
        cmqm_distribution(df).to_excel(writer, sheet_name="CMQM Distribution")

        # Harm distribution
        harm_distribution(df).to_excel(writer, sheet_name="Harm Distribution")

        # Cross-model agreement
        cross_model_agreement(df).to_excel(writer, sheet_name="Model Agreement")

        # CMQM by language (long form for charting)
        cmqm_by_language(df).to_excel(
            writer, sheet_name="CMQM by Language", index=False
        )

        # Harm by language
        harm_by_language(df).to_excel(
            writer, sheet_name="Harm by Language", index=False
        )

    print(f"Analysis workbook saved to {out}")
