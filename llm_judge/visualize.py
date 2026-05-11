"""
Comprehensive visualization and analysis for LLM judge results.

Generates publication-quality charts and a detailed report including
human-vs-LLM comparison using Emily's Chinese Mandarin annotations.

Usage:
    python -m llm_judge.visualize
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from .config import (
    RESULTS_DIR, ALL_CMQM_IDS, CMQM_ID_TO_NAME, HARM_LEVELS, LANGUAGES,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.figsize": (14, 7),
    "figure.facecolor": "white",
})

CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Short display names for models
SHORT_NAMES = {
    "Qwen/Qwen3-30B-A3B": "Qwen3-30B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama3.3-70B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama3.1-8B",
    "google/gemma-4-31B-it": "Gemma4-31B",
    "deepseek-ai/DeepSeek-V3-0324": "DeepSeek-V3",
    "CohereLabs/aya-expanse-32b": "AyaExpanse-32B",
    "utter-project/EuroLLM-22B-Instruct-2512": "EuroLLM-22B",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B",
}

def short(model_id: str) -> str:
    return SHORT_NAMES.get(model_id, model_id.split("/")[-1][:20])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results() -> pd.DataFrame:
    frames = []
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(RESULTS_DIR, fname)
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
        raise FileNotFoundError(f"No .jsonl files in {RESULTS_DIR}")
    df = pd.concat(frames, ignore_index=True)
    # Only keep models with full runs (>= 20000 items)
    model_counts = df.groupby("model").size()
    full_models = model_counts[model_counts >= 20000].index
    df = df[df["model"].isin(full_models)].copy()
    df["short_model"] = df["model"].map(short)
    return df


def load_emily(csv_path: str) -> pd.DataFrame:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ident = row.get("\ufeffIdentifier") or row.get("Identifier") or ""
            edit_req = (row.get("Edit Required") or "").strip().lower()
            edit_req = "yes" if edit_req == "yes" else "no"
            cats_raw = (row.get("CMQM Categories") or "").strip()
            cats = [c.strip() for c in cats_raw.split(",") if c.strip()] if cats_raw else []
            harm = (row.get("Clinical Harm Potential") or "").strip().lower()
            rows.append({
                "identifier": ident.strip(),
                "edit_required_human": edit_req,
                "cmqm_human": cats,
                "harm_human": harm if harm else "none",
                "post_edit_human": (row.get("Post-Edited Translation") or "").strip(),
                "notes_human": (row.get("Notes") or "").strip(),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Per-model summary
# ---------------------------------------------------------------------------

def fig_summary_stats(df: pd.DataFrame):
    models = sorted(df["model"].unique(), key=short)
    stats = []
    for m in models:
        mdf = df[df["model"] == m]
        total = len(mdf)
        valid = mdf[mdf["edit_required"].isin(["yes", "no"])]
        edit_yes = (valid["edit_required"] == "yes").sum()
        edit_rate = edit_yes / len(valid) if len(valid) > 0 else 0
        errs = (mdf["edit_required"].isin(["error", "unknown"])).sum()
        stats.append({
            "model": short(m), "total": total,
            "edit_yes": edit_yes, "edit_rate": edit_rate,
            "errors": errs,
        })
    sdf = pd.DataFrame(stats)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: edit rate
    colors = plt.cm.Set2(np.linspace(0, 1, len(sdf)))
    bars = axes[0].barh(sdf["model"], sdf["edit_rate"] * 100, color=colors)
    axes[0].set_xlabel("Edit Required Rate (%)")
    axes[0].set_title("Edit Rate by Model")
    for bar, rate in zip(bars, sdf["edit_rate"]):
        axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{rate*100:.1f}%", va="center", fontsize=9)

    # Bar chart: absolute edit counts
    bars2 = axes[1].barh(sdf["model"], sdf["edit_yes"], color=colors)
    axes[1].set_xlabel("Number of Edits Flagged")
    axes[1].set_title("Total Edits Flagged by Model")
    for bar, count in zip(bars2, sdf["edit_yes"]):
        axes[1].text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                     str(count), va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "01_summary_stats.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return sdf


# ---------------------------------------------------------------------------
# 2. Edit rate by language (heatmap)
# ---------------------------------------------------------------------------

def fig_edit_rate_heatmap(df: pd.DataFrame):
    models = sorted(df["model"].unique(), key=short)
    langs = sorted(df["language"].unique())

    pivot = pd.DataFrame(index=langs, columns=[short(m) for m in models], dtype=float)
    for m in models:
        for lang in langs:
            sub = df[(df["model"] == m) & (df["language"] == lang)]
            valid = sub[sub["edit_required"].isin(["yes", "no"])]
            if len(valid) > 0:
                pivot.loc[lang, short(m)] = (valid["edit_required"] == "yes").sum() / len(valid)

    fig, ax = plt.subplots(figsize=(14, 7))
    data = pivot.values.astype(float) * 100
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = data[i, j]
            color = "white" if val > 30 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_title("Edit Rate (%) by Language and Model")
    fig.colorbar(im, ax=ax, label="Edit Rate (%)", shrink=0.8)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "02_edit_rate_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return pivot


# ---------------------------------------------------------------------------
# 3. CMQM category distribution
# ---------------------------------------------------------------------------

def fig_cmqm_distribution(df: pd.DataFrame):
    models = sorted(df["model"].unique(), key=short)
    cat_names = [CMQM_ID_TO_NAME[c] for c in ALL_CMQM_IDS]

    counts = {}
    for m in models:
        mdf = df[(df["model"] == m) & (df["edit_required"] == "yes")]
        counter = Counter()
        for cats in mdf["cmqm_categories"]:
            if isinstance(cats, list):
                for c in cats:
                    counter[c] += 1
        counts[short(m)] = [counter.get(cid, 0) for cid in ALL_CMQM_IDS]

    x = np.arange(len(cat_names))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    for i, m in enumerate(models):
        ax.bar(x + i * width, counts[short(m)], width, label=short(m), color=colors[i])

    ax.set_xticks(x + width * len(models) / 2)
    ax.set_xticklabels(cat_names, rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("CMQM Error Category Distribution by Model")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "03_cmqm_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 4. Harm potential distribution
# ---------------------------------------------------------------------------

def fig_harm_distribution(df: pd.DataFrame):
    models = sorted(df["model"].unique(), key=short)
    harm_levels = ["none", "low", "moderate", "high"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    bottoms = np.zeros(len(models))
    colors_map = {"none": "#2ecc71", "low": "#f1c40f", "moderate": "#e67e22", "high": "#e74c3c"}

    for level in harm_levels:
        vals = []
        for m in models:
            mdf = df[df["model"] == m]
            valid = mdf[mdf["edit_required"].isin(["yes", "no"])]
            count = (valid["harm_potential"] == level).sum()
            vals.append(count / len(valid) * 100 if len(valid) > 0 else 0)
        ax.bar(x, vals, bottom=bottoms, label=level.capitalize(),
               color=colors_map[level])
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in models], rotation=45, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Clinical Harm Potential Distribution by Model")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "04_harm_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 5. Cross-model agreement heatmap
# ---------------------------------------------------------------------------

def fig_cross_model_agreement(df: pd.DataFrame):
    models = sorted(df["model"].unique(), key=short)
    # Build lookup
    lookup = defaultdict(dict)
    for _, row in df.iterrows():
        key = f"{row['language']}|{row['identifier']}"
        lookup[key][row["model"]] = row["edit_required"]

    n = len(models)
    matrix = np.zeros((n, n))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            agree = total = 0
            for key, verdicts in lookup.items():
                if m1 in verdicts and m2 in verdicts:
                    v1, v2 = verdicts[m1], verdicts[m2]
                    if v1 in ("yes", "no") and v2 in ("yes", "no"):
                        total += 1
                        if v1 == v2:
                            agree += 1
            matrix[i, j] = agree / max(total, 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix * 100, cmap="RdYlGn", vmin=50, vmax=100)
    labels = [short(m) for m in models]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i,j]*100:.1f}", ha="center", va="center",
                    fontsize=9, color="white" if matrix[i,j] < 0.7 else "black")
    ax.set_title("Cross-Model Agreement on Edit Required (%)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "05_cross_model_agreement.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return matrix, models


# ---------------------------------------------------------------------------
# 6. Scale comparison (large vs small within family)
# ---------------------------------------------------------------------------

def fig_scale_comparison(df: pd.DataFrame):
    pairs = [
        ("Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-8B", "Qwen3"),
        ("meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Llama"),
    ]
    langs = sorted(df["language"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (large, small, family) in zip(axes, pairs):
        rates_large = []
        rates_small = []
        for lang in langs:
            for m, rates in [(large, rates_large), (small, rates_small)]:
                sub = df[(df["model"] == m) & (df["language"] == lang)]
                valid = sub[sub["edit_required"].isin(["yes", "no"])]
                rate = (valid["edit_required"] == "yes").sum() / len(valid) * 100 if len(valid) > 0 else 0
                rates.append(rate)

        x = np.arange(len(langs))
        w = 0.35
        ax.bar(x - w/2, rates_large, w, label=short(large), color="#3498db")
        ax.bar(x + w/2, rates_small, w, label=short(small), color="#e74c3c")
        ax.set_xticks(x)
        ax.set_xticklabels(langs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Edit Rate (%)")
        ax.set_title(f"{family} Scale Comparison")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "06_scale_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 7. Emily (human) vs LLM comparison — Chinese Mandarin only
# ---------------------------------------------------------------------------

def fig_emily_vs_models(df: pd.DataFrame, emily_df: pd.DataFrame):
    """Compare all models against Emily's human labels on Chinese Mandarin."""
    chinese = df[df["language"] == "Chinese Mandarin"].copy()
    models = sorted(chinese["model"].unique(), key=short)

    # Build Emily lookup
    emily_lookup = {}
    for _, row in emily_df.iterrows():
        emily_lookup[row["identifier"]] = row

    results = []
    for m in models:
        mdf = chinese[chinese["model"] == m]
        tp = fp = fn = tn = 0
        cmqm_jaccard_sum = 0
        matched = 0

        for _, row in mdf.iterrows():
            ident = row["identifier"]
            if ident not in emily_lookup:
                continue
            human = emily_lookup[ident]
            matched += 1

            llm_edit = row["edit_required"]
            human_edit = human["edit_required_human"]

            if human_edit == "yes" and llm_edit == "yes":
                tp += 1
            elif human_edit == "no" and llm_edit == "yes":
                fp += 1
            elif human_edit == "yes" and llm_edit == "no":
                fn += 1
            elif human_edit == "no" and llm_edit == "no":
                tn += 1

            # CMQM Jaccard
            llm_cats = set(row.get("cmqm_categories", []) or [])
            human_cats = set(human.get("cmqm_human", []) or [])
            if llm_cats or human_cats:
                jaccard = len(llm_cats & human_cats) / len(llm_cats | human_cats)
            else:
                jaccard = 1.0
            cmqm_jaccard_sum += jaccard

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy = (tp + tn) / max(matched, 1)

        results.append({
            "model": short(m),
            "matched": matched,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cmqm_jaccard": cmqm_jaccard_sum / max(matched, 1),
        })

    rdf = pd.DataFrame(results)

    # --- Chart A: Accuracy, Precision, Recall, F1 ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(rdf))
    w = 0.2
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        axes[0].bar(x + i * w, rdf[metric] * 100, w, label=metric.capitalize(), color=color)

    axes[0].set_xticks(x + w * 1.5)
    axes[0].set_xticklabels(rdf["model"], rotation=45, ha="right")
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Human (Emily) vs LLM: Edit Detection Metrics\n(Chinese Mandarin, 774 items)")
    axes[0].legend()
    axes[0].set_ylim(0, 105)

    # --- Chart B: Confusion matrix style (TP, FP, FN, TN as stacked) ---
    categories = ["TP", "FP", "FN", "TN"]
    cat_colors = ["#2ecc71", "#e67e22", "#e74c3c", "#3498db"]
    bottoms = np.zeros(len(rdf))
    for cat, color in zip(categories, cat_colors):
        vals = rdf[cat.lower()].values
        axes[1].bar(x, vals, bottom=bottoms, label=cat, color=color)
        bottoms += vals

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(rdf["model"], rotation=45, ha="right")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confusion Matrix Components (vs Emily)")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "07_emily_vs_models.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # --- Chart C: CMQM category agreement ---
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    bars = ax2.barh(rdf["model"], rdf["cmqm_jaccard"] * 100,
                    color=plt.cm.Set2(np.linspace(0, 1, len(rdf))))
    ax2.set_xlabel("CMQM Category Jaccard Similarity (%)")
    ax2.set_title("CMQM Category Agreement: Emily (Human) vs LLM\n(Chinese Mandarin)")
    for bar, val in zip(bars, rdf["cmqm_jaccard"]):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val*100:.1f}%", va="center", fontsize=9)
    ax2.set_xlim(0, 105)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "08_emily_cmqm_agreement.png")
    fig2.savefig(path, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {path}")

    return rdf


# ---------------------------------------------------------------------------
# 8. Per-language CMQM category heatmap
# ---------------------------------------------------------------------------

def fig_cmqm_by_language(df: pd.DataFrame, model_id: str):
    """CMQM category distribution by language for a single model."""
    mdf = df[(df["model"] == model_id) & (df["edit_required"] == "yes")]
    langs = sorted(mdf["language"].unique())
    cat_names = [CMQM_ID_TO_NAME[c] for c in ALL_CMQM_IDS]

    data = np.zeros((len(langs), len(ALL_CMQM_IDS)))
    for i, lang in enumerate(langs):
        ldf = mdf[mdf["language"] == lang]
        counter = Counter()
        for cats in ldf["cmqm_categories"]:
            if isinstance(cats, list):
                for c in cats:
                    counter[c] += 1
        for j, cid in enumerate(ALL_CMQM_IDS):
            data[i, j] = counter.get(cid, 0)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(data, aspect="auto", cmap="YlOrBr")
    ax.set_xticks(range(len(cat_names)))
    ax.set_xticklabels(cat_names, rotation=30, ha="right")
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels(langs)
    for i in range(len(langs)):
        for j in range(len(cat_names)):
            ax.text(j, i, f"{int(data[i,j])}", ha="center", va="center",
                    fontsize=8, color="white" if data[i,j] > data.max()*0.6 else "black")
    ax.set_title(f"CMQM Error Categories by Language — {short(model_id)}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Count")
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f"09_cmqm_by_language_{short(model_id).replace(' ','_')}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# 9. Self-evaluation bias (Llama judging its own translations)
# ---------------------------------------------------------------------------

def fig_self_eval_bias(df: pd.DataFrame):
    """Compare Llama-3.3-70B (translator model) edit rate vs other models."""
    models = sorted(df["model"].unique(), key=short)
    langs = sorted(df["language"].unique())

    llama_id = "meta-llama/Llama-3.3-70B-Instruct"
    if llama_id not in df["model"].values:
        return

    rates = {}
    for m in models:
        r = []
        for lang in langs:
            sub = df[(df["model"] == m) & (df["language"] == lang)]
            valid = sub[sub["edit_required"].isin(["yes", "no"])]
            rate = (valid["edit_required"] == "yes").sum() / len(valid) * 100 if len(valid) > 0 else 0
            r.append(rate)
        rates[m] = np.mean(r)

    fig, ax = plt.subplots(figsize=(12, 5))
    models_sorted = sorted(models, key=lambda m: rates[m])
    vals = [rates[m] for m in models_sorted]
    colors = ["#e74c3c" if m == llama_id else "#3498db" for m in models_sorted]
    labels = [short(m) for m in models_sorted]

    bars = ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Average Edit Rate (%)")
    ax.set_title("Self-Evaluation Bias: Llama-3.3-70B (red) Judging Its Own Translations")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, "10_self_eval_bias.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading results...")
    df = load_all_results()
    print(f"  {len(df)} total judgments, {df['model'].nunique()} models, "
          f"{df['language'].nunique()} languages")

    # Load Emily's human annotations
    emily_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "emily.csv"
    )
    emily_df = None
    if os.path.exists(emily_path):
        emily_df = load_emily(emily_path)
        print(f"  Emily annotations: {len(emily_df)} items, "
              f"{(emily_df['edit_required_human'] == 'yes').sum()} edits")

    print("\nGenerating charts...")

    # 1. Summary stats
    sdf = fig_summary_stats(df)
    print(f"\n--- Per-Model Summary ---")
    print(sdf.to_string(index=False))

    # 2. Edit rate heatmap
    pivot = fig_edit_rate_heatmap(df)

    # 3. CMQM distribution
    fig_cmqm_distribution(df)

    # 4. Harm distribution
    fig_harm_distribution(df)

    # 5. Cross-model agreement
    matrix, models = fig_cross_model_agreement(df)

    # 6. Scale comparison
    fig_scale_comparison(df)

    # 7 & 8. Emily vs models (if available)
    if emily_df is not None:
        rdf = fig_emily_vs_models(df, emily_df)
        print(f"\n--- Emily vs LLM (Chinese Mandarin) ---")
        print(rdf[["model", "accuracy", "precision", "recall", "f1", "cmqm_jaccard"]].to_string(index=False))

    # 9. CMQM by language for best model
    best_model = df.groupby("model").apply(
        lambda g: (g["edit_required"] == "error").sum()
    ).idxmin()
    fig_cmqm_by_language(df, best_model)

    # 10. Self-eval bias
    fig_self_eval_bias(df)

    # Save text report
    report_path = os.path.join(RESULTS_DIR, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("LLM TRANSLATION QUALITY JUDGE - ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total judgments: {len(df)}\n")
        f.write(f"Models: {df['model'].nunique()}\n")
        f.write(f"Languages: {df['language'].nunique()}\n\n")

        f.write("PER-MODEL SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write(sdf.to_string(index=False) + "\n\n")

        f.write("EDIT RATE BY LANGUAGE\n")
        f.write("-" * 50 + "\n")
        f.write(pivot.to_string() + "\n\n")

        if emily_df is not None:
            f.write("EMILY (HUMAN) VS LLM — CHINESE MANDARIN\n")
            f.write("-" * 50 + "\n")
            f.write(rdf.to_string(index=False) + "\n\n")

    print(f"\n  Report saved to {report_path}")
    print(f"  Charts saved to {CHARTS_DIR}/")
    print(f"\nDone! {len(os.listdir(CHARTS_DIR))} charts generated.")


if __name__ == "__main__":
    main()
