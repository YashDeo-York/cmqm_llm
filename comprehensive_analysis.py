"""
Comprehensive Analysis of LLM Translation Quality Judges
=========================================================
Covers:
  Part 1 – Cross-judge × cross-language analysis
  Part 2 – Human vs. LLM-judge comparison (Chinese & Portuguese)

Outputs figures to llm_judge_results/charts/ and a report to
llm_judge_results/comprehensive_report.txt
"""

import json, os, glob, warnings, itertools
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    cohen_kappa_score, confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score
)

warnings.filterwarnings("ignore")
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sns.set_theme(style="whitegrid", font_scale=1.1)

BASE = Path("d:/Projects/UFONIA/Translation_u")
RESULTS = BASE / "llm_judge_results"
CHARTS  = RESULTS / "charts"
CHARTS.mkdir(exist_ok=True)

MODELS_SHORT = {
    "CohereLabs__aya-expanse-32b":             "Aya-32B",
    "Qwen__Qwen3-30B-A3B":                    "Qwen3-30B",
    "Qwen__Qwen3-8B":                         "Qwen3-8B",
    "deepseek-ai__DeepSeek-V3-0324":           "DeepSeek-V3",
    "google__gemma-4-31B-it":                  "Gemma4-31B",
    "meta-llama__Llama-3_1-8B-Instruct":       "Llama3.1-8B",
    "meta-llama__Llama-3_3-70B-Instruct":      "Llama3.3-70B",
    "utter-project__EuroLLM-22B-Instruct-2512":"EuroLLM-22B",
}

LANG_SHORT = {
    "Arabic": "AR", "Bengali": "BN", "Chinese Mandarin": "ZH",
    "French": "FR", "German": "DE", "Polish": "PL",
    "Portuguese (Brazilian)": "PT-BR", "Spanish": "ES",
    "Turkish": "TR", "Urdu": "UR",
}

CMQM_CATS = [
    "clinical_accuracy", "ungrounded_content", "negation_polarity",
    "linguistic_quality", "patient_communication",
]

# ── 1. Load all judge data ──────────────────────────────────────────────
print("Loading judge data …")
rows = []
for fp in sorted(RESULTS.glob("*.jsonl")):
    if ".bak" in fp.name:
        continue
    model_key = fp.stem
    with open(fp, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            r["model_key"] = model_key
            r["model_short"] = MODELS_SHORT.get(model_key, model_key)
            r["lang_short"] = LANG_SHORT.get(r["language"], r["language"])
            # Normalise edit_required to binary int
            r["edit_binary"] = 1 if r["edit_required"] == "yes" else 0
            # Normalise harm to ordinal
            harm_map = {"none": 0, "low": 1, "moderate": 2, "high": 3}
            r["harm_ord"] = harm_map.get(r.get("harm_potential", "none"), 0)
            # CMQM binary flags
            cats = r.get("cmqm_categories") or []
            for c in CMQM_CATS:
                r[f"cmqm_{c}"] = 1 if c in cats else 0
            rows.append(r)

df = pd.DataFrame(rows)
print(f"  Total judge records: {len(df):,}")

# ── 2. Load human annotations ──────────────────────────────────────────
print("Loading human annotations …")
emily = pd.read_csv(BASE / "emily.csv")
emily["edit_binary_human"] = emily["Edit Required"].apply(
    lambda x: 1 if str(x).strip().lower() == "yes" else 0
)
emily["language"] = "Chinese Mandarin"
emily["lang_short"] = "ZH"
emily.rename(columns={"Identifier": "identifier", "Topic Key": "topic_key"}, inplace=True)

port = pd.read_excel(BASE / "Portuguese__Brazilian__annotations.xlsx")
port["edit_binary_human"] = port["Edit Required"].apply(
    lambda x: 1 if str(x).strip().lower() == "yes" else 0
)
port["language"] = "Portuguese (Brazilian)"
port["lang_short"] = "PT-BR"
port.rename(columns={"Identifier": "identifier", "Topic Key": "topic_key"}, inplace=True)

# Harmonise harm potential from human
harm_map_human = {"none": 0, "minor": 1, "low": 1, "moderate": 2, "major": 3, "high": 3}
emily["harm_ord_human"] = emily["Clinical Harm Potential"].map(
    lambda x: harm_map_human.get(str(x).strip().lower(), np.nan)
)
port["harm_ord_human"] = port["Clinical Harm Potential"].map(
    lambda x: harm_map_human.get(str(x).strip().lower(), np.nan)
)

# CMQM from humans
for human_df in [emily, port]:
    cats_col = human_df.get("CMQM Categories")
    for c in CMQM_CATS:
        human_df[f"cmqm_{c}_human"] = 0
    if cats_col is not None:
        for idx, val in cats_col.items():
            if pd.notna(val):
                for c in CMQM_CATS:
                    if c in str(val).lower():
                        human_df.loc[idx, f"cmqm_{c}_human"] = 1

print(f"  Emily (ZH): {len(emily)} rows, {emily['edit_binary_human'].sum()} edits")
print(f"  Portuguese:  {len(port)} rows, {port['edit_binary_human'].sum()} edits")

# ─────────────────────────────────────────────────────────────────────────
# PART 1: CROSS-JUDGE × CROSS-LANGUAGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────
report_lines = []
def rprint(s=""):
    report_lines.append(s)
    print(s)

rprint("=" * 80)
rprint("PART 1: CROSS-JUDGE × CROSS-LANGUAGE ANALYSIS")
rprint("=" * 80)

# 1a. Edit-required rates ─────────────────────────────────────────────────
edit_rate = df.groupby(["model_short", "lang_short"])["edit_binary"].mean().unstack()
edit_rate = edit_rate[sorted(edit_rate.columns)]

rprint("\n── 1a. Edit-Required Rate (%) by Judge × Language ──")
rprint((edit_rate * 100).round(1).to_string())

# Overall per model
model_edit = df.groupby("model_short")["edit_binary"].agg(["mean", "sum", "count"])
model_edit["mean"] = (model_edit["mean"] * 100).round(1)
rprint("\n── Overall edit rate per judge (%) ──")
rprint(model_edit.to_string())

# Overall per language
lang_edit = df.groupby("lang_short")["edit_binary"].agg(["mean", "sum", "count"])
lang_edit["mean"] = (lang_edit["mean"] * 100).round(1)
rprint("\n── Overall edit rate per language (%) ──")
rprint(lang_edit.to_string())

# Heatmap
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(edit_rate * 100, annot=True, fmt=".1f", cmap="YlOrRd",
            ax=ax, cbar_kws={"label": "Edit Required %"})
ax.set_title("Edit-Required Rate (%) by LLM Judge × Language", fontsize=14)
ax.set_ylabel("LLM Judge")
ax.set_xlabel("Target Language")
plt.tight_layout()
fig.savefig(CHARTS / "heatmap_edit_rate.png", dpi=200)
plt.close()

# 1b. Harm potential distribution ─────────────────────────────────────────
harm_dist = df.groupby(["model_short", "lang_short"])["harm_ord"].apply(
    lambda x: (x > 0).mean()
).unstack()
harm_dist = harm_dist[sorted(harm_dist.columns)]

rprint("\n── 1b. Non-zero Harm Rate (%) by Judge × Language ──")
rprint((harm_dist * 100).round(1).to_string())

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(harm_dist * 100, annot=True, fmt=".1f", cmap="YlOrRd",
            ax=ax, cbar_kws={"label": "Harm Flagged %"})
ax.set_title("Harm Flagged Rate (%) by LLM Judge × Language", fontsize=14)
ax.set_ylabel("LLM Judge")
ax.set_xlabel("Target Language")
plt.tight_layout()
fig.savefig(CHARTS / "heatmap_harm_rate.png", dpi=200)
plt.close()

# 1c. CMQM category breakdown ─────────────────────────────────────────────
rprint("\n── 1c. CMQM Category Distribution (among flagged items) ──")
flagged = df[df["edit_binary"] == 1]
if len(flagged) > 0:
    cmqm_by_model = flagged.groupby("model_short")[[f"cmqm_{c}" for c in CMQM_CATS]].mean()
    cmqm_by_model.columns = CMQM_CATS
    rprint((cmqm_by_model * 100).round(1).to_string())

    cmqm_by_lang = flagged.groupby("lang_short")[[f"cmqm_{c}" for c in CMQM_CATS]].mean()
    cmqm_by_lang.columns = CMQM_CATS
    rprint("\n  By language:")
    rprint((cmqm_by_lang * 100).round(1).to_string())

    # Stacked bar chart – CMQM by model
    fig, ax = plt.subplots(figsize=(12, 6))
    cmqm_by_model_plot = cmqm_by_model * 100
    cmqm_by_model_plot.plot(kind="bar", stacked=False, ax=ax, width=0.75)
    ax.set_title("CMQM Category Prevalence Among Flagged Items by Judge", fontsize=13)
    ax.set_ylabel("% of flagged items")
    ax.set_xlabel("")
    ax.legend(title="CMQM Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(CHARTS / "cmqm_by_judge.png", dpi=200)
    plt.close()

    # CMQM by language
    fig, ax = plt.subplots(figsize=(12, 6))
    cmqm_by_lang_plot = cmqm_by_lang * 100
    cmqm_by_lang_plot.plot(kind="bar", stacked=False, ax=ax, width=0.75)
    ax.set_title("CMQM Category Prevalence Among Flagged Items by Language", fontsize=13)
    ax.set_ylabel("% of flagged items")
    ax.set_xlabel("")
    ax.legend(title="CMQM Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(CHARTS / "cmqm_by_language.png", dpi=200)
    plt.close()

# 1d. Inter-judge agreement (pairwise Cohen's kappa on edit_required) ─────
rprint("\n── 1d. Pairwise Inter-Judge Agreement (Cohen's κ on edit_required) ──")
models = sorted(df["model_short"].unique())
# Build pivot: each row = (language, identifier, topic_key), cols = model edit_binary
pivot = df.pivot_table(
    index=["lang_short", "identifier", "topic_key"],
    columns="model_short",
    values="edit_binary",
    aggfunc="first",
)
pivot = pivot.dropna()

kappa_matrix = pd.DataFrame(np.nan, index=models, columns=models)
for m1, m2 in itertools.combinations(models, 2):
    k = cohen_kappa_score(pivot[m1], pivot[m2])
    kappa_matrix.loc[m1, m2] = k
    kappa_matrix.loc[m2, m1] = k
for m in models:
    kappa_matrix.loc[m, m] = 1.0

rprint(kappa_matrix.round(3).to_string())

# Mean kappa per judge
mean_kappa = kappa_matrix.apply(lambda col: col[col.index != col.name].mean())
rprint("\n  Mean pairwise κ per judge:")
rprint(mean_kappa.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(kappa_matrix, dtype=bool), k=1)
sns.heatmap(kappa_matrix.astype(float), annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0, vmax=1, mask=mask, ax=ax, square=True)
ax.set_title("Pairwise Cohen's κ (edit_required) Between LLM Judges", fontsize=13)
plt.tight_layout()
fig.savefig(CHARTS / "kappa_interjudge.png", dpi=200)
plt.close()

# 1e. Per-language inter-judge agreement ──────────────────────────────────
rprint("\n── 1e. Mean Pairwise κ by Language ──")
lang_kappas = {}
for lang in sorted(df["lang_short"].unique()):
    sub = pivot.loc[lang]
    ks = []
    for m1, m2 in itertools.combinations(models, 2):
        ks.append(cohen_kappa_score(sub[m1], sub[m2]))
    lang_kappas[lang] = {"mean_kappa": np.mean(ks), "std_kappa": np.std(ks),
                         "min_kappa": np.min(ks), "max_kappa": np.max(ks)}

lang_kappa_df = pd.DataFrame(lang_kappas).T.round(3)
rprint(lang_kappa_df.to_string())

# 1f. Fleiss-like multi-rater agreement per language ──────────────────────
rprint("\n── 1f. Proportion of Unanimous Agreement by Language ──")
for lang in sorted(df["lang_short"].unique()):
    sub = pivot.loc[lang]
    n = len(sub)
    unanimous = ((sub.sum(axis=1) == 0) | (sub.sum(axis=1) == len(models))).sum()
    rprint(f"  {lang}: {unanimous}/{n} = {unanimous/n*100:.1f}% unanimous")

# 1g. Row type analysis (question vs answer) ─────────────────────────────
rprint("\n── 1g. Edit Rate by Row Type ──")
rowtype_edit = df.groupby(["model_short", "row_type"])["edit_binary"].mean().unstack()
rprint((rowtype_edit * 100).round(1).to_string())

# 1h. Topic key analysis ──────────────────────────────────────────────────
rprint("\n── 1h. Edit Rate by Topic (aggregated across models and languages) ──")
topic_edit = df.groupby("topic_key")["edit_binary"].agg(["mean", "count"])
topic_edit["mean"] = (topic_edit["mean"] * 100).round(1)
topic_edit = topic_edit.sort_values("mean", ascending=False)
rprint(topic_edit.to_string())

# 1i. Parse error rates ───────────────────────────────────────────────────
rprint("\n── 1i. Parse Error/Repair Rates by Judge ──")
parse_stats = df.groupby("model_short").agg(
    parse_error_rate=("_parse_error", "mean"),
    parse_repaired_rate=("_parse_repaired", "mean"),
)
rprint((parse_stats * 100).round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────
# PART 2: HUMAN VS. LLM JUDGE COMPARISON
# ─────────────────────────────────────────────────────────────────────────
rprint("\n" + "=" * 80)
rprint("PART 2: HUMAN vs. LLM-JUDGE COMPARISON")
rprint("=" * 80)

def compare_human_judge(human_df, judge_df, lang_name, lang_filter):
    """Compare human annotations to each LLM judge for a specific language."""
    results = []

    judge_lang = judge_df[judge_df["language"] == lang_filter].copy()

    for model in sorted(judge_lang["model_short"].unique()):
        jm = judge_lang[judge_lang["model_short"] == model]

        merged = human_df.merge(
            jm[["identifier", "topic_key", "edit_binary", "harm_ord"] +
               [f"cmqm_{c}" for c in CMQM_CATS]],
            on=["identifier", "topic_key"],
            how="inner",
            suffixes=("_human", "_judge"),
        )

        if len(merged) == 0:
            continue

        y_human = merged["edit_binary_human"]
        y_judge = merged["edit_binary"]

        kappa = cohen_kappa_score(y_human, y_judge)
        acc = accuracy_score(y_human, y_judge)

        # Human = gold standard
        if y_human.sum() > 0 and y_judge.sum() > 0:
            prec = precision_score(y_human, y_judge, zero_division=0)
            rec  = recall_score(y_human, y_judge, zero_division=0)
            f1   = f1_score(y_human, y_judge, zero_division=0)
        else:
            prec = rec = f1 = 0.0

        cm = confusion_matrix(y_human, y_judge, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Specificity
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        results.append({
            "model": model,
            "n": len(merged),
            "human_edit_rate": y_human.mean(),
            "judge_edit_rate": y_judge.mean(),
            "kappa": kappa,
            "accuracy": acc,
            "precision": prec,
            "recall_sensitivity": rec,
            "specificity": spec,
            "f1": f1,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })

    return pd.DataFrame(results)


# ── 2a. Chinese (Emily) comparison ──────────────────────────────────────
rprint("\n── 2a. Chinese Mandarin: Emily vs. LLM Judges (edit_required) ──")
zh_comp = compare_human_judge(emily, df, "Chinese Mandarin", "Chinese Mandarin")
rprint(zh_comp.round(3).to_string(index=False))

# ── 2b. Portuguese comparison ────────────────────────────────────────────
rprint("\n── 2b. Portuguese (Brazilian): Human vs. LLM Judges (edit_required) ──")
pt_comp = compare_human_judge(port, df, "Portuguese (Brazilian)", "Portuguese (Brazilian)")
rprint(pt_comp.round(3).to_string(index=False))

# ── 2c. Combined comparison figure ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for ax, comp_df, title in [
    (axes[0], zh_comp, "Chinese Mandarin (Emily)"),
    (axes[1], pt_comp, "Portuguese-BR (Human)"),
]:
    metrics = ["kappa", "accuracy", "precision", "recall_sensitivity", "specificity", "f1"]
    x = np.arange(len(comp_df))
    w = 0.12
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, comp_df[m], w, label=m.replace("_", " ").title())
    ax.set_xticks(x + w * 2.5)
    ax.set_xticklabels(comp_df["model"], rotation=35, ha="right", fontsize=9)
    ax.set_title(f"Human vs. Judge — {title}", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylabel("Score")

plt.tight_layout()
fig.savefig(CHARTS / "human_vs_judge_metrics.png", dpi=200)
plt.close()

# ── 2d. Confusion matrices for best/worst kappa judges ──────────────────
for comp_df, lang_label, human_df_ref, lang_filter in [
    (zh_comp, "Chinese", emily, "Chinese Mandarin"),
    (pt_comp, "Portuguese", port, "Portuguese (Brazilian)"),
]:
    best_model = comp_df.loc[comp_df["kappa"].idxmax(), "model"]
    worst_model = comp_df.loc[comp_df["kappa"].idxmin(), "model"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model_name, label in [
        (axes[0], best_model, f"Best κ: {best_model}"),
        (axes[1], worst_model, f"Worst κ: {worst_model}"),
    ]:
        jm = df[(df["language"] == lang_filter) & (df["model_short"] == model_name)]
        merged = human_df_ref.merge(
            jm[["identifier", "topic_key", "edit_binary"]],
            on=["identifier", "topic_key"], how="inner",
        )
        cm = confusion_matrix(merged["edit_binary_human"], merged["edit_binary"], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Edit", "Edit"], yticklabels=["No Edit", "Edit"])
        ax.set_xlabel("Judge")
        ax.set_ylabel("Human")
        kval = comp_df.loc[comp_df["model"] == model_name, "kappa"].values[0]
        ax.set_title(f"{label} (κ={kval:.3f})")

    fig.suptitle(f"Confusion Matrices — {lang_label}", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(CHARTS / f"confusion_{lang_label.lower()}.png", dpi=200, bbox_inches="tight")
    plt.close()

# ── 2e. CMQM category agreement (where both flagged edit) ───────────────
rprint("\n── 2e. CMQM Category Agreement (among items both human & judge flagged) ──")

for human_df_ref, lang_filter, lang_label in [
    (emily, "Chinese Mandarin", "Chinese"),
    (port, "Portuguese (Brazilian)", "Portuguese"),
]:
    rprint(f"\n  {lang_label}:")
    judge_lang = df[df["language"] == lang_filter]

    for model in sorted(judge_lang["model_short"].unique()):
        jm = judge_lang[judge_lang["model_short"] == model]
        merged = human_df_ref.merge(
            jm[["identifier", "topic_key", "edit_binary"] + [f"cmqm_{c}" for c in CMQM_CATS]],
            on=["identifier", "topic_key"], how="inner",
        )
        both_flagged = merged[(merged["edit_binary_human"] == 1) & (merged["edit_binary"] == 1)]
        if len(both_flagged) < 3:
            continue

        cat_agree = {}
        for c in CMQM_CATS:
            hcol = f"cmqm_{c}_human"
            jcol = f"cmqm_{c}"
            if hcol in both_flagged.columns and both_flagged[hcol].sum() + both_flagged[jcol].sum() > 0:
                agree = (both_flagged[hcol] == both_flagged[jcol]).mean()
                cat_agree[c] = agree
        if cat_agree:
            rprint(f"    {model} (n={len(both_flagged)}): " +
                   ", ".join(f"{k}={v:.2f}" for k, v in cat_agree.items()))

# ── 2f. Harm potential comparison ────────────────────────────────────────
rprint("\n── 2f. Harm Potential Agreement ──")
for human_df_ref, lang_filter, lang_label in [
    (emily, "Chinese Mandarin", "Chinese"),
    (port, "Portuguese (Brazilian)", "Portuguese"),
]:
    rprint(f"\n  {lang_label}:")
    judge_lang = df[df["language"] == lang_filter]

    for model in sorted(judge_lang["model_short"].unique()):
        jm = judge_lang[judge_lang["model_short"] == model]
        merged = human_df_ref.merge(
            jm[["identifier", "topic_key", "harm_ord"]],
            on=["identifier", "topic_key"], how="inner",
            suffixes=("_human", "_judge"),
        )
        valid = merged.dropna(subset=["harm_ord_human"])
        if len(valid) < 5:
            continue

        # Binary: any harm vs no harm
        h_bin = (valid["harm_ord_human"] > 0).astype(int)
        j_bin = (valid["harm_ord"] > 0).astype(int)
        kappa = cohen_kappa_score(h_bin, j_bin) if h_bin.nunique() > 1 and j_bin.nunique() > 1 else float("nan")
        agree = (h_bin == j_bin).mean()
        rprint(f"    {model} (n={len(valid)}): agreement={agree:.3f}, κ={kappa:.3f}")

# ── 2g. Judge edit rate vs human edit rate scatter ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, comp_df, title in [
    (axes[0], zh_comp, "Chinese Mandarin"),
    (axes[1], pt_comp, "Portuguese-BR"),
]:
    ax.scatter(comp_df["human_edit_rate"] * 100, comp_df["judge_edit_rate"] * 100, s=80, zorder=5)
    for _, row in comp_df.iterrows():
        ax.annotate(row["model"], (row["human_edit_rate"] * 100, row["judge_edit_rate"] * 100),
                    fontsize=8, ha="left", va="bottom")
    lims = [0, max(comp_df["judge_edit_rate"].max(), comp_df["human_edit_rate"].max()) * 110]
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Perfect agreement")
    ax.set_xlabel("Human Edit Rate (%)")
    ax.set_ylabel("Judge Edit Rate (%)")
    ax.set_title(title)
    ax.legend()
plt.tight_layout()
fig.savefig(CHARTS / "edit_rate_scatter.png", dpi=200)
plt.close()

# ── 2h. Statistical tests ───────────────────────────────────────────────
rprint("\n── 2h. Statistical Tests ──")

# McNemar's test for each judge vs human
rprint("\n  McNemar's test (edit_required): Judge vs Human")
for human_df_ref, lang_filter, lang_label in [
    (emily, "Chinese Mandarin", "Chinese"),
    (port, "Portuguese (Brazilian)", "Portuguese"),
]:
    rprint(f"\n  {lang_label}:")
    judge_lang = df[df["language"] == lang_filter]

    for model in sorted(judge_lang["model_short"].unique()):
        jm = judge_lang[judge_lang["model_short"] == model]
        merged = human_df_ref.merge(
            jm[["identifier", "topic_key", "edit_binary"]],
            on=["identifier", "topic_key"], how="inner",
        )
        cm = confusion_matrix(merged["edit_binary_human"], merged["edit_binary"], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        # McNemar's: compare discordant pairs
        if fp + fn > 0:
            chi2 = (abs(fp - fn) - 1) ** 2 / (fp + fn)
            p = 1 - stats.chi2.cdf(chi2, df=1)
        else:
            chi2, p = 0, 1.0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        rprint(f"    {model}: FP={fp}, FN={fn}, χ²={chi2:.2f}, p={p:.4f} {sig}")

# ── Summary statistics ───────────────────────────────────────────────────
rprint("\n" + "=" * 80)
rprint("SUMMARY")
rprint("=" * 80)
rprint(f"\nDataset: {len(df):,} judge evaluations across {df['model_short'].nunique()} LLM judges, "
       f"{df['lang_short'].nunique()} languages, {df['identifier'].nunique()} unique items per language")
rprint(f"Overall edit-required rate: {df['edit_binary'].mean()*100:.1f}%")
rprint(f"Overall mean pairwise inter-judge κ: {mean_kappa.mean():.3f}")
rprint(f"\nChinese (Emily) — best judge: {zh_comp.loc[zh_comp['kappa'].idxmax(), 'model']} "
       f"(κ={zh_comp['kappa'].max():.3f})")
rprint(f"Chinese (Emily) — worst judge: {zh_comp.loc[zh_comp['kappa'].idxmin(), 'model']} "
       f"(κ={zh_comp['kappa'].min():.3f})")
rprint(f"Portuguese — best judge: {pt_comp.loc[pt_comp['kappa'].idxmax(), 'model']} "
       f"(κ={pt_comp['kappa'].max():.3f})")
rprint(f"Portuguese — worst judge: {pt_comp.loc[pt_comp['kappa'].idxmin(), 'model']} "
       f"(κ={pt_comp['kappa'].min():.3f})")

# Write report
report_path = RESULTS / "comprehensive_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"\nReport saved to {report_path}")
print(f"Charts saved to {CHARTS}")
