"""
Comprehensive Human (Atlas) vs LLM Judge comparison analysis.

Compares professional human annotations against LLM judge outputs on:
  1. Edit-required agreement (binary classification metrics)
  2. CMQM category agreement (Jaccard, per-category F1)
  3. Clinical harm agreement
  4. Per-language breakdown
  5. Cross-model ranking vs human
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix, matthews_corrcoef,
)

RESULTS_DIR = Path("llm_judge_results")
HUMAN_DIR = Path("human_professional")
CHARTS_DIR = RESULTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

HUMAN_FILES = {
    "Bengali": "Bengali_atlas_clean.xlsx",
    "Chinese Mandarin": "Chinese_Mandarin_atlas_clean.xlsx",
    "French": "French_atlas_clean.xlsx",
    "Polish": "Polish_atlas_clean.xlsx",
    "Portuguese (Brazilian)": "Portuguese_Brazilian_atlas_clean.xlsx",
    "Spanish": "Spanish_atlas_clean.xlsx",
    "Turkish": "Turkish_atlas_clean.xlsx",
    "Urdu": "Urdu_atlas_clean.xlsx",
}

# Map from config model ID to short name
SHORT_NAMES = {
    "Qwen/Qwen3-30B-A3B": "Qwen3-30B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama3.3-70B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama3.1-8B",
    "google/gemma-4-31B-it": "Gemma4-31B",
    "deepseek-ai/DeepSeek-V3-0324": "DeepSeek-V3",
    "CohereLabs/aya-expanse-32b": "AyaExpanse-32B",
}

# Human harm labels mapped to LLM scale.
# LLM prompt only offers low|moderate|high (no "none") for edit=yes items.
# Human "none" on edit=yes means "no clinical harm" = LLM "low" (lowest severity).
# Human "minor" = LLM "low", human "major" = LLM "high".
HARM_MAP_EDIT_YES = {
    "none": "low", "": "low",
    "minor": "low", "low": "low",
    "moderate": "moderate",
    "major": "high", "high": "high",
}
# For edit=no items, harm is always none on both sides
HARM_MAP_EDIT_NO = {
    "none": "none", "": "none",
    "minor": "none", "low": "none",
    "moderate": "none",
    "major": "none", "high": "none",
}

CMQM_IDS = [
    "clinical_accuracy", "ungrounded_content", "negation_polarity",
    "linguistic_quality", "patient_communication",
]


def load_human_annotations():
    """Load all human atlas annotations, keyed by (language, identifier)."""
    human = {}
    for lang, fname in HUMAN_FILES.items():
        path = HUMAN_DIR / fname
        wb = openpyxl.load_workbook(path, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        headers = [str(h).replace("\ufeff", "").strip() if h else "" for h in rows[0]]

        col = {h: i for i, h in enumerate(headers)}

        for row in rows[1:]:
            ident = str(row[col["Identifier"]]).strip()
            edit = str(row[col["Edit Required"]] or "").strip().lower()
            edit = "yes" if edit == "yes" else "no"

            harm_raw = str(row[col["Clinical Harm Potential"]] or "").strip().lower()
            harm_map = HARM_MAP_EDIT_YES if edit == "yes" else HARM_MAP_EDIT_NO
            harm = harm_map.get(harm_raw, "low" if edit == "yes" else "none")

            cmqm_raw = str(row[col["CMQM Categories"]] or "").strip()
            cmqm = set()
            if cmqm_raw and cmqm_raw != "None":
                for c in cmqm_raw.replace(";", ",").split(","):
                    c = c.strip()
                    if c:
                        cmqm.add(c)

            mqm_raw = str(row[col["MQM Categories"]] or "").strip()
            mqm = set()
            if mqm_raw and mqm_raw != "None":
                for c in mqm_raw.replace(";", ",").split(","):
                    c = c.strip()
                    if c:
                        mqm.add(c)

            human[(lang, ident)] = {
                "edit_required": edit,
                "harm": harm,
                "cmqm": cmqm,
                "mqm": mqm,
            }
        wb.close()
    return human


def load_llm_results():
    """Load all LLM CMQM results, keyed by (model_id, language, identifier)."""
    llm = {}
    for f in sorted(RESULTS_DIR.glob("*.jsonl")):
        if f.name.endswith(".bak"):
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("edit_required") == "error":
                    continue
                model = rec["model"]
                lang = rec["language"]
                ident = rec["identifier"]
                llm[(model, lang, ident)] = {
                    "edit_required": rec["edit_required"],
                    "harm": rec.get("harm_potential", "none"),
                    "cmqm": set(rec.get("cmqm_categories", [])),
                }
    return llm


def compute_metrics(h_edits, m_edits):
    """Compute classification metrics for edit-required binary task."""
    acc = accuracy_score(h_edits, m_edits)
    prec = precision_score(h_edits, m_edits, pos_label="yes", zero_division=0)
    rec = recall_score(h_edits, m_edits, pos_label="yes", zero_division=0)
    f1 = f1_score(h_edits, m_edits, pos_label="yes", zero_division=0)
    kappa = cohen_kappa_score(h_edits, m_edits)
    mcc = matthews_corrcoef(
        [1 if e == "yes" else 0 for e in h_edits],
        [1 if e == "yes" else 0 for e in m_edits],
    )
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "kappa": kappa, "mcc": mcc}


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def main():
    human = load_human_annotations()
    llm = load_llm_results()

    # Get available models
    models = sorted(set(k[0] for k in llm.keys()))
    models = [m for m in models if m in SHORT_NAMES]
    languages = sorted(HUMAN_FILES.keys())

    print("=" * 80)
    print("  HUMAN vs LLM JUDGE — COMPREHENSIVE COMPARISON")
    print("=" * 80)
    print(f"  Human annotations: {len(human)} items across {len(languages)} languages")
    print(f"  LLM models: {len(models)}")
    print()

    # ── 1. Edit-Required Agreement ──
    print("\n1. EDIT-REQUIRED AGREEMENT (all languages pooled)")
    print("-" * 80)
    print(f"  {'Model':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} "
          f"{'Kappa':>7} {'MCC':>6}  H_yes  M_yes  N")

    all_metrics = {}
    for model in models:
        short = SHORT_NAMES[model]
        h_edits, m_edits = [], []
        for lang in languages:
            for (l, ident), h in human.items():
                if l != lang:
                    continue
                key = (model, lang, ident)
                if key not in llm:
                    continue
                h_edits.append(h["edit_required"])
                m_edits.append(llm[key]["edit_required"])

        if not h_edits:
            continue
        metrics = compute_metrics(h_edits, m_edits)
        all_metrics[model] = metrics
        h_yes = sum(1 for e in h_edits if e == "yes")
        m_yes = sum(1 for e in m_edits if e == "yes")
        print(f"  {short:<20} {metrics['accuracy']:>6.3f} {metrics['precision']:>6.3f} "
              f"{metrics['recall']:>6.3f} {metrics['f1']:>6.3f} {metrics['kappa']:>7.3f} "
              f"{metrics['mcc']:>6.3f}  {h_yes:>5}  {m_yes:>5}  {len(h_edits):>4}")

    # ── 2. Per-Language Edit Agreement ──
    print("\n\n2. EDIT-REQUIRED F1 BY LANGUAGE")
    print("-" * 80)
    header = f"  {'Language':<25}" + "".join(f"{SHORT_NAMES[m]:>14}" for m in models)
    print(header)

    per_lang_f1 = defaultdict(dict)
    per_lang_kappa = defaultdict(dict)
    for lang in languages:
        row_f1 = f"  {lang:<25}"
        row_k = ""
        for model in models:
            short = SHORT_NAMES[model]
            h_edits, m_edits = [], []
            for (l, ident), h in human.items():
                if l != lang:
                    continue
                key = (model, lang, ident)
                if key not in llm:
                    continue
                h_edits.append(h["edit_required"])
                m_edits.append(llm[key]["edit_required"])
            if h_edits:
                f1 = f1_score(h_edits, m_edits, pos_label="yes", zero_division=0)
                kappa = cohen_kappa_score(h_edits, m_edits)
                per_lang_f1[lang][model] = f1
                per_lang_kappa[lang][model] = kappa
                row_f1 += f"{f1:>14.3f}"
            else:
                row_f1 += f"{'N/A':>14}"
        print(row_f1)

    # ── 3. CMQM Category Agreement ──
    print("\n\n3. CMQM CATEGORY AGREEMENT (edit=yes items, all languages)")
    print("-" * 80)
    print(f"  {'Model':<20} {'Jaccard':>8} {'exact':>7}  ", end="")
    for cid in CMQM_IDS:
        print(f"{cid[:10]:>12}", end="")
    print()

    cmqm_agreement = {}
    for model in models:
        short = SHORT_NAMES[model]
        jaccards = []
        exact = 0
        total = 0
        cat_tp = Counter()
        cat_fp = Counter()
        cat_fn = Counter()

        for lang in languages:
            for (l, ident), h in human.items():
                if l != lang:
                    continue
                key = (model, lang, ident)
                if key not in llm:
                    continue
                # Only compare where either flagged edit
                if h["edit_required"] != "yes" and llm[key]["edit_required"] != "yes":
                    continue

                total += 1
                h_cmqm = h["cmqm"]
                m_cmqm = llm[key]["cmqm"]
                jaccards.append(jaccard(h_cmqm, m_cmqm))
                if h_cmqm == m_cmqm:
                    exact += 1

                for cid in CMQM_IDS:
                    in_h = cid in h_cmqm
                    in_m = cid in m_cmqm
                    if in_h and in_m:
                        cat_tp[cid] += 1
                    elif in_m and not in_h:
                        cat_fp[cid] += 1
                    elif in_h and not in_m:
                        cat_fn[cid] += 1

        if not jaccards:
            continue
        mean_j = np.mean(jaccards)
        exact_pct = exact / total if total else 0
        cmqm_agreement[model] = {"jaccard": mean_j, "exact": exact_pct}

        print(f"  {short:<20} {mean_j:>8.3f} {exact_pct:>6.1%}  ", end="")
        for cid in CMQM_IDS:
            tp = cat_tp[cid]
            fp = cat_fp[cid]
            fn = cat_fn[cid]
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            print(f"{f1:>12.3f}", end="")
        print()

    # ── 4. Harm Agreement ──
    print("\n\n4. CLINICAL HARM AGREEMENT (edit=yes items)")
    print("-" * 80)
    harm_levels = ["none", "low", "moderate", "high"]
    print(f"  {'Model':<20} {'Acc':>6} {'Kappa':>7}  Distribution (Human -> Model)")

    for model in models:
        short = SHORT_NAMES[model]
        h_harms, m_harms = [], []
        for lang in languages:
            for (l, ident), h in human.items():
                if l != lang or h["edit_required"] != "yes":
                    continue
                key = (model, lang, ident)
                if key not in llm or llm[key]["edit_required"] != "yes":
                    continue
                h_harms.append(h["harm"])
                m_harms.append(llm[key]["harm"])

        if not h_harms:
            continue
        acc = accuracy_score(h_harms, m_harms)
        # Kappa needs at least 2 classes
        try:
            kappa = cohen_kappa_score(h_harms, m_harms)
        except Exception:
            kappa = 0.0
        h_dist = Counter(h_harms)
        m_dist = Counter(m_harms)
        dist_str = " | ".join(
            f"{h}: {h_dist.get(h,0)}->{m_dist.get(h,0)}"
            for h in harm_levels if h_dist.get(h, 0) + m_dist.get(h, 0) > 0
        )
        print(f"  {short:<20} {acc:>6.3f} {kappa:>7.3f}  {dist_str}")

    # ── 5. Human edit rate vs LLM edit rate per language ──
    print("\n\n5. EDIT RATE COMPARISON BY LANGUAGE")
    print("-" * 80)
    header = f"  {'Language':<25} {'Human%':>7}" + "".join(
        f"{SHORT_NAMES[m]:>14}" for m in models
    )
    print(header)

    lang_edit_rates = {}
    for lang in languages:
        h_items = [(l, i) for (l, i) in human if l == lang]
        h_yes = sum(1 for (l, i) in h_items if human[(l, i)]["edit_required"] == "yes")
        h_rate = h_yes / len(h_items) if h_items else 0
        row = f"  {lang:<25} {h_rate:>6.1%}"
        lang_edit_rates[lang] = {"human": h_rate}

        for model in models:
            short = SHORT_NAMES[model]
            m_yes = 0
            m_total = 0
            for (l, i) in h_items:
                key = (model, lang, i)
                if key in llm:
                    m_total += 1
                    if llm[key]["edit_required"] == "yes":
                        m_yes += 1
            m_rate = m_yes / m_total if m_total else 0
            lang_edit_rates[lang][model] = m_rate
            row += f"{m_rate:>13.1%}"
        print(row)

    # ── CHARTS ──
    print("\n\nGenerating charts...")
    _generate_charts(models, languages, all_metrics, per_lang_f1,
                     per_lang_kappa, lang_edit_rates, human, llm)


def _generate_charts(models, languages, all_metrics, per_lang_f1,
                     per_lang_kappa, lang_edit_rates, human, llm):
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle("Human (Atlas Professional) vs LLM Judges", fontsize=16, fontweight="bold")

    short_labels = [SHORT_NAMES[m] for m in models]
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    # 1. Overall metrics bar chart
    ax = axes[0, 0]
    metric_names = ["accuracy", "precision", "recall", "f1", "kappa", "mcc"]
    x = np.arange(len(metric_names))
    w = 0.8 / len(models)
    for i, model in enumerate(models):
        if model not in all_metrics:
            continue
        vals = [all_metrics[model][m] for m in metric_names]
        ax.bar(x + i * w - 0.4 + w/2, vals, w, label=SHORT_NAMES[model], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=30, ha="right")
    ax.set_ylim(-0.2, 1.05)
    ax.set_title("Edit-Required: Overall Metrics")
    ax.legend(fontsize=6, ncol=2)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # 2. F1 heatmap by language
    ax = axes[0, 1]
    f1_matrix = np.zeros((len(languages), len(models)))
    for i, lang in enumerate(languages):
        for j, model in enumerate(models):
            f1_matrix[i, j] = per_lang_f1.get(lang, {}).get(model, 0)
    im = ax.imshow(f1_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(languages)))
    ax.set_yticklabels(languages, fontsize=8)
    for i in range(len(languages)):
        for j in range(len(models)):
            ax.text(j, i, f"{f1_matrix[i,j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Edit-Required F1 by Language")

    # 3. Edit rate comparison (human vs models)
    ax = axes[0, 2]
    x = np.arange(len(languages))
    h_rates = [lang_edit_rates[l]["human"] for l in languages]
    ax.bar(x, h_rates, 0.12, label="Human", color="black", alpha=0.8)
    for i, model in enumerate(models):
        m_rates = [lang_edit_rates[l].get(model, 0) for l in languages]
        ax.bar(x + 0.12 * (i + 1), m_rates, 0.12, label=SHORT_NAMES[model], color=colors[i])
    ax.set_xticks(x + 0.12 * len(models) / 2)
    ax.set_xticklabels([l[:8] for l in languages], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Edit Rate")
    ax.set_title("Edit Rate: Human vs Models")
    ax.legend(fontsize=5, ncol=2)

    # 4. CMQM per-category F1 (grouped by category)
    ax = axes[1, 0]
    cat_f1s = defaultdict(dict)
    for model in models:
        for cid in CMQM_IDS:
            tp, fp, fn = 0, 0, 0
            for lang in languages:
                for (l, ident), h in human.items():
                    if l != lang:
                        continue
                    key = (model, lang, ident)
                    if key not in llm:
                        continue
                    if h["edit_required"] != "yes" and llm[key]["edit_required"] != "yes":
                        continue
                    in_h = cid in h["cmqm"]
                    in_m = cid in llm[key]["cmqm"]
                    if in_h and in_m:
                        tp += 1
                    elif in_m:
                        fp += 1
                    elif in_h:
                        fn += 1
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            cat_f1s[cid][model] = f1

    x = np.arange(len(CMQM_IDS))
    w = 0.8 / len(models)
    for i, model in enumerate(models):
        vals = [cat_f1s[cid].get(model, 0) for cid in CMQM_IDS]
        ax.bar(x + i * w - 0.4 + w/2, vals, w, label=SHORT_NAMES[model], color=colors[i])
    short_cmqm = ["clinical\nacc.", "ungrounded\ncontent", "negation\npolarity",
                   "linguistic\nquality", "patient\ncomm."]
    ax.set_xticks(x)
    ax.set_xticklabels(short_cmqm, fontsize=7)
    ax.set_ylabel("F1 Score")
    ax.set_title("CMQM Category F1 vs Human")
    ax.legend(fontsize=5, ncol=2)

    # 5. Confusion matrix for best model (highest kappa)
    ax = axes[1, 1]
    best_model = max(all_metrics, key=lambda m: all_metrics[m]["kappa"])
    best_short = SHORT_NAMES[best_model]
    h_edits, m_edits = [], []
    for lang in languages:
        for (l, ident), h in human.items():
            if l != lang:
                continue
            key = (best_model, lang, ident)
            if key not in llm:
                continue
            h_edits.append(h["edit_required"])
            m_edits.append(llm[key]["edit_required"])
    cm = confusion_matrix(h_edits, m_edits, labels=["yes", "no"])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Yes", "No"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Yes", "No"])
    ax.set_xlabel(best_short); ax.set_ylabel("Human")
    ax.set_title(f"Confusion Matrix ({best_short})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    # 6. Kappa heatmap by language
    ax = axes[1, 2]
    kappa_matrix = np.zeros((len(languages), len(models)))
    for i, lang in enumerate(languages):
        for j, model in enumerate(models):
            kappa_matrix[i, j] = per_lang_kappa.get(lang, {}).get(model, 0)
    im = ax.imshow(kappa_matrix, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.6)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(languages)))
    ax.set_yticklabels(languages, fontsize=8)
    for i in range(len(languages)):
        for j in range(len(models)):
            ax.text(j, i, f"{kappa_matrix[i,j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Cohen's Kappa by Language")

    plt.tight_layout()
    out = CHARTS_DIR / "human_vs_llm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")

    # ── Second figure: deeper CMQM analysis ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
    fig2.suptitle("CMQM Taxonomy Analysis — Human vs LLM Judges", fontsize=14, fontweight="bold")

    # A. CMQM category distribution: human vs all models
    ax = axes2[0]
    h_cats = Counter()
    m_cats_all = defaultdict(Counter)
    for lang in languages:
        for (l, ident), h in human.items():
            if l != lang or h["edit_required"] != "yes":
                continue
            for c in h["cmqm"]:
                h_cats[c] += 1
            for model in models:
                key = (model, lang, ident)
                if key in llm and llm[key]["edit_required"] == "yes":
                    for c in llm[key]["cmqm"]:
                        m_cats_all[model][c] += 1

    x = np.arange(len(CMQM_IDS))
    w = 0.8 / (len(models) + 1)
    ax.bar(x, [h_cats.get(c, 0) for c in CMQM_IDS], w, label="Human", color="black", alpha=0.8)
    for i, model in enumerate(models):
        vals = [m_cats_all[model].get(c, 0) for c in CMQM_IDS]
        ax.bar(x + (i + 1) * w, vals, w, label=SHORT_NAMES[model], color=colors[i])
    ax.set_xticks(x + w * len(models) / 2)
    ax.set_xticklabels(short_cmqm, fontsize=7)
    ax.set_ylabel("Count")
    ax.set_title("CMQM Category Distribution")
    ax.legend(fontsize=5, ncol=2)

    # B. Harm level distribution: human vs models
    ax = axes2[1]
    harm_levels = ["none", "low", "moderate", "high"]
    h_harm_dist = Counter()
    m_harm_dist = defaultdict(Counter)
    for lang in languages:
        for (l, ident), h in human.items():
            if l != lang or h["edit_required"] != "yes":
                continue
            h_harm_dist[h["harm"]] += 1
            for model in models:
                key = (model, lang, ident)
                if key in llm and llm[key]["edit_required"] == "yes":
                    m_harm_dist[model][llm[key]["harm"]] += 1

    x = np.arange(len(harm_levels))
    ax.bar(x, [h_harm_dist.get(h, 0) for h in harm_levels], w, label="Human", color="black", alpha=0.8)
    for i, model in enumerate(models):
        vals = [m_harm_dist[model].get(h, 0) for h in harm_levels]
        ax.bar(x + (i + 1) * w, vals, w, label=SHORT_NAMES[model], color=colors[i])
    ax.set_xticks(x + w * len(models) / 2)
    ax.set_xticklabels(harm_levels)
    ax.set_ylabel("Count")
    ax.set_title("Clinical Harm Distribution (edit=yes)")
    ax.legend(fontsize=5, ncol=2)

    # C. Scatter: human edit rate vs best-model F1 by language
    ax = axes2[2]
    for lang in languages:
        h_rate = lang_edit_rates[lang]["human"]
        # Average F1 across models for this language
        f1s = [per_lang_f1.get(lang, {}).get(m, 0) for m in models]
        avg_f1 = np.mean(f1s) if f1s else 0
        ax.scatter(h_rate, avg_f1, s=80, zorder=5)
        ax.annotate(lang[:8], (h_rate, avg_f1), fontsize=7, ha="left",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Human Edit Rate")
    ax.set_ylabel("Mean LLM F1 Score")
    ax.set_title("Human Edit Rate vs LLM Agreement")

    plt.tight_layout()
    out2 = CHARTS_DIR / "cmqm_taxonomy_analysis.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out2}")


if __name__ == "__main__":
    main()
