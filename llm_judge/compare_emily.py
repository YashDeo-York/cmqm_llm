"""Compare Qwen3-30B and Qwen3-8B with Emily's human annotations (Chinese Mandarin)."""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from pathlib import Path

RESULTS_DIR = Path("llm_judge_results")
CHARTS_DIR = RESULTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_chinese(filename):
    results = {}
    with open(RESULTS_DIR / filename, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("language") == "Chinese Mandarin":
                results[row["identifier"]] = row
    return results


def emily_edit(row):
    val = str(row["Edit Required"]).strip().lower()
    return "yes" if val == "yes" else "no"


def emily_cmqm(row):
    val = str(row["CMQM Categories"]).strip()
    if val == "nan" or val == "":
        return []
    return [c.strip() for c in val.replace(";", ",").split(",") if c.strip()]


def emily_harm(row):
    val = str(row["Clinical Harm Potential"]).strip().lower()
    if val == "nan" or val == "":
        return "none"
    return val


def main():
    emily = pd.read_csv("emily.csv", encoding="utf-8-sig")
    qwen30 = load_model_chinese("Qwen__Qwen3-30B-A3B.jsonl")
    qwen8 = load_model_chinese("Qwen__Qwen3-8B.jsonl")

    # Build comparison records
    records = []
    for _, erow in emily.iterrows():
        eid = erow["Identifier"].strip()
        e_edit = emily_edit(erow)
        e_cmqm = set(emily_cmqm(erow))
        e_harm = emily_harm(erow)

        for model_name, model_data in [("Qwen3-30B", qwen30), ("Qwen3-8B", qwen8)]:
            mrow = model_data.get(eid)
            if not mrow:
                continue
            m_edit = mrow["edit_required"]
            m_cmqm = set(mrow.get("cmqm_categories", []))
            m_harm = mrow.get("harm_potential", "none")

            if e_cmqm or m_cmqm:
                jaccard = len(e_cmqm & m_cmqm) / len(e_cmqm | m_cmqm)
            else:
                jaccard = 1.0

            records.append({
                "identifier": eid,
                "model": model_name,
                "emily_edit": e_edit,
                "model_edit": m_edit,
                "emily_cmqm": e_cmqm,
                "model_cmqm": m_cmqm,
                "emily_harm": e_harm,
                "model_harm": m_harm,
                "cmqm_jaccard": jaccard,
                "edit_agree": e_edit == m_edit,
            })

    df = pd.DataFrame(records)

    # ── Print detailed stats ──
    for model in ["Qwen3-30B", "Qwen3-8B"]:
        mdf = df[df["model"] == model]
        e = mdf["emily_edit"].values
        m = mdf["model_edit"].values

        print(f"\n{'='*60}")
        print(f"  {model} vs Emily (Chinese Mandarin, n={len(mdf)})")
        print(f"{'='*60}")

        print(f"\nEmily: {sum(e=='yes')} edits needed, {sum(e=='no')} no edit")
        print(f"{model}: {sum(m=='yes')} edits flagged, {sum(m=='no')} no edit")

        print(f"\nClassification Report (edit=yes as positive):")
        print(classification_report(e, m, labels=["yes", "no"],
                                    target_names=["Edit Yes", "Edit No"]))

        kappa = cohen_kappa_score(e, m)
        print(f"Cohen's Kappa: {kappa:.3f}")

        cm = confusion_matrix(e, m, labels=["yes", "no"])
        print(f"\nConfusion Matrix (rows=Emily, cols={model}):")
        print(f"             Pred Yes  Pred No")
        print(f"  True Yes:  {cm[0,0]:>8}  {cm[0,1]:>7}")
        print(f"  True No:   {cm[1,0]:>8}  {cm[1,1]:>7}")

        edit_rows = mdf[(mdf["emily_edit"] == "yes") | (mdf["model_edit"] == "yes")]
        mean_jaccard = edit_rows["cmqm_jaccard"].mean()
        print(f"\nCMQM Jaccard (among flagged rows): {mean_jaccard:.3f} (n={len(edit_rows)})")

        all_cats = set()
        for cats in edit_rows["emily_cmqm"]:
            all_cats |= cats
        for cats in edit_rows["model_cmqm"]:
            all_cats |= cats

        print(f"\nCMQM Category Breakdown (flagged rows):")
        print(f"  {'Category':<25} {'Emily':>6} {model:>10} {'Both':>6}")
        for cat in sorted(all_cats):
            e_count = sum(cat in s for s in edit_rows["emily_cmqm"])
            m_count = sum(cat in s for s in edit_rows["model_cmqm"])
            both = sum((cat in ec) and (cat in mc)
                       for ec, mc in zip(edit_rows["emily_cmqm"], edit_rows["model_cmqm"]))
            print(f"  {cat:<25} {e_count:>6} {m_count:>10} {both:>6}")

        print(f"\nHarm Distribution:")
        print(f"  {'Level':<12} {'Emily':>6} {model:>10}")
        all_harms = sorted(set(mdf["emily_harm"]) | set(mdf["model_harm"]))
        for h in all_harms:
            e_c = sum(mdf["emily_harm"] == h)
            m_c = sum(mdf["model_harm"] == h)
            print(f"  {h:<12} {e_c:>6} {m_c:>10}")

    # ── Charts ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Qwen3 vs Emily \u2014 Chinese Mandarin Translation Quality",
                 fontsize=16, fontweight="bold")

    w = 0.35
    all_cats_list = ["clinical_accuracy", "ungrounded_content", "negation_polarity",
                     "linguistic_quality", "patient_communication"]
    short_cats = ["Clinical\nAccuracy", "Ungrounded\nContent", "Negation\nPolarity",
                  "Linguistic\nQuality", "Patient\nComm."]

    # 1. Confusion matrix Qwen3-30B
    ax = axes[0, 0]
    mdf = df[df["model"] == "Qwen3-30B"]
    cm = confusion_matrix(mdf["emily_edit"], mdf["model_edit"], labels=["yes", "no"])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Yes", "No"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Yes", "No"])
    ax.set_xlabel("Qwen3-30B"); ax.set_ylabel("Emily")
    ax.set_title("Edit Required \u2014 Qwen3-30B")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    # 2. Confusion matrix Qwen3-8B
    ax = axes[0, 1]
    mdf = df[df["model"] == "Qwen3-8B"]
    cm = confusion_matrix(mdf["emily_edit"], mdf["model_edit"], labels=["yes", "no"])
    ax.imshow(cm, cmap="Oranges")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Yes", "No"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Yes", "No"])
    ax.set_xlabel("Qwen3-8B"); ax.set_ylabel("Emily")
    ax.set_title("Edit Required \u2014 Qwen3-8B")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=18,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    # 3. Metrics comparison
    ax = axes[0, 2]
    metrics = {}
    for model in ["Qwen3-30B", "Qwen3-8B"]:
        mdf = df[df["model"] == model]
        e = mdf["emily_edit"].values
        m = mdf["model_edit"].values
        tp = np.sum((e == "yes") & (m == "yes"))
        fp = np.sum((e == "no") & (m == "yes"))
        fn = np.sum((e == "yes") & (m == "no"))
        acc = np.mean(e == m)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        kappa = cohen_kappa_score(e, m)
        metrics[model] = {"Accuracy": acc, "Precision": prec, "Recall": rec,
                          "F1": f1, "Kappa": kappa}

    x = np.arange(5)
    labels_list = ["Accuracy", "Precision", "Recall", "F1", "Kappa"]
    vals_30 = [metrics["Qwen3-30B"][l] for l in labels_list]
    vals_8 = [metrics["Qwen3-8B"][l] for l in labels_list]
    bars1 = ax.bar(x - w / 2, vals_30, w, label="Qwen3-30B", color="#2196F3")
    bars2 = ax.bar(x + w / 2, vals_8, w, label="Qwen3-8B", color="#FF9800")
    ax.set_xticks(x); ax.set_xticklabels(labels_list, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Agreement Metrics vs Emily")
    ax.legend()
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", fontsize=8)

    # 4. CMQM categories Qwen3-30B vs Emily
    ax = axes[1, 0]
    mdf30 = df[df["model"] == "Qwen3-30B"]
    edit_rows = mdf30[(mdf30["emily_edit"] == "yes") | (mdf30["model_edit"] == "yes")]
    e_counts = [sum(c in s for s in edit_rows["emily_cmqm"]) for c in all_cats_list]
    m_counts = [sum(c in s for s in edit_rows["model_cmqm"]) for c in all_cats_list]
    x2 = np.arange(len(all_cats_list))
    ax.bar(x2 - w / 2, e_counts, w, label="Emily", color="#4CAF50")
    ax.bar(x2 + w / 2, m_counts, w, label="Qwen3-30B", color="#2196F3")
    ax.set_xticks(x2); ax.set_xticklabels(short_cats, fontsize=8)
    ax.set_title("CMQM Categories \u2014 Qwen3-30B vs Emily")
    ax.legend(); ax.set_ylabel("Count")

    # 5. CMQM categories Qwen3-8B vs Emily
    ax = axes[1, 1]
    mdf8 = df[df["model"] == "Qwen3-8B"]
    edit_rows8 = mdf8[(mdf8["emily_edit"] == "yes") | (mdf8["model_edit"] == "yes")]
    e_counts8 = [sum(c in s for s in edit_rows8["emily_cmqm"]) for c in all_cats_list]
    m_counts8 = [sum(c in s for s in edit_rows8["model_cmqm"]) for c in all_cats_list]
    ax.bar(x2 - w / 2, e_counts8, w, label="Emily", color="#4CAF50")
    ax.bar(x2 + w / 2, m_counts8, w, label="Qwen3-8B", color="#FF9800")
    ax.set_xticks(x2); ax.set_xticklabels(short_cats, fontsize=8)
    ax.set_title("CMQM Categories \u2014 Qwen3-8B vs Emily")
    ax.legend(); ax.set_ylabel("Count")

    # 6. Harm distribution
    ax = axes[1, 2]
    harm_levels = ["none", "minor", "low", "moderate", "major", "high"]
    mdf30 = df[df["model"] == "Qwen3-30B"]
    e_harms = [sum(mdf30["emily_harm"] == h) for h in harm_levels]
    m30_harms = [sum(mdf30["model_harm"] == h) for h in harm_levels]
    mdf8 = df[df["model"] == "Qwen3-8B"]
    m8_harms = [sum(mdf8["model_harm"] == h) for h in harm_levels]
    x3 = np.arange(len(harm_levels))
    bw = 0.25
    ax.bar(x3 - bw, e_harms, bw, label="Emily", color="#4CAF50")
    ax.bar(x3, m30_harms, bw, label="Qwen3-30B", color="#2196F3")
    ax.bar(x3 + bw, m8_harms, bw, label="Qwen3-8B", color="#FF9800")
    ax.set_xticks(x3); ax.set_xticklabels(harm_levels)
    ax.set_title("Clinical Harm Distribution")
    ax.legend(); ax.set_ylabel("Count")

    plt.tight_layout()
    out = CHARTS_DIR / "qwen3_vs_emily_chinese.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
