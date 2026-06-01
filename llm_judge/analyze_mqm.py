"""
MQM analysis: cross-model comparison, per-language breakdown,
CMQM-vs-MQM correlation, and publication-quality charts.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("llm_judge_results")
MQM_DIR = RESULTS_DIR / "mqm"
CHARTS_DIR = RESULTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Short names for display
SHORT_NAMES = {
    "Qwen/Qwen3-30B-A3B": "Qwen3-30B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama3.3-70B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama3.1-8B",
    "google/gemma-4-31B-it": "Gemma4-31B",
    "deepseek-ai/DeepSeek-V3-0324": "DeepSeek-V3",
    "CohereLabs/aya-expanse-32b": "AyaExpanse-32B",
}

# CMQM harm weights for comparison
CMQM_HARM_WEIGHTS = {"none": 0, "low": -1, "moderate": -5, "high": -25}


def load_mqm_results():
    """Load all MQM JSONL files into a dict of model_id -> list of records."""
    data = {}
    for f in sorted(MQM_DIR.glob("*.jsonl")):
        if f.name.endswith(".bak"):
            continue
        records = []
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("_parse_error"):
                    records.append(rec)
        model_id = records[0]["model"] if records else f.stem.replace("__", "/")
        data[model_id] = records
    return data


def load_cmqm_results(model_id, languages):
    """Load CMQM results for a model, filtered to languages and edit_required=yes."""
    safe = model_id.replace("/", "__").replace(".", "_")
    path = RESULTS_DIR / f"{safe}.jsonl"
    results = {}
    lang_set = set(languages)
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("language") in lang_set and rec.get("edit_required") == "yes":
                key = f"{rec['language']}|{rec['identifier']}"
                results[key] = rec
    return results


def main():
    mqm_data = load_mqm_results()
    languages = ["Chinese Mandarin", "Urdu"]

    print("=" * 70)
    print("  MQM ANALYSIS REPORT")
    print("=" * 70)

    # ── 1. Per-model summary ──
    print("\n1. PER-MODEL MQM SUMMARY")
    print("-" * 70)
    model_stats = {}
    for model_id, records in sorted(mqm_data.items()):
        short = SHORT_NAMES.get(model_id, model_id.split("/")[-1])
        scores = [r["mqm_score"] for r in records]
        n_errors = sum(len(r.get("errors", [])) for r in records)
        sevs = defaultdict(int)
        cats = defaultdict(int)
        for r in records:
            for e in r.get("errors", []):
                sevs[e["severity"]] += 1
                cats[e["category"]] += 1

        per_lang = defaultdict(list)
        for r in records:
            per_lang[r["language"]].append(r["mqm_score"])

        model_stats[model_id] = {
            "short": short, "scores": scores, "sevs": dict(sevs),
            "cats": dict(cats), "per_lang": dict(per_lang),
            "n_items": len(records), "n_errors": n_errors,
        }

        print(f"\n  {short} ({len(records)} items)")
        print(f"    Mean MQM: {np.mean(scores):.2f}  |  Median: {np.median(scores):.1f}  |  Range: [{min(scores)}, {max(scores)}]")
        print(f"    Errors: {n_errors} total  |  Per item: {n_errors/len(records):.2f}")
        print(f"    Severity: {sevs.get('critical',0)} critical, {sevs.get('major',0)} major, {sevs.get('minor',0)} minor")
        for lang in languages:
            ls = per_lang.get(lang, [])
            if ls:
                print(f"    {lang}: n={len(ls)}, mean={np.mean(ls):.2f}")

    # ── 2. Top error categories across models ──
    print(f"\n\n2. TOP ERROR CATEGORIES (all models combined)")
    print("-" * 70)
    global_cats = defaultdict(int)
    for stats in model_stats.values():
        for cat, cnt in stats["cats"].items():
            global_cats[cat] += cnt
    for cat, cnt in sorted(global_cats.items(), key=lambda x: -x[1])[:15]:
        print(f"  {cat:<40} {cnt:>5}")

    # ── 3. Cross-model MQM score comparison ──
    print(f"\n\n3. CROSS-MODEL MQM SCORE COMPARISON")
    print("-" * 70)
    # Find common items (scored by all models)
    all_keys_per_model = {}
    key_to_mqm = {}
    for model_id, records in mqm_data.items():
        keys = set()
        for r in records:
            k = f"{r['language']}|{r['identifier']}"
            keys.add(k)
            key_to_mqm[(model_id, k)] = r["mqm_score"]
        all_keys_per_model[model_id] = keys

    common_keys = None
    for keys in all_keys_per_model.values():
        common_keys = keys if common_keys is None else common_keys & keys
    print(f"  Items scored by ALL {len(mqm_data)} models: {len(common_keys)}")

    if len(common_keys) > 10:
        model_ids = sorted(mqm_data.keys())
        print(f"\n  Pairwise Pearson correlation on {len(common_keys)} common items:")
        corr_matrix = np.zeros((len(model_ids), len(model_ids)))
        for i, m1 in enumerate(model_ids):
            for j, m2 in enumerate(model_ids):
                s1 = [key_to_mqm[(m1, k)] for k in sorted(common_keys)]
                s2 = [key_to_mqm[(m2, k)] for k in sorted(common_keys)]
                corr_matrix[i, j] = np.corrcoef(s1, s2)[0, 1]
        short_names = [SHORT_NAMES.get(m, m.split("/")[-1]) for m in model_ids]
        header = "  " + " " * 18 + "  ".join(f"{s:>12}" for s in short_names)
        print(header)
        for i, name in enumerate(short_names):
            row = f"  {name:<18}" + "  ".join(f"{corr_matrix[i,j]:>12.3f}" for j in range(len(model_ids)))
            print(row)

    # ── 4. CMQM harm vs MQM score correlation ──
    print(f"\n\n4. CMQM HARM vs MQM SCORE CORRELATION")
    print("-" * 70)
    cmqm_mqm_pairs = {}
    for model_id in mqm_data:
        try:
            cmqm = load_cmqm_results(model_id, languages)
        except FileNotFoundError:
            continue
        pairs = []
        for r in mqm_data[model_id]:
            k = f"{r['language']}|{r['identifier']}"
            if k in cmqm:
                harm = cmqm[k].get("harm_potential", "none")
                cmqm_score = CMQM_HARM_WEIGHTS.get(harm, 0)
                pairs.append((cmqm_score, r["mqm_score"], harm))
        cmqm_mqm_pairs[model_id] = pairs
        short = SHORT_NAMES.get(model_id, model_id.split("/")[-1])

        if len(pairs) > 5:
            cmqm_scores = [p[0] for p in pairs]
            mqm_scores = [p[1] for p in pairs]
            corr = np.corrcoef(cmqm_scores, mqm_scores)[0, 1]
            harm_dist = defaultdict(list)
            for cs, ms, h in pairs:
                harm_dist[h].append(ms)
            print(f"\n  {short} (n={len(pairs)}, Pearson r={corr:.3f})")
            for h in ["none", "low", "moderate", "high"]:
                if h in harm_dist:
                    vals = harm_dist[h]
                    print(f"    harm={h:<10}: n={len(vals):>4}, mean MQM={np.mean(vals):.2f}")

    # ── 5. Per-language comparison ──
    print(f"\n\n5. PER-LANGUAGE MQM SCORES")
    print("-" * 70)
    for lang in languages:
        print(f"\n  {lang}:")
        for model_id, stats in sorted(model_stats.items(), key=lambda x: x[1]["short"]):
            ls = stats["per_lang"].get(lang, [])
            if ls:
                print(f"    {stats['short']:<20} n={len(ls):>4}  mean={np.mean(ls):.2f}  median={np.median(ls):.1f}")

    # ── CHARTS ──
    print(f"\n\nGenerating charts...")
    _generate_charts(model_stats, mqm_data, cmqm_mqm_pairs, common_keys, key_to_mqm, languages)
    print("Done.")


def _generate_charts(model_stats, mqm_data, cmqm_mqm_pairs, common_keys, key_to_mqm, languages):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("MQM Error Analysis — Chinese Mandarin & Urdu", fontsize=16, fontweight="bold")

    models_sorted = sorted(model_stats.keys(), key=lambda m: model_stats[m]["short"])
    short_labels = [model_stats[m]["short"] for m in models_sorted]
    colors = plt.cm.Set2(np.linspace(0, 1, len(models_sorted)))

    # 1. Mean MQM score per model
    ax = axes[0, 0]
    means = [np.mean(model_stats[m]["scores"]) for m in models_sorted]
    bars = ax.barh(short_labels, means, color=colors)
    ax.set_xlabel("Mean MQM Score")
    ax.set_title("Mean MQM Score by Model")
    ax.invert_xaxis()
    for bar, val in zip(bars, means):
        ax.text(val - 0.1, bar.get_y() + bar.get_height()/2, f"{val:.2f}",
                ha="right", va="center", fontsize=9)

    # 2. Severity distribution stacked bar
    ax = axes[0, 1]
    crit = [model_stats[m]["sevs"].get("critical", 0) for m in models_sorted]
    maj = [model_stats[m]["sevs"].get("major", 0) for m in models_sorted]
    minn = [model_stats[m]["sevs"].get("minor", 0) for m in models_sorted]
    x = np.arange(len(short_labels))
    ax.bar(x, crit, label="Critical", color="#d32f2f")
    ax.bar(x, maj, bottom=crit, label="Major", color="#ff9800")
    ax.bar(x, minn, bottom=[c+m for c,m in zip(crit, maj)], label="Minor", color="#4caf50")
    ax.set_xticks(x); ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Error Count")
    ax.set_title("Error Severity Distribution")
    ax.legend(fontsize=8)

    # 3. Top error categories (global)
    ax = axes[0, 2]
    global_cats = defaultdict(int)
    for stats in model_stats.values():
        for cat, cnt in stats["cats"].items():
            global_cats[cat] += cnt
    top_cats = sorted(global_cats.items(), key=lambda x: -x[1])[:10]
    cat_names = [c[0] for c in top_cats]
    cat_counts = [c[1] for c in top_cats]
    ax.barh(cat_names[::-1], cat_counts[::-1], color="#2196F3")
    ax.set_xlabel("Count (all models)")
    ax.set_title("Top 10 MQM Error Categories")

    # 4. Per-language mean MQM
    ax = axes[1, 0]
    w = 0.35
    x = np.arange(len(short_labels))
    for li, lang in enumerate(languages):
        vals = []
        for m in models_sorted:
            ls = model_stats[m]["per_lang"].get(lang, [])
            vals.append(np.mean(ls) if ls else 0)
        offset = -w/2 + li * w
        ax.bar(x + offset, vals, w, label=lang)
    ax.set_xticks(x); ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean MQM Score")
    ax.set_title("MQM Score by Language")
    ax.legend(fontsize=8)

    # 5. CMQM harm vs MQM score (box plot for one model)
    ax = axes[1, 1]
    # Pick model with most data
    best_model = max(cmqm_mqm_pairs.keys(), key=lambda m: len(cmqm_mqm_pairs.get(m, [])))
    pairs = cmqm_mqm_pairs[best_model]
    harm_levels = ["none", "low", "moderate", "high"]
    box_data = []
    box_labels = []
    for h in harm_levels:
        vals = [p[1] for p in pairs if p[2] == h]
        if vals:
            box_data.append(vals)
            box_labels.append(f"{h}\n(n={len(vals)})")
        else:
            box_data.append([0])
            box_labels.append(f"{h}\n(n=0)")
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    box_colors = ["#4caf50", "#ffeb3b", "#ff9800", "#d32f2f"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("MQM Score")
    short = SHORT_NAMES.get(best_model, best_model.split("/")[-1])
    ax.set_title(f"CMQM Harm vs MQM Score ({short})")

    # 6. Cross-model correlation heatmap on common items
    ax = axes[1, 2]
    if common_keys and len(common_keys) > 10:
        model_ids = sorted(mqm_data.keys())
        n = len(model_ids)
        corr_matrix = np.zeros((n, n))
        for i, m1 in enumerate(model_ids):
            for j, m2 in enumerate(model_ids):
                s1 = [key_to_mqm[(m1, k)] for k in sorted(common_keys)]
                s2 = [key_to_mqm[(m2, k)] for k in sorted(common_keys)]
                corr_matrix[i, j] = np.corrcoef(s1, s2)[0, 1]
        sn = [SHORT_NAMES.get(m, m.split("/")[-1]) for m in model_ids]
        im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
        ax.set_xticks(range(n)); ax.set_xticklabels(sn, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(sn, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"MQM Score Correlation (n={len(common_keys)})")
    else:
        ax.text(0.5, 0.5, "Not enough\ncommon items", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("MQM Score Correlation")

    plt.tight_layout()
    out = CHARTS_DIR / "mqm_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")

    # ── Second figure: per-model error category breakdown ──
    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))
    fig2.suptitle("MQM Error Categories by Model", fontsize=16, fontweight="bold")

    top_global = [c[0] for c in sorted(global_cats.items(), key=lambda x: -x[1])[:8]]

    for idx, model_id in enumerate(models_sorted[:6]):
        ax = axes2[idx // 3, idx % 3]
        stats = model_stats[model_id]
        cat_vals = [stats["cats"].get(c, 0) for c in top_global]
        short_cat_labels = [c.replace("accuracy/", "acc/").replace("fluency/", "flu/")
                           .replace("terminology/", "term/").replace("style/", "sty/")
                           for c in top_global]
        ax.barh(short_cat_labels[::-1], cat_vals[::-1], color=colors[idx])
        ax.set_title(stats["short"], fontsize=11)
        ax.set_xlabel("Count")

    plt.tight_layout()
    out2 = CHARTS_DIR / "mqm_categories_by_model.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out2}")


if __name__ == "__main__":
    main()
