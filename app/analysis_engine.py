"""
Core analysis engine for LLM Translation Quality Judge evaluation.
Provides all computation separated from UI concerns.
"""

import json, itertools
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    cohen_kappa_score, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
)

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_SHORT = {
    "CohereLabs__aya-expanse-32b":              "Aya-32B",
    "Qwen__Qwen3-30B-A3B":                     "Qwen3-30B",
    "Qwen__Qwen3-8B":                          "Qwen3-8B",
    "deepseek-ai__DeepSeek-V3-0324":            "DeepSeek-V3",
    "google__gemma-4-31B-it":                   "Gemma4-31B",
    "meta-llama__Llama-3_1-8B-Instruct":        "Llama3.1-8B",
    "meta-llama__Llama-3_3-70B-Instruct":       "Llama3.3-70B",
    "utter-project__EuroLLM-22B-Instruct-2512": "EuroLLM-22B",
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

HARM_MAP = {"none": 0, "low": 1, "moderate": 2, "high": 3}
HARM_MAP_HUMAN = {"none": 0, "minor": 1, "low": 1, "moderate": 2, "major": 3, "high": 3}


def load_judge_jsonl(fp, model_key=None):
    """Load a single JSONL judge file into a list of dicts."""
    if model_key is None:
        model_key = Path(fp).stem
    rows = []
    with open(fp, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            r["model_key"] = model_key
            r["model_short"] = MODELS_SHORT.get(model_key, model_key)
            r["lang_short"] = LANG_SHORT.get(r.get("language", ""), r.get("language", ""))
            r["edit_binary"] = 1 if r.get("edit_required") == "yes" else 0
            r["harm_ord"] = HARM_MAP.get(r.get("harm_potential", "none"), 0)
            cats = r.get("cmqm_categories") or []
            for c in CMQM_CATS:
                r[f"cmqm_{c}"] = 1 if c in cats else 0
            rows.append(r)
    return rows


def load_all_judges(results_dir):
    """Load all JSONL judge files from a directory."""
    results_dir = Path(results_dir)
    all_rows = []
    for fp in sorted(results_dir.glob("*.jsonl")):
        if ".bak" in fp.name:
            continue
        all_rows.extend(load_judge_jsonl(fp))
    return pd.DataFrame(all_rows)


def load_judge_from_upload(file_obj, model_name=None):
    """Load judge data from an uploaded file object (BytesIO/StringIO)."""
    content = file_obj.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    rows = []
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if model_name:
            r["model_key"] = model_name
            r["model_short"] = model_name
        else:
            r["model_short"] = r.get("model", "uploaded_judge")
            r["model_key"] = r["model_short"]
        r["lang_short"] = LANG_SHORT.get(r.get("language", ""), r.get("language", ""))
        r["edit_binary"] = 1 if r.get("edit_required") == "yes" else 0
        r["harm_ord"] = HARM_MAP.get(r.get("harm_potential", "none"), 0)
        cats = r.get("cmqm_categories") or []
        for c in CMQM_CATS:
            r[f"cmqm_{c}"] = 1 if c in cats else 0
        rows.append(r)
    return pd.DataFrame(rows)


def load_human_annotations(file_obj, file_name=""):
    """Load human annotations from uploaded CSV or XLSX."""
    if isinstance(file_obj, (str, Path)):
        ext = str(file_obj).lower()
        if ext.endswith(".xlsx"):
            df = pd.read_excel(file_obj)
        else:
            df = pd.read_csv(file_obj)
    else:
        if file_name.lower().endswith(".xlsx"):
            df = pd.read_excel(file_obj)
        else:
            df = pd.read_csv(file_obj)

    # Standardise column names
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "identifier":
            col_map[c] = "identifier"
        elif cl == "topic key":
            col_map[c] = "topic_key"
        elif cl == "edit required":
            col_map[c] = "Edit Required"
        elif cl == "clinical harm potential":
            col_map[c] = "Clinical Harm Potential"
        elif cl == "cmqm categories":
            col_map[c] = "CMQM Categories"
        elif cl == "english source":
            col_map[c] = "English Source"
        elif cl == "machine translation":
            col_map[c] = "Machine Translation"
    df.rename(columns=col_map, inplace=True)

    # Binary edit required (blank / NaN = no)
    df["edit_binary_human"] = df["Edit Required"].apply(
        lambda x: 1 if str(x).strip().lower() == "yes" else 0
    )

    # Harm ordinal
    df["harm_ord_human"] = df.get("Clinical Harm Potential", pd.Series(dtype=float)).map(
        lambda x: HARM_MAP_HUMAN.get(str(x).strip().lower(), np.nan)
    )

    # CMQM binary flags
    for c in CMQM_CATS:
        df[f"cmqm_{c}_human"] = 0
    if "CMQM Categories" in df.columns:
        for idx, val in df["CMQM Categories"].items():
            if pd.notna(val):
                for c in CMQM_CATS:
                    if c in str(val).lower():
                        df.loc[idx, f"cmqm_{c}_human"] = 1

    return df


# ─── Cross-Judge Analysis Functions ─────────────────────────────────────

def edit_rate_by_judge_language(df):
    """Pivot table of edit-required rate by judge x language."""
    return df.groupby(["model_short", "lang_short"])["edit_binary"].mean().unstack()


def overall_edit_rate_by_model(df):
    agg = df.groupby("model_short")["edit_binary"].agg(["mean", "sum", "count"])
    agg.columns = ["edit_rate", "edits", "total"]
    return agg.sort_values("edit_rate")


def overall_edit_rate_by_language(df):
    agg = df.groupby("lang_short")["edit_binary"].agg(["mean", "sum", "count"])
    agg.columns = ["edit_rate", "edits", "total"]
    return agg.sort_values("edit_rate")


def harm_rate_by_judge_language(df):
    return df.groupby(["model_short", "lang_short"])["harm_ord"].apply(
        lambda x: (x > 0).mean()
    ).unstack()


def harm_level_distribution(df):
    """Full harm level distribution per judge."""
    df_copy = df.copy()
    df_copy["harm_label"] = df_copy["harm_potential"].fillna("none")
    return df_copy.groupby(["model_short", "harm_label"]).size().unstack(fill_value=0)


def cmqm_by_model(df):
    flagged = df[df["edit_binary"] == 1]
    if len(flagged) == 0:
        return pd.DataFrame()
    cols = [f"cmqm_{c}" for c in CMQM_CATS]
    result = flagged.groupby("model_short")[cols].mean()
    result.columns = CMQM_CATS
    return result


def cmqm_by_language(df):
    flagged = df[df["edit_binary"] == 1]
    if len(flagged) == 0:
        return pd.DataFrame()
    cols = [f"cmqm_{c}" for c in CMQM_CATS]
    result = flagged.groupby("lang_short")[cols].mean()
    result.columns = CMQM_CATS
    return result


def cmqm_by_model_language(df):
    """CMQM breakdown by model AND language for deep drill-down."""
    flagged = df[df["edit_binary"] == 1]
    if len(flagged) == 0:
        return pd.DataFrame()
    cols = [f"cmqm_{c}" for c in CMQM_CATS]
    result = flagged.groupby(["model_short", "lang_short"])[cols].mean()
    result.columns = CMQM_CATS
    return result


def edit_rate_by_row_type(df):
    return df.groupby(["model_short", "row_type"])["edit_binary"].mean().unstack()


def edit_rate_by_topic(df):
    agg = df.groupby("topic_key")["edit_binary"].agg(["mean", "count"])
    agg.columns = ["edit_rate", "count"]
    return agg.sort_values("edit_rate", ascending=False)


def parse_error_rates(df):
    return df.groupby("model_short").agg(
        parse_error_pct=("_parse_error", lambda x: x.mean() * 100),
        parse_repaired_pct=("_parse_repaired", lambda x: x.mean() * 100),
    )


def pairwise_kappa_matrix(df):
    """Cohen's kappa between every pair of judges on edit_required."""
    models = sorted(df["model_short"].unique())
    pivot = df.pivot_table(
        index=["lang_short", "identifier", "topic_key"],
        columns="model_short",
        values="edit_binary",
        aggfunc="first",
    ).dropna()

    mat = pd.DataFrame(np.nan, index=models, columns=models)
    for m1, m2 in itertools.combinations(models, 2):
        k = cohen_kappa_score(pivot[m1], pivot[m2])
        mat.loc[m1, m2] = k
        mat.loc[m2, m1] = k
    for m in models:
        mat.loc[m, m] = 1.0
    return mat, pivot


def per_language_kappa(df):
    """Mean pairwise kappa per language."""
    models = sorted(df["model_short"].unique())
    pivot = df.pivot_table(
        index=["lang_short", "identifier", "topic_key"],
        columns="model_short",
        values="edit_binary",
        aggfunc="first",
    ).dropna()

    results = {}
    for lang in sorted(df["lang_short"].unique()):
        if lang not in pivot.index.get_level_values(0):
            continue
        sub = pivot.loc[lang]
        ks = []
        for m1, m2 in itertools.combinations(models, 2):
            try:
                ks.append(cohen_kappa_score(sub[m1], sub[m2]))
            except Exception:
                pass
        if ks:
            results[lang] = {
                "mean_kappa": np.mean(ks), "std": np.std(ks),
                "min": np.min(ks), "max": np.max(ks),
            }
    return pd.DataFrame(results).T


def unanimous_agreement(df):
    """Proportion of items where all judges agree, per language."""
    models = sorted(df["model_short"].unique())
    n_models = len(models)
    pivot = df.pivot_table(
        index=["lang_short", "identifier", "topic_key"],
        columns="model_short",
        values="edit_binary",
        aggfunc="first",
    ).dropna()

    results = {}
    for lang in sorted(pivot.index.get_level_values(0).unique()):
        sub = pivot.loc[lang]
        n = len(sub)
        row_sums = sub.sum(axis=1)
        unanimous = ((row_sums == 0) | (row_sums == n_models)).sum()
        results[lang] = {"unanimous": int(unanimous), "total": n, "pct": unanimous / n * 100}
    return pd.DataFrame(results).T


def majority_vote_analysis(df):
    """Compute majority-vote labels and agreement stats."""
    models = sorted(df["model_short"].unique())
    n_models = len(models)
    pivot = df.pivot_table(
        index=["lang_short", "identifier", "topic_key"],
        columns="model_short",
        values="edit_binary",
        aggfunc="first",
    ).dropna()

    pivot["vote_sum"] = pivot[models].sum(axis=1)
    pivot["majority_edit"] = (pivot["vote_sum"] > n_models / 2).astype(int)
    pivot["agreement_ratio"] = pivot[models].apply(
        lambda row: max(row.sum(), n_models - row.sum()) / n_models, axis=1
    )
    return pivot


# ─── Human vs Judge Comparison ──────────────────────────────────────────

def compare_human_vs_judges(human_df, judge_df, language_filter):
    """Full comparison of human annotations against all judges for one language."""
    judge_lang = judge_df[judge_df["language"] == language_filter].copy()
    results = []

    for model in sorted(judge_lang["model_short"].unique()):
        jm = judge_lang[judge_lang["model_short"] == model]
        merged = human_df.merge(
            jm[["identifier", "topic_key", "edit_binary", "harm_ord",
                "brief_rationale"] + [f"cmqm_{c}" for c in CMQM_CATS]],
            on=["identifier", "topic_key"],
            how="inner",
            suffixes=("_human", "_judge"),
        )
        if len(merged) == 0:
            continue

        y_h = merged["edit_binary_human"]
        y_j = merged["edit_binary"]

        kappa = cohen_kappa_score(y_h, y_j) if y_h.nunique() > 1 else 0.0
        acc = accuracy_score(y_h, y_j)
        prec = precision_score(y_h, y_j, zero_division=0)
        rec = recall_score(y_h, y_j, zero_division=0)
        f1 = f1_score(y_h, y_j, zero_division=0)

        cm = confusion_matrix(y_h, y_j, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        # McNemar's test
        if fp + fn > 0:
            chi2 = (abs(fp - fn) - 1) ** 2 / (fp + fn)
            p_val = 1 - stats.chi2.cdf(chi2, df=1)
        else:
            chi2, p_val = 0.0, 1.0

        # Prevalence and bias indices
        prevalence_idx = (tp + fn) / len(merged)
        bias_idx = (fp - fn) / len(merged)

        results.append({
            "Judge": model,
            "N": len(merged),
            "Human Edit %": round(y_h.mean() * 100, 1),
            "Judge Edit %": round(y_j.mean() * 100, 1),
            "Cohen's Kappa": round(kappa, 3),
            "Accuracy": round(acc, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "Specificity": round(spec, 3),
            "F1": round(f1, 3),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "McNemar Chi2": round(chi2, 2),
            "McNemar p": round(p_val, 4),
            "Prevalence Index": round(prevalence_idx, 3),
            "Bias Index": round(bias_idx, 3),
        })

    return pd.DataFrame(results)


def per_item_agreement_detail(human_df, judge_df, language_filter):
    """Return merged per-item dataframe showing human vs each judge."""
    judge_lang = judge_df[judge_df["language"] == language_filter]
    all_merged = []

    for model in sorted(judge_lang["model_short"].unique()):
        jm = judge_lang[judge_lang["model_short"] == model]
        merged = human_df.merge(
            jm[["identifier", "topic_key", "edit_binary", "harm_ord",
                "brief_rationale"] + [f"cmqm_{c}" for c in CMQM_CATS]],
            on=["identifier", "topic_key"],
            how="inner",
        )
        merged["judge_model"] = model
        merged["agree"] = (merged["edit_binary_human"] == merged["edit_binary"]).astype(int)
        all_merged.append(merged)

    if all_merged:
        return pd.concat(all_merged, ignore_index=True)
    return pd.DataFrame()


def cmqm_agreement_detail(human_df, judge_df, language_filter):
    """CMQM category-level agreement among items both flagged."""
    judge_lang = judge_df[judge_df["language"] == language_filter]
    results = []

    for model in sorted(judge_lang["model_short"].unique()):
        jm = judge_lang[judge_lang["model_short"] == model]
        merged = human_df.merge(
            jm[["identifier", "topic_key", "edit_binary"] + [f"cmqm_{c}" for c in CMQM_CATS]],
            on=["identifier", "topic_key"], how="inner",
        )
        both = merged[(merged["edit_binary_human"] == 1) & (merged["edit_binary"] == 1)]
        if len(both) < 2:
            continue

        row = {"Judge": model, "N_both_flagged": len(both)}
        for c in CMQM_CATS:
            hcol = f"cmqm_{c}_human"
            jcol = f"cmqm_{c}"
            if hcol in both.columns:
                row[f"{c}_agree"] = round((both[hcol] == both[jcol]).mean(), 3)
                row[f"{c}_human"] = int(both[hcol].sum())
                row[f"{c}_judge"] = int(both[jcol].sum())
        results.append(row)

    return pd.DataFrame(results)


def harm_agreement_detail(human_df, judge_df, language_filter):
    """Harm potential agreement between human and judges."""
    judge_lang = judge_df[judge_df["language"] == language_filter]
    results = []

    for model in sorted(judge_lang["model_short"].unique()):
        jm = judge_lang[judge_lang["model_short"] == model]
        merged = human_df.merge(
            jm[["identifier", "topic_key", "harm_ord"]],
            on=["identifier", "topic_key"], how="inner",
            suffixes=("_human", "_judge"),
        )
        valid = merged.dropna(subset=["harm_ord_human"])
        if len(valid) < 3:
            continue

        h_bin = (valid["harm_ord_human"] > 0).astype(int)
        j_bin = (valid["harm_ord"] > 0).astype(int)

        try:
            kappa = cohen_kappa_score(h_bin, j_bin)
        except Exception:
            kappa = np.nan

        agree = (h_bin == j_bin).mean()
        results.append({
            "Judge": model, "N": len(valid),
            "Agreement": round(agree, 3),
            "Kappa": round(kappa, 3) if not np.isnan(kappa) else "N/A",
            "Human Harm %": round(h_bin.mean() * 100, 1),
            "Judge Harm %": round(j_bin.mean() * 100, 1),
        })

    return pd.DataFrame(results)


def disagreement_examples(human_df, judge_df, language_filter, model_name, n=20):
    """Get examples where human and a specific judge disagree."""
    judge_lang = judge_df[
        (judge_df["language"] == language_filter) &
        (judge_df["model_short"] == model_name)
    ]
    merged = human_df.merge(
        judge_lang[["identifier", "topic_key", "edit_binary",
                     "brief_rationale", "machine_translation"]],
        on=["identifier", "topic_key"], how="inner",
        suffixes=("_human", "_judge"),
    )
    disagreements = merged[merged["edit_binary_human"] != merged["edit_binary"]]

    cols = ["identifier", "topic_key"]
    if "English Source" in disagreements.columns:
        cols.append("English Source")
    if "machine_translation" in disagreements.columns:
        cols.append("machine_translation")
    cols += ["edit_binary_human", "edit_binary", "brief_rationale"]

    return disagreements[cols].head(n)


# ─── Atlas Professional Annotations ───────────────────────────────────

ATLAS_FILES = {
    "Bengali":                 ("Bengali_atlas_clean.xlsx",              "Bengali",                "BN"),
    "Chinese Mandarin":        ("Chinese_Mandarin_atlas_clean.xlsx",     "Chinese Mandarin",       "ZH"),
    "French":                  ("French_atlas_clean.xlsx",               "French",                 "FR"),
    "Polish":                  ("Polish_atlas_clean.xlsx",               "Polish",                 "PL"),
    "Portuguese (Brazilian)":  ("Portuguese_Brazilian_atlas_clean.xlsx",  "Portuguese (Brazilian)", "PT-BR"),
    "Spanish":                 ("Spanish_atlas_clean.xlsx",              "Spanish",                "ES"),
    "Turkish":                 ("Turkish_atlas_clean.xlsx",              "Turkish",                "TR"),
    "Urdu":                    ("Urdu_atlas_clean.xlsx",                 "Urdu",                   "UR"),
}


def load_atlas_annotations(language_key):
    """Load a single atlas professional annotation file."""
    info = ATLAS_FILES.get(language_key)
    if info is None:
        return pd.DataFrame()
    fname, lang_full, lang_short = info
    fp = BASE_DIR / "human_professional" / fname
    if not fp.exists():
        return pd.DataFrame()
    df = pd.read_excel(fp)

    # Standardise columns
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "identifier":
            col_map[c] = "identifier"
        elif cl == "topic key":
            col_map[c] = "topic_key"
        elif cl == "edit required":
            col_map[c] = "Edit Required"
        elif cl == "clinical harm potential":
            col_map[c] = "Clinical Harm Potential"
        elif cl == "cmqm categories":
            col_map[c] = "CMQM Categories"
        elif cl == "english source":
            col_map[c] = "English Source"
        elif cl == "machine translation":
            col_map[c] = "Machine Translation"
    df.rename(columns=col_map, inplace=True)

    df["language"] = lang_full
    df["lang_short"] = lang_short
    df["edit_binary_human"] = df["Edit Required"].apply(
        lambda x: 1 if str(x).strip().lower() == "yes" else 0
    )
    df["harm_ord_human"] = df.get("Clinical Harm Potential", pd.Series(dtype=float)).map(
        lambda x: HARM_MAP_HUMAN.get(str(x).strip().lower(), np.nan)
    )
    for c in CMQM_CATS:
        df[f"cmqm_{c}_human"] = 0
    if "CMQM Categories" in df.columns:
        for idx, val in df["CMQM Categories"].items():
            if pd.notna(val):
                for c in CMQM_CATS:
                    if c in str(val).lower():
                        df.loc[idx, f"cmqm_{c}_human"] = 1
    return df


def load_all_atlas():
    """Load all atlas annotations into a dict keyed by language."""
    result = {}
    for lang_key in ATLAS_FILES:
        df = load_atlas_annotations(lang_key)
        if not df.empty:
            result[lang_key] = df
    return result


def atlas_summary(all_atlas):
    """Summary statistics for all atlas annotation sets."""
    rows = []
    for lang, df in all_atlas.items():
        edit_yes = df["edit_binary_human"].sum()
        edit_no = len(df) - edit_yes
        harm_any = df["harm_ord_human"].apply(lambda x: x > 0 if pd.notna(x) else False).sum()
        rows.append({
            "Language": lang,
            "Total Items": len(df),
            "Edit Yes": int(edit_yes),
            "Edit No": int(edit_no),
            "Edit Rate %": round(edit_yes / len(df) * 100, 1),
            "Harm Flagged": int(harm_any),
        })
    return pd.DataFrame(rows).sort_values("Language")


def cross_language_human_vs_llm(all_atlas, judge_df):
    """Compare human vs all LLM judges across all languages with atlas data."""
    results = []
    for lang, human_df in all_atlas.items():
        comp = compare_human_vs_judges(human_df, judge_df, lang)
        if comp.empty:
            continue
        comp["Language"] = lang
        results.append(comp)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


# ─── MQM Analysis Functions ───────────────────────────────────────────

MQM_SEVERITY_WEIGHTS = {"critical": -25, "major": -5, "minor": -1}


def load_all_mqm(results_dir=None):
    """Load all MQM JSONL results into a DataFrame."""
    if results_dir is None:
        results_dir = BASE_DIR / "llm_judge_results" / "mqm"
    else:
        results_dir = Path(results_dir)
    if not results_dir.exists():
        return pd.DataFrame()

    all_rows = []
    for fp in sorted(results_dir.glob("*.jsonl")):
        model_key = fp.stem
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                r["model_key"] = model_key
                r["model_short"] = MODELS_SHORT.get(model_key, model_key)
                r["lang_short"] = LANG_SHORT.get(r.get("language", ""), r.get("language", ""))
                # Flatten error categories
                errors = r.get("errors", [])
                cats = Counter()
                for err in errors:
                    if isinstance(err, dict):
                        cats[err.get("category", "other")] += 1
                r["error_categories"] = dict(cats)
                r["n_errors"] = len(errors)
                all_rows.append(r)
    return pd.DataFrame(all_rows)


def mqm_summary_by_model(mqm_df):
    """Aggregate MQM stats per model."""
    if mqm_df.empty:
        return pd.DataFrame()
    valid = mqm_df[mqm_df["_parse_error"] == False].copy()
    agg = valid.groupby("model_short").agg(
        items=("identifier", "count"),
        mean_score=("mqm_score", "mean"),
        median_score=("mqm_score", "median"),
        mean_errors=("n_errors", "mean"),
        n_critical=("n_critical", "sum"),
        n_major=("n_major", "sum"),
        n_minor=("n_minor", "sum"),
    ).round(2)
    return agg.sort_values("mean_score")


def mqm_summary_by_language(mqm_df):
    """Aggregate MQM stats per language."""
    if mqm_df.empty:
        return pd.DataFrame()
    valid = mqm_df[mqm_df["_parse_error"] == False].copy()
    agg = valid.groupby("lang_short").agg(
        items=("identifier", "count"),
        mean_score=("mqm_score", "mean"),
        median_score=("mqm_score", "median"),
        mean_errors=("n_errors", "mean"),
        n_critical=("n_critical", "sum"),
        n_major=("n_major", "sum"),
        n_minor=("n_minor", "sum"),
    ).round(2)
    return agg.sort_values("mean_score")


def mqm_score_heatmap(mqm_df):
    """Pivot: mean MQM score by model x language."""
    if mqm_df.empty:
        return pd.DataFrame()
    valid = mqm_df[mqm_df["_parse_error"] == False]
    return valid.groupby(["model_short", "lang_short"])["mqm_score"].mean().unstack().round(2)


def mqm_severity_distribution(mqm_df):
    """Severity distribution per model."""
    if mqm_df.empty:
        return pd.DataFrame()
    valid = mqm_df[mqm_df["_parse_error"] == False]
    return valid.groupby("model_short")[["n_critical", "n_major", "n_minor"]].sum()


def mqm_top_error_categories(mqm_df, top_n=15):
    """Most common MQM error categories across all models."""
    if mqm_df.empty:
        return pd.DataFrame()
    valid = mqm_df[mqm_df["_parse_error"] == False]
    all_cats = Counter()
    for cats in valid["error_categories"]:
        if isinstance(cats, dict):
            all_cats.update(cats)
    rows = [{"category": k, "count": v} for k, v in all_cats.most_common(top_n)]
    return pd.DataFrame(rows)


def mqm_top_categories_by_model(mqm_df, top_n=10):
    """Top error categories per model."""
    if mqm_df.empty:
        return pd.DataFrame()
    valid = mqm_df[mqm_df["_parse_error"] == False]
    results = {}
    for model in sorted(valid["model_short"].unique()):
        sub = valid[valid["model_short"] == model]
        cats = Counter()
        for c in sub["error_categories"]:
            if isinstance(c, dict):
                cats.update(c)
        results[model] = dict(cats.most_common(top_n))
    return pd.DataFrame(results).fillna(0).astype(int)


def cmqm_vs_mqm_comparison(judge_df, mqm_df):
    """Compare CMQM harm/categories with MQM scores for overlapping items."""
    if mqm_df.empty or judge_df.empty:
        return pd.DataFrame()
    valid_mqm = mqm_df[mqm_df["_parse_error"] == False][
        ["model_short", "language", "identifier", "topic_key", "mqm_score",
         "n_critical", "n_major", "n_minor", "n_errors"]
    ].copy()

    # Get CMQM data for matching items
    cmqm_cols = ["model_short", "language", "identifier", "topic_key",
                 "harm_potential", "harm_ord"] + [f"cmqm_{c}" for c in CMQM_CATS]
    cmqm_sub = judge_df[judge_df["edit_binary"] == 1][cmqm_cols].copy()

    merged = valid_mqm.merge(
        cmqm_sub,
        on=["model_short", "language", "identifier", "topic_key"],
        how="inner",
    )
    return merged


# ─── Harm Rescore Functions ───────────────────────────────────────────

def load_harm_rescore(results_dir=None):
    """Load harm re-scoring results."""
    if results_dir is None:
        results_dir = BASE_DIR / "llm_judge_results" / "harm_rescore"
    else:
        results_dir = Path(results_dir)
    if not results_dir.exists():
        return pd.DataFrame()

    all_rows = []
    for fp in sorted(results_dir.glob("*.jsonl")):
        model_key = fp.stem
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                r["model_key"] = model_key
                r["model_short"] = MODELS_SHORT.get(model_key, model_key)
                r["lang_short"] = LANG_SHORT.get(r.get("language", ""), r.get("language", ""))
                all_rows.append(r)
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)
