"""
LLM Translation Quality Judge – Analysis Dashboard
====================================================
Run with:  streamlit run app/streamlit_app.py
"""

import sys, os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Allow imports when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.analysis_engine import (
    load_all_judges, load_judge_from_upload, load_human_annotations,
    edit_rate_by_judge_language, overall_edit_rate_by_model,
    overall_edit_rate_by_language, harm_rate_by_judge_language,
    harm_level_distribution, cmqm_by_model, cmqm_by_language,
    cmqm_by_model_language, edit_rate_by_row_type, edit_rate_by_topic,
    parse_error_rates, pairwise_kappa_matrix, per_language_kappa,
    unanimous_agreement, majority_vote_analysis,
    compare_human_vs_judges, per_item_agreement_detail,
    cmqm_agreement_detail, harm_agreement_detail,
    disagreement_examples, CMQM_CATS, LANG_SHORT,
)

# ─── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Translation Quality Judge Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parent.parent / "llm_judge_results"

# ─── Cached Data Loading ────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading judge data…")
def get_base_judge_data():
    return load_all_judges(DATA_DIR)

@st.cache_data(show_spinner="Loading human annotations…")
def get_builtin_human(name):
    base = Path(__file__).resolve().parent.parent
    if name == "emily":
        df = load_human_annotations(base / "emily.csv", "emily.csv")
        df["language"] = "Chinese Mandarin"
        df["lang_short"] = "ZH"
        return df
    elif name == "portuguese":
        df = load_human_annotations(
            base / "Portuguese__Brazilian__annotations.xlsx",
            "Portuguese__Brazilian__annotations.xlsx",
        )
        df["language"] = "Portuguese (Brazilian)"
        df["lang_short"] = "PT-BR"
        return df
    return pd.DataFrame()


# ─── Helper plotting ────────────────────────────────────────────────────

def plot_heatmap(data, title, fmt=".1f", cmap="YlOrRd", vmin=None, vmax=None,
                 figsize=(14, 6), cbar_label=""):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                vmin=vmin, vmax=vmax, cbar_kws={"label": cbar_label})
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_grouped_bar(data, title, ylabel="", figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    data.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, title=""):
    from sklearn.metrics import confusion_matrix as cm_func
    cm = cm_func(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Edit", "Edit"], yticklabels=["No Edit", "Edit"])
    ax.set_xlabel("Judge")
    ax.set_ylabel("Human")
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    return fig


# ─── Sidebar ────────────────────────────────────────────────────────────
st.sidebar.title("🔬 Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "🔥 Cross-Judge Analysis",
    "🤝 Inter-Judge Agreement",
    "👤 Human vs Judge",
    "📤 Upload New Judge",
    "📋 Upload Human Annotations",
])

# ─── Load Data ──────────────────────────────────────────────────────────
df_base = get_base_judge_data()

# Handle uploaded judges stored in session
if "extra_judges" not in st.session_state:
    st.session_state.extra_judges = []

if st.session_state.extra_judges:
    df = pd.concat([df_base] + st.session_state.extra_judges, ignore_index=True)
else:
    df = df_base.copy()

# Handle uploaded human annotations
if "uploaded_humans" not in st.session_state:
    st.session_state.uploaded_humans = {}


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("LLM Translation Quality Judge – Dashboard")
    st.markdown("""
    This dashboard analyses **LLM-as-judge** evaluations of machine-translated
    clinical dialogue, comparing multiple LLM judges against each other and
    against human annotations.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Evaluations", f"{len(df):,}")
    c2.metric("LLM Judges", df["model_short"].nunique())
    c3.metric("Languages", df["lang_short"].nunique())
    c4.metric("Items / Language", f"{df.groupby('lang_short')['identifier'].nunique().median():.0f}")

    st.subheader("Overall Edit-Required Rate by Judge")
    model_rates = overall_edit_rate_by_model(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("RdYlGn_r", len(model_rates))
    bars = ax.barh(model_rates.index, model_rates["edit_rate"] * 100, color=colors)
    ax.set_xlabel("Edit Required (%)")
    ax.set_title("Overall Edit-Required Rate by Judge")
    for bar, val in zip(bars, model_rates["edit_rate"] * 100):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Overall Edit-Required Rate by Language")
    lang_rates = overall_edit_rate_by_language(df)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("YlOrRd", len(lang_rates))
    bars = ax.barh(lang_rates.index, lang_rates["edit_rate"] * 100, color=colors)
    ax.set_xlabel("Edit Required (%)")
    ax.set_title("Overall Edit-Required Rate by Target Language")
    for bar, val in zip(bars, lang_rates["edit_rate"] * 100):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Parse Error / Repair Rates")
    pe = parse_error_rates(df)
    st.dataframe(pe.round(2).style.format("{:.2f}%"), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Cross-Judge Analysis
# ═════════════════════════════════════════════════════════════════════════
elif page == "🔥 Cross-Judge Analysis":
    st.title("Cross-Judge × Cross-Language Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Edit Rate Heatmap", "Harm Analysis", "CMQM Categories",
        "Question vs Answer", "Topic Breakdown",
    ])

    with tab1:
        st.subheader("Edit-Required Rate (%) by Judge × Language")
        er = edit_rate_by_judge_language(df) * 100
        er = er[sorted(er.columns)]
        fig = plot_heatmap(er, "Edit-Required Rate (%)", cbar_label="Edit %")
        st.pyplot(fig); plt.close()
        st.dataframe(er.round(1), use_container_width=True)

    with tab2:
        st.subheader("Harm Flagged Rate (%) by Judge × Language")
        hr = harm_rate_by_judge_language(df) * 100
        hr = hr[sorted(hr.columns)]
        fig = plot_heatmap(hr, "Harm Flagged Rate (%)", cbar_label="Harm %")
        st.pyplot(fig); plt.close()

        st.subheader("Harm Level Distribution by Judge")
        hd = harm_level_distribution(df)
        fig = plot_grouped_bar(hd, "Harm Level Counts by Judge", "Count")
        st.pyplot(fig); plt.close()

    with tab3:
        st.subheader("CMQM Error Categories Among Flagged Items")

        st.markdown("**By Judge:**")
        cm_model = cmqm_by_model(df) * 100
        fig = plot_grouped_bar(cm_model, "CMQM Prevalence by Judge (%)", "% of flagged")
        st.pyplot(fig); plt.close()
        st.dataframe(cm_model.round(1), use_container_width=True)

        st.markdown("**By Language:**")
        cm_lang = cmqm_by_language(df) * 100
        fig = plot_grouped_bar(cm_lang, "CMQM Prevalence by Language (%)", "% of flagged")
        st.pyplot(fig); plt.close()
        st.dataframe(cm_lang.round(1), use_container_width=True)

        st.markdown("**By Judge × Language (drill-down):**")
        cm_both = cmqm_by_model_language(df) * 100
        if not cm_both.empty:
            sel_cat = st.selectbox("Select CMQM category", CMQM_CATS)
            pivot_cat = cm_both[sel_cat].unstack()
            fig = plot_heatmap(pivot_cat, f"{sel_cat} prevalence (%)",
                               cbar_label="%", figsize=(14, 6))
            st.pyplot(fig); plt.close()

    with tab4:
        st.subheader("Edit Rate by Turn Type (Question vs Answer)")
        rt = edit_rate_by_row_type(df) * 100
        fig = plot_grouped_bar(rt, "Edit Rate by Row Type (%)", "Edit %")
        st.pyplot(fig); plt.close()
        st.dataframe(rt.round(1), use_container_width=True)

    with tab5:
        st.subheader("Edit Rate by Clinical Topic")
        tp = edit_rate_by_topic(df)
        tp["edit_rate"] = tp["edit_rate"] * 100
        st.dataframe(tp.round(1), use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        tp_sorted = tp.sort_values("edit_rate")
        ax.barh(tp_sorted.index, tp_sorted["edit_rate"],
                color=sns.color_palette("YlOrRd", len(tp_sorted)))
        ax.set_xlabel("Edit Required (%)")
        ax.set_title("Edit Rate by Clinical Topic (all judges, all languages)")
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Inter-Judge Agreement
# ═════════════════════════════════════════════════════════════════════════
elif page == "🤝 Inter-Judge Agreement":
    st.title("Inter-Judge Agreement Analysis")

    tab1, tab2, tab3 = st.tabs([
        "Pairwise Cohen's κ", "Per-Language Agreement", "Majority Vote",
    ])

    with tab1:
        st.subheader("Pairwise Cohen's κ (edit_required)")
        kmat, pivot = pairwise_kappa_matrix(df)

        fig, ax = plt.subplots(figsize=(9, 7))
        mask = np.triu(np.ones_like(kmat, dtype=bool), k=1)
        sns.heatmap(kmat.astype(float), annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0, vmax=1, mask=mask, ax=ax, square=True)
        ax.set_title("Pairwise Cohen's κ Between LLM Judges")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        mean_k = kmat.apply(lambda col: col[col.index != col.name].mean())
        st.subheader("Mean Pairwise κ per Judge")
        mk_df = mean_k.to_frame("Mean κ").sort_values("Mean κ", ascending=False)
        st.dataframe(mk_df.round(3), use_container_width=True)

        st.metric("Overall Mean Inter-Judge κ", f"{mean_k.mean():.3f}")

    with tab2:
        st.subheader("Agreement Statistics by Language")
        lk = per_language_kappa(df)
        st.dataframe(lk.round(3), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(lk.index, lk["mean_kappa"],
               yerr=lk["std"], capsize=4,
               color=sns.color_palette("Set2", len(lk)))
        ax.set_ylabel("Mean Pairwise κ")
        ax.set_title("Mean Inter-Judge κ by Language")
        ax.axhline(y=0.2, color="red", linestyle="--", alpha=0.5, label="Fair (0.2)")
        ax.axhline(y=0.4, color="orange", linestyle="--", alpha=0.5, label="Moderate (0.4)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.subheader("Unanimous Agreement by Language")
        ua = unanimous_agreement(df)
        st.dataframe(ua.round(1), use_container_width=True)

    with tab3:
        st.subheader("Majority Vote Analysis")
        mv = majority_vote_analysis(df)
        models = sorted(df["model_short"].unique())

        majority_rate = mv.groupby(level=0)["majority_edit"].mean() * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        majority_rate.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(majority_rate)))
        ax.set_ylabel("Majority-Vote Edit Rate (%)")
        ax.set_title("Majority-Vote Edit Rate by Language")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.subheader("Agreement Strength Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(mv["agreement_ratio"], bins=20, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Agreement Ratio (max(yes,no)/total judges)")
        ax.set_ylabel("Number of Items")
        ax.set_title("Distribution of Per-Item Judge Agreement Strength")
        ax.axvline(x=0.5, color="red", linestyle="--", label="Bare majority")
        ax.axvline(x=1.0, color="green", linestyle="--", label="Unanimous")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Per-judge deviation from majority
        st.subheader("Judge Deviation from Majority Vote")
        dev = {}
        for m in models:
            dev[m] = (mv[m] != mv["majority_edit"]).mean() * 100
        dev_df = pd.DataFrame.from_dict(dev, orient="index", columns=["Deviation %"])
        dev_df = dev_df.sort_values("Deviation %")
        st.dataframe(dev_df.round(1), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Human vs Judge
# ═════════════════════════════════════════════════════════════════════════
elif page == "👤 Human vs Judge":
    st.title("Human Annotation vs LLM Judge Comparison")

    # Determine available human datasets
    available = {}
    available["Emily (Chinese Mandarin)"] = ("emily", "Chinese Mandarin")
    available["Portuguese-BR Annotator"] = ("portuguese", "Portuguese (Brazilian)")
    for name, udf in st.session_state.uploaded_humans.items():
        available[f"Uploaded: {name}"] = (name, udf["language"])

    if not available:
        st.warning("No human annotations available. Upload some in the sidebar.")
        st.stop()

    selected = st.selectbox("Select human annotation set", list(available.keys()))
    key, language = available[selected]

    if key in ("emily", "portuguese"):
        human_df = get_builtin_human(key)
    else:
        human_df = st.session_state.uploaded_humans[key]["df"]

    st.info(f"**Language:** {language} | **Items:** {len(human_df)} | "
            f"**Human Edit Rate:** {human_df['edit_binary_human'].mean()*100:.1f}%")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Main Metrics", "Confusion Matrices", "CMQM Agreement",
        "Harm Agreement", "Disagreement Examples",
    ])

    with tab1:
        st.subheader("Edit-Required: Human vs Each Judge")
        comp = compare_human_vs_judges(human_df, df, language)
        if comp.empty:
            st.error("No matching items found between human annotations and judge data.")
            st.stop()

        # Highlight best kappa
        st.dataframe(
            comp.style.highlight_max(subset=["Cohen's Kappa", "F1"], color="#90EE90")
                .highlight_min(subset=["Cohen's Kappa", "F1"], color="#FFB6C1"),
            use_container_width=True,
        )

        st.subheader("Visual Comparison")
        metrics = ["Cohen's Kappa", "Accuracy", "Precision", "Recall", "Specificity", "F1"]
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(comp))
        w = 0.13
        for i, m in enumerate(metrics):
            ax.bar(x + i * w, comp[m], w, label=m)
        ax.set_xticks(x + w * 2.5)
        ax.set_xticklabels(comp["Judge"], rotation=30, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"Human vs Judge Metrics — {language}")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Edit rate comparison
        st.subheader("Edit Rate: Human vs Judge")
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(comp))
        ax.bar(x - 0.15, comp["Human Edit %"], 0.3, label="Human", color="#4CAF50")
        ax.bar(x + 0.15, comp["Judge Edit %"], 0.3, label="Judge", color="#2196F3")
        ax.set_xticks(x)
        ax.set_xticklabels(comp["Judge"], rotation=30, ha="right")
        ax.set_ylabel("Edit Rate (%)")
        ax.set_title("Edit Rate Comparison")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # McNemar's
        st.subheader("McNemar's Test (systematic bias)")
        mcn = comp[["Judge", "FP", "FN", "McNemar Chi2", "McNemar p"]].copy()
        mcn["Significant"] = mcn["McNemar p"].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        )
        st.dataframe(mcn, use_container_width=True)

    with tab2:
        st.subheader("Confusion Matrices")
        detail = per_item_agreement_detail(human_df, df, language)
        models_available = sorted(detail["judge_model"].unique()) if not detail.empty else []

        cols = st.columns(min(4, len(models_available)))
        for i, model in enumerate(models_available):
            sub = detail[detail["judge_model"] == model]
            kappa_val = comp.loc[comp["Judge"] == model, "Cohen's Kappa"].values
            kappa_str = f" (κ={kappa_val[0]:.3f})" if len(kappa_val) > 0 else ""
            fig = plot_confusion_matrix(
                sub["edit_binary_human"], sub["edit_binary"],
                f"{model}{kappa_str}"
            )
            with cols[i % len(cols)]:
                st.pyplot(fig); plt.close()

    with tab3:
        st.subheader("CMQM Category Agreement (among items both flagged)")
        cmqm_ag = cmqm_agreement_detail(human_df, df, language)
        if cmqm_ag.empty:
            st.info("Not enough overlapping flagged items for CMQM analysis.")
        else:
            st.dataframe(cmqm_ag, use_container_width=True)

            # Visualise
            agree_cols = [c for c in cmqm_ag.columns if c.endswith("_agree")]
            if agree_cols:
                plot_data = cmqm_ag.set_index("Judge")[agree_cols]
                plot_data.columns = [c.replace("_agree", "") for c in agree_cols]
                fig = plot_grouped_bar(plot_data * 100, "CMQM Agreement Rate (%)", "Agreement %")
                st.pyplot(fig); plt.close()

    with tab4:
        st.subheader("Harm Potential Agreement")
        harm_ag = harm_agreement_detail(human_df, df, language)
        if harm_ag.empty:
            st.info("Not enough harm annotations for comparison.")
        else:
            st.dataframe(harm_ag, use_container_width=True)

    with tab5:
        st.subheader("Disagreement Examples")
        detail = per_item_agreement_detail(human_df, df, language)
        models_available = sorted(detail["judge_model"].unique()) if not detail.empty else []
        sel_model = st.selectbox("Select judge", models_available)
        if sel_model:
            examples = disagreement_examples(human_df, df, language, sel_model, n=30)
            if examples.empty:
                st.success("No disagreements found!")
            else:
                st.write(f"Showing up to 30 disagreements with **{sel_model}**:")
                st.dataframe(examples, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Upload New Judge
# ═════════════════════════════════════════════════════════════════════════
elif page == "📤 Upload New Judge":
    st.title("Upload New LLM Judge Results")
    st.markdown("""
    Upload a `.jsonl` file with the same format as the existing judge results.
    Each line should be a JSON object with fields: `model`, `language`,
    `identifier`, `topic_key`, `row_type`, `original_text`,
    `machine_translation`, `edit_required`, `harm_potential`,
    `cmqm_categories`, `brief_rationale`.
    """)

    model_name = st.text_input("Judge display name", placeholder="e.g. GPT-4o")
    uploaded = st.file_uploader("Upload JSONL file", type=["jsonl"])

    if uploaded and model_name:
        if st.button("Add Judge & Recalculate", type="primary"):
            try:
                new_df = load_judge_from_upload(uploaded, model_name)
                st.session_state.extra_judges.append(new_df)
                st.success(f"Added **{model_name}** with {len(new_df):,} evaluations. "
                           f"Switch to other pages to see updated analysis.")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")

    if st.session_state.extra_judges:
        st.subheader("Uploaded Judges")
        for i, ej in enumerate(st.session_state.extra_judges):
            name = ej["model_short"].iloc[0] if len(ej) > 0 else f"Judge {i}"
            st.write(f"- **{name}**: {len(ej):,} evaluations")
        if st.button("Clear all uploaded judges"):
            st.session_state.extra_judges = []
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════
# PAGE: Upload Human Annotations
# ═════════════════════════════════════════════════════════════════════════
elif page == "📋 Upload Human Annotations":
    st.title("Upload Human Annotations")
    st.markdown("""
    Upload a CSV or XLSX file with human annotations. Required columns:
    - **Identifier** – matches judge data identifiers
    - **Topic Key** – matches judge data topic keys
    - **Edit Required** – `Yes` / `No` / blank (blank = No)

    Optional columns: `Clinical Harm Potential`, `CMQM Categories`,
    `English Source`, `Machine Translation`
    """)

    ann_name = st.text_input("Annotation set name", placeholder="e.g. Dr. Smith - Arabic")
    lang_options = list(LANG_SHORT.keys())
    ann_lang = st.selectbox("Language", lang_options)
    uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

    if uploaded and ann_name and ann_lang:
        if st.button("Load Annotations & Analyse", type="primary"):
            try:
                hdf = load_human_annotations(uploaded, uploaded.name)
                hdf["language"] = ann_lang
                hdf["lang_short"] = LANG_SHORT.get(ann_lang, ann_lang)
                st.session_state.uploaded_humans[ann_name] = {
                    "df": hdf, "language": ann_lang,
                }
                st.success(
                    f"Loaded **{ann_name}** ({ann_lang}): {len(hdf)} items, "
                    f"{hdf['edit_binary_human'].sum()} edits. "
                    f"Go to **👤 Human vs Judge** to see the analysis."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # Show existing
    st.subheader("Available Human Annotation Sets")
    st.write("**Built-in:**")
    st.write("- Emily (Chinese Mandarin): 774 items")
    st.write("- Portuguese-BR Annotator: 774 items")

    if st.session_state.uploaded_humans:
        st.write("**Uploaded:**")
        for name, info in st.session_state.uploaded_humans.items():
            st.write(f"- {name} ({info['language']}): {len(info['df'])} items")
        if st.button("Clear uploaded annotations"):
            st.session_state.uploaded_humans = {}
            st.rerun()
