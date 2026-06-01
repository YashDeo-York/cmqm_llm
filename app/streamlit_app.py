"""
LLM Translation Quality Judge - Analysis Dashboard
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
    load_all_atlas, atlas_summary, cross_language_human_vs_llm,
    load_all_mqm, mqm_summary_by_model, mqm_summary_by_language,
    mqm_score_heatmap, mqm_severity_distribution,
    mqm_top_error_categories, mqm_top_categories_by_model,
    cmqm_vs_mqm_comparison, load_harm_rescore,
)

# --- Page Config ---
st.set_page_config(
    page_title="Translation Quality Judge Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).resolve().parent.parent / "llm_judge_results"

# --- Cached Data Loading ---

@st.cache_data(show_spinner="Loading judge data...")
def get_base_judge_data():
    return load_all_judges(DATA_DIR)

@st.cache_data(show_spinner="Loading atlas professional annotations...")
def get_all_atlas():
    return load_all_atlas()

@st.cache_data(show_spinner="Loading MQM results...")
def get_mqm_data():
    return load_all_mqm()

@st.cache_data(show_spinner="Loading harm re-scoring results...")
def get_harm_rescore():
    return load_harm_rescore()


# --- Helper plotting ---

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


# --- Sidebar ---
st.sidebar.title("🔬 Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "🔥 Cross-Judge Analysis",
    "🤝 Inter-Judge Agreement",
    "👤 Human vs LLM (All Languages)",
    "🔎 Human vs LLM (Deep Dive)",
    "📐 MQM Error Analysis",
    "⚖️ CMQM vs MQM Comparison",
    "📤 Upload New Judge",
])

# --- Load Data ---
df_base = get_base_judge_data()

# Handle uploaded judges stored in session
if "extra_judges" not in st.session_state:
    st.session_state.extra_judges = []

if st.session_state.extra_judges:
    df = pd.concat([df_base] + st.session_state.extra_judges, ignore_index=True)
else:
    df = df_base.copy()


# =======================================================================
# PAGE: Overview
# =======================================================================
if page == "📊 Overview":
    st.title("LLM Translation Quality Judge - Dashboard")
    st.markdown("""
    Analysis of **LLM-as-judge** evaluations of machine-translated clinical dialogue
    (Llama-3.3-70B translations), comparing multiple open-source LLM judges against
    each other and against **professional human annotators** across 8 languages.
    """)

    # Top-level metrics
    all_atlas = get_all_atlas()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total LLM Evaluations", f"{len(df):,}")
    c2.metric("LLM Judges", df["model_short"].nunique())
    c3.metric("Languages", df["lang_short"].nunique())
    c4.metric("Human Languages", len(all_atlas))
    c5.metric("Items / Language", f"{df.groupby('lang_short')['identifier'].nunique().median():.0f}")

    # Human annotations summary
    if all_atlas:
        st.subheader("Professional Human Annotations Summary")
        summary = atlas_summary(all_atlas)
        st.dataframe(summary.set_index("Language"), use_container_width=True)

        # Human vs LLM edit rate comparison
        st.subheader("Human vs LLM Mean Edit Rate by Language")
        human_rates = summary.set_index("Language")["Edit Rate %"]

        llm_lang_rates = df.groupby("language")["edit_binary"].mean() * 100
        compare_df = pd.DataFrame({
            "Human (Atlas)": human_rates,
            "LLM Mean": llm_lang_rates,
        }).dropna()

        if not compare_df.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(compare_df))
            ax.bar(x - 0.2, compare_df["Human (Atlas)"], 0.35, label="Human (Atlas)", color="#4CAF50")
            ax.bar(x + 0.2, compare_df["LLM Mean"], 0.35, label="LLM Mean", color="#2196F3")
            ax.set_xticks(x)
            ax.set_xticklabels(compare_df.index, rotation=30, ha="right")
            ax.set_ylabel("Edit Required Rate (%)")
            ax.set_title("Human Professional vs LLM Mean Edit Rate")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close()

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
    st.pyplot(fig); plt.close()

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
    st.pyplot(fig); plt.close()

    st.subheader("Parse Error / Repair Rates")
    pe = parse_error_rates(df)
    st.dataframe(pe.round(2).style.format("{:.2f}%"), use_container_width=True)


# =======================================================================
# PAGE: Cross-Judge Analysis
# =======================================================================
elif page == "🔥 Cross-Judge Analysis":
    st.title("Cross-Judge x Cross-Language Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Edit Rate Heatmap", "Harm Analysis", "CMQM Categories",
        "Question vs Answer", "Topic Breakdown",
    ])

    with tab1:
        st.subheader("Edit-Required Rate (%) by Judge x Language")
        er = edit_rate_by_judge_language(df) * 100
        er = er[sorted(er.columns)]
        fig = plot_heatmap(er, "Edit-Required Rate (%)", cbar_label="Edit %")
        st.pyplot(fig); plt.close()
        st.dataframe(er.round(1), use_container_width=True)

    with tab2:
        st.subheader("Harm Flagged Rate (%) by Judge x Language")
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

        st.markdown("**By Judge x Language (drill-down):**")
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


# =======================================================================
# PAGE: Inter-Judge Agreement
# =======================================================================
elif page == "🤝 Inter-Judge Agreement":
    st.title("Inter-Judge Agreement Analysis")

    tab1, tab2, tab3 = st.tabs([
        "Pairwise Cohen's k", "Per-Language Agreement", "Majority Vote",
    ])

    with tab1:
        st.subheader("Pairwise Cohen's k (edit_required)")
        kmat, pivot = pairwise_kappa_matrix(df)

        fig, ax = plt.subplots(figsize=(9, 7))
        mask = np.triu(np.ones_like(kmat, dtype=bool), k=1)
        sns.heatmap(kmat.astype(float), annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0, vmax=1, mask=mask, ax=ax, square=True)
        ax.set_title("Pairwise Cohen's k Between LLM Judges")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        mean_k = kmat.apply(lambda col: col[col.index != col.name].mean())
        st.subheader("Mean Pairwise k per Judge")
        mk_df = mean_k.to_frame("Mean k").sort_values("Mean k", ascending=False)
        st.dataframe(mk_df.round(3), use_container_width=True)

        st.metric("Overall Mean Inter-Judge k", f"{mean_k.mean():.3f}")

    with tab2:
        st.subheader("Agreement Statistics by Language")
        lk = per_language_kappa(df)
        st.dataframe(lk.round(3), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(lk.index, lk["mean_kappa"],
               yerr=lk["std"], capsize=4,
               color=sns.color_palette("Set2", len(lk)))
        ax.set_ylabel("Mean Pairwise k")
        ax.set_title("Mean Inter-Judge k by Language")
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

        st.subheader("Judge Deviation from Majority Vote")
        dev = {}
        for m in models:
            dev[m] = (mv[m] != mv["majority_edit"]).mean() * 100
        dev_df = pd.DataFrame.from_dict(dev, orient="index", columns=["Deviation %"])
        dev_df = dev_df.sort_values("Deviation %")
        st.dataframe(dev_df.round(1), use_container_width=True)


# =======================================================================
# PAGE: Human vs LLM (All Languages)
# =======================================================================
elif page == "👤 Human vs LLM (All Languages)":
    st.title("Human Professional vs LLM Judges - Cross-Language Overview")
    st.markdown("""
    Comparison of **8 professional human annotators** (Atlas) against all LLM judges.
    Each language has a dedicated professional annotator who labelled the same items.
    """)

    all_atlas = get_all_atlas()
    if not all_atlas:
        st.error("No atlas professional annotation files found in human_professional/")
        st.stop()

    cross = cross_language_human_vs_llm(all_atlas, df)
    if cross.empty:
        st.error("No matching items found between atlas annotations and judge data.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Kappa Overview", "Per-Language F1", "Edit Rate Comparison", "Harm Agreement",
    ])

    with tab1:
        st.subheader("Cohen's Kappa: Each Judge vs Human, by Language")
        kappa_pivot = cross.pivot_table(
            index="Judge", columns="Language", values="Cohen's Kappa"
        )
        fig = plot_heatmap(kappa_pivot, "Cohen's Kappa (Human vs LLM Judge)",
                           fmt=".3f", cmap="RdYlGn", vmin=0, vmax=0.6,
                           cbar_label="Kappa", figsize=(14, 7))
        st.pyplot(fig); plt.close()

        st.subheader("Mean Kappa per Judge (across languages)")
        mean_kappa = kappa_pivot.mean(axis=1).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(mean_kappa.index, mean_kappa.values,
                       color=sns.color_palette("RdYlGn", len(mean_kappa)))
        ax.set_xlabel("Mean Cohen's Kappa")
        ax.set_title("Mean Kappa per Judge (averaged across languages)")
        for bar, val in zip(bars, mean_kappa.values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.subheader("Mean Kappa per Language (across judges)")
        mean_kappa_lang = kappa_pivot.mean(axis=0).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(mean_kappa_lang.index, mean_kappa_lang.values,
               color=sns.color_palette("Set2", len(mean_kappa_lang)))
        ax.set_ylabel("Mean Cohen's Kappa")
        ax.set_title("Mean Human-LLM Agreement per Language")
        ax.axhline(y=0.2, color="red", linestyle="--", alpha=0.5, label="Fair (0.2)")
        ax.axhline(y=0.4, color="orange", linestyle="--", alpha=0.5, label="Moderate (0.4)")
        ax.legend()
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab2:
        st.subheader("F1 Score: Each Judge vs Human, by Language")
        f1_pivot = cross.pivot_table(
            index="Judge", columns="Language", values="F1"
        )
        fig = plot_heatmap(f1_pivot, "F1 Score (Human vs LLM Judge)",
                           fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1.0,
                           cbar_label="F1", figsize=(14, 7))
        st.pyplot(fig); plt.close()

        st.subheader("Precision vs Recall (per Judge, averaged)")
        prec_mean = cross.groupby("Judge")["Precision"].mean()
        rec_mean = cross.groupby("Judge")["Recall"].mean()
        fig, ax = plt.subplots(figsize=(8, 8))
        for judge in prec_mean.index:
            ax.scatter(prec_mean[judge], rec_mean[judge], s=100, zorder=5)
            ax.annotate(judge, (prec_mean[judge], rec_mean[judge]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.set_xlabel("Mean Precision")
        ax.set_ylabel("Mean Recall")
        ax.set_title("Precision-Recall Trade-off (Human as ground truth)")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        st.subheader("Edit Rate: Human vs Each Judge")
        er_pivot_human = cross.pivot_table(
            index="Judge", columns="Language", values="Human Edit %"
        )
        er_pivot_judge = cross.pivot_table(
            index="Judge", columns="Language", values="Judge Edit %"
        )

        # Bias: judge - human
        bias = er_pivot_judge - er_pivot_human
        fig = plot_heatmap(bias, "Edit Rate Bias (Judge - Human, pp)",
                           fmt=".1f", cmap="RdBu_r", vmin=-30, vmax=30,
                           cbar_label="Bias (pp)", figsize=(14, 7))
        st.pyplot(fig); plt.close()

        st.markdown("**Positive = judge flags more edits than human (over-sensitive), "
                    "Negative = judge flags fewer (under-sensitive)**")

        st.subheader("Mean Bias per Judge")
        mean_bias = bias.mean(axis=1).sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#4CAF50" if v < 0 else "#F44336" for v in mean_bias.values]
        ax.barh(mean_bias.index, mean_bias.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean Edit Rate Bias (pp)")
        ax.set_title("Judge Over/Under-sensitivity vs Human")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab4:
        st.subheader("Harm Agreement by Language")
        harm_rows = []
        for lang, human_df in all_atlas.items():
            harm_ag = harm_agreement_detail(human_df, df, lang)
            if not harm_ag.empty:
                harm_ag["Language"] = lang
                harm_rows.append(harm_ag)

        if harm_rows:
            all_harm = pd.concat(harm_rows, ignore_index=True)
            st.dataframe(all_harm, use_container_width=True)

            # Heatmap of harm agreement
            harm_pivot = all_harm.pivot_table(
                index="Judge", columns="Language", values="Agreement"
            )
            if not harm_pivot.empty:
                fig = plot_heatmap(harm_pivot, "Harm Agreement Rate (Human vs LLM)",
                                   fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1.0,
                                   cbar_label="Agreement", figsize=(14, 7))
                st.pyplot(fig); plt.close()
        else:
            st.info("Not enough harm annotations for comparison.")


# =======================================================================
# PAGE: Human vs LLM (Deep Dive)
# =======================================================================
elif page == "🔎 Human vs LLM (Deep Dive)":
    st.title("Human vs LLM Judge - Single Language Deep Dive")

    all_atlas = get_all_atlas()
    available = {}
    for lang_key, adf in all_atlas.items():
        available[f"Atlas: {lang_key}"] = (lang_key, adf)

    if not available:
        st.warning("No human annotations available.")
        st.stop()

    selected = st.selectbox("Select human annotation set", list(available.keys()))
    language, human_df = available[selected]

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
        ax.set_title(f"Human vs Judge Metrics - {language}")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

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

        cols = st.columns(min(4, max(1, len(models_available))))
        for i, model in enumerate(models_available):
            sub = detail[detail["judge_model"] == model]
            kappa_val = comp.loc[comp["Judge"] == model, "Cohen's Kappa"].values
            kappa_str = f" (k={kappa_val[0]:.3f})" if len(kappa_val) > 0 else ""
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


# =======================================================================
# PAGE: MQM Error Analysis
# =======================================================================
elif page == "📐 MQM Error Analysis":
    st.title("MQM Error Analysis (GEMBA-MQM)")
    st.markdown("""
    MQM (Multidimensional Quality Metrics) scoring of items flagged as requiring edits.
    Only items with `edit_required=yes` from each model's CMQM screening were scored
    using the GEMBA-MQM taxonomy (Chinese Mandarin + Urdu).
    """)

    mqm_df = get_mqm_data()
    if mqm_df.empty:
        st.warning("No MQM results found in llm_judge_results/mqm/")
        st.stop()

    valid_mqm = mqm_df[mqm_df.get("_parse_error", True) == False] if "_parse_error" in mqm_df.columns else mqm_df

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total MQM Annotations", f"{len(valid_mqm):,}")
    c2.metric("Models", valid_mqm["model_short"].nunique())
    c3.metric("Languages", valid_mqm["lang_short"].nunique())
    c4.metric("Mean MQM Score", f"{valid_mqm['mqm_score'].mean():.1f}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Summary", "Error Categories", "Score Heatmap", "Score Distributions",
    ])

    with tab1:
        st.subheader("MQM Summary by Model")
        model_summary = mqm_summary_by_model(mqm_df)
        if not model_summary.empty:
            st.dataframe(model_summary, use_container_width=True)

        st.subheader("MQM Summary by Language")
        lang_summary = mqm_summary_by_language(mqm_df)
        if not lang_summary.empty:
            st.dataframe(lang_summary, use_container_width=True)

        st.subheader("Severity Distribution by Model")
        sev = mqm_severity_distribution(mqm_df)
        if not sev.empty:
            fig = plot_grouped_bar(sev, "Error Severity Distribution by Model", "Count")
            st.pyplot(fig); plt.close()

    with tab2:
        st.subheader("Top MQM Error Categories (all models)")
        top_cats = mqm_top_error_categories(mqm_df, top_n=20)
        if not top_cats.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            top_sorted = top_cats.sort_values("count")
            ax.barh(top_sorted["category"], top_sorted["count"],
                    color=sns.color_palette("viridis", len(top_sorted)))
            ax.set_xlabel("Count")
            ax.set_title("Most Common MQM Error Categories")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        st.subheader("Top Error Categories by Model")
        cats_by_model = mqm_top_categories_by_model(mqm_df, top_n=8)
        if not cats_by_model.empty:
            fig = plot_heatmap(cats_by_model.T, "Error Category Counts by Model",
                               fmt=".0f", cmap="YlOrRd", figsize=(14, 8),
                               cbar_label="Count")
            st.pyplot(fig); plt.close()

    with tab3:
        st.subheader("Mean MQM Score by Model x Language")
        score_heat = mqm_score_heatmap(mqm_df)
        if not score_heat.empty:
            fig = plot_heatmap(score_heat, "Mean MQM Score (Model x Language)",
                               fmt=".1f", cmap="RdYlGn", figsize=(10, 7),
                               cbar_label="MQM Score")
            st.pyplot(fig); plt.close()

    with tab4:
        st.subheader("MQM Score Distributions by Model")
        fig, ax = plt.subplots(figsize=(12, 6))
        models = sorted(valid_mqm["model_short"].unique())
        data_to_plot = [valid_mqm[valid_mqm["model_short"] == m]["mqm_score"].values for m in models]
        bp = ax.boxplot(data_to_plot, labels=models, patch_artist=True)
        colors = sns.color_palette("Set2", len(models))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_ylabel("MQM Score")
        ax.set_title("MQM Score Distribution by Model")
        ax.axhline(y=-5, color="orange", linestyle="--", alpha=0.5, label="Major threshold")
        ax.axhline(y=-25, color="red", linestyle="--", alpha=0.5, label="Critical threshold")
        ax.legend()
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.subheader("MQM Score Distribution by Language")
        fig, ax = plt.subplots(figsize=(10, 6))
        langs = sorted(valid_mqm["lang_short"].unique())
        data_lang = [valid_mqm[valid_mqm["lang_short"] == l]["mqm_score"].values for l in langs]
        bp = ax.boxplot(data_lang, labels=langs, patch_artist=True)
        colors = sns.color_palette("Set3", len(langs))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_ylabel("MQM Score")
        ax.set_title("MQM Score Distribution by Language")
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# =======================================================================
# PAGE: CMQM vs MQM Comparison
# =======================================================================
elif page == "⚖️ CMQM vs MQM Comparison":
    st.title("CMQM vs MQM Taxonomy Comparison")
    st.markdown("""
    Comparing our custom **Clinical MQM (CMQM)** taxonomy with the standard
    **GEMBA-MQM** taxonomy on the same items. This helps answer whether the
    clinical-specific taxonomy captures different error patterns than the generic one.
    """)

    mqm_df = get_mqm_data()
    if mqm_df.empty:
        st.warning("No MQM results available for comparison.")
        st.stop()

    merged = cmqm_vs_mqm_comparison(df, mqm_df)
    if merged.empty:
        st.warning("No overlapping items between CMQM and MQM results.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Overlapping Items", f"{len(merged):,}")
    c2.metric("Models", merged["model_short"].nunique())
    c3.metric("Languages", merged["language"].nunique())

    tab1, tab2, tab3 = st.tabs([
        "Harm vs MQM Score", "CMQM Categories vs MQM", "Model Comparison",
    ])

    with tab1:
        st.subheader("CMQM Harm Level vs MQM Score")

        # Map harm to labels
        harm_labels = {0: "none", 1: "low", 2: "moderate", 3: "high"}
        merged["harm_label"] = merged["harm_ord"].map(harm_labels)

        fig, ax = plt.subplots(figsize=(10, 6))
        harm_order = ["none", "low", "moderate", "high"]
        harm_present = [h for h in harm_order if h in merged["harm_label"].values]
        data_groups = [merged[merged["harm_label"] == h]["mqm_score"].values for h in harm_present]
        if data_groups:
            bp = ax.boxplot(data_groups, labels=harm_present, patch_artist=True)
            colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]
            for i, patch in enumerate(bp["boxes"]):
                if i < len(colors):
                    patch.set_facecolor(colors[i])
            ax.set_xlabel("CMQM Harm Level")
            ax.set_ylabel("MQM Score")
            ax.set_title("MQM Score Distribution by CMQM Harm Level")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Correlation
        if "harm_ord" in merged.columns and "mqm_score" in merged.columns:
            valid = merged.dropna(subset=["harm_ord", "mqm_score"])
            if len(valid) > 10:
                corr, pval = stats.spearmanr(valid["harm_ord"], valid["mqm_score"])
                st.metric("Spearman Correlation (Harm vs MQM)", f"{corr:.3f} (p={pval:.4f})")

    with tab2:
        st.subheader("MQM Score by CMQM Category Presence")
        for cat in CMQM_CATS:
            col = f"cmqm_{cat}"
            if col in merged.columns:
                with_cat = merged[merged[col] == 1]["mqm_score"]
                without_cat = merged[merged[col] == 0]["mqm_score"]
                if len(with_cat) > 0 and len(without_cat) > 0:
                    st.markdown(f"**{cat}**: present={len(with_cat)} (mean MQM={with_cat.mean():.1f}), "
                                f"absent={len(without_cat)} (mean MQM={without_cat.mean():.1f})")

        st.subheader("MQM Score Distribution: Clinical Accuracy Present vs Absent")
        if "cmqm_clinical_accuracy" in merged.columns:
            fig, ax = plt.subplots(figsize=(10, 5))
            for label, val in [("Present", 1), ("Absent", 0)]:
                sub = merged[merged["cmqm_clinical_accuracy"] == val]["mqm_score"]
                if len(sub) > 0:
                    ax.hist(sub, bins=20, alpha=0.6, label=f"Clinical Accuracy {label} (n={len(sub)})")
            ax.set_xlabel("MQM Score")
            ax.set_ylabel("Count")
            ax.set_title("MQM Score: Clinical Accuracy Present vs Absent")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    with tab3:
        st.subheader("Per-Model: Mean MQM Score vs CMQM Harm Rate")
        model_agg = merged.groupby("model_short").agg(
            mean_mqm=("mqm_score", "mean"),
            harm_rate=("harm_ord", lambda x: (x > 0).mean() * 100),
            n_items=("identifier", "count"),
        ).round(2)
        st.dataframe(model_agg, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        for model in model_agg.index:
            ax.scatter(model_agg.loc[model, "harm_rate"],
                       model_agg.loc[model, "mean_mqm"], s=100, zorder=5)
            ax.annotate(model, (model_agg.loc[model, "harm_rate"],
                                model_agg.loc[model, "mean_mqm"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax.set_xlabel("CMQM Harm Rate (%)")
        ax.set_ylabel("Mean MQM Score")
        ax.set_title("CMQM Harm Rate vs Mean MQM Score per Model")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.subheader("Error Count Comparison")
        err_agg = merged.groupby("model_short").agg(
            n_items=("identifier", "count"),
            mean_mqm_errors=("n_errors", "mean"),
            mean_cmqm_cats=("cmqm_clinical_accuracy", lambda x: sum(
                merged.loc[x.index, [f"cmqm_{c}" for c in CMQM_CATS]].sum(axis=1)
            ) / len(x)),
        ).round(2)
        err_agg.columns = ["Items", "Mean MQM Errors/Item", "Mean CMQM Cats/Item"]
        st.dataframe(err_agg, use_container_width=True)


# =======================================================================
# PAGE: Upload New Judge
# =======================================================================
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
