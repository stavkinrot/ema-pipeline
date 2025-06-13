from io import BytesIO
import sys
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import base64
from matplotlib.patches import Patch
from scipy.stats import ttest_ind
import itertools
from matplotlib.colors import to_rgba
from statannotations.Annotator import Annotator

from colors_config import PALETTE, AGGRESSION_COLORS
from factor_config import FACTOR_QUESTIONS
from included_questions import INCLUDED_QUESTIONS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Shared Config ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "output", "merged_surveys.csv")
ARI_PATH = os.path.join(PROJECT_ROOT, "data", "ARI.xlsx")
AVG_PATH = os.path.join(PROJECT_ROOT, "output", "child_parent_averages.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Export Helpers ===
def add_pvalue_manual(ax, x1, x2, y, h, pval, color="gray", fontsize=8, stars=True):
    # Do not display non-significant
    if pval >= 0.05:
        return
    # Stars format
    if stars:
        if pval < 0.001:
            text = "***"
        elif pval < 0.01:
            text = "**"
        else:
            text = "*"
    else:
        text = f"p = {pval:.3f}"

    # Line
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c=color)
    # Text
    ax.text((x1 + x2) / 2, y + h + 0.003, text, ha='center', va='bottom', fontsize=fontsize, color=color)

def apply_modern_mpl_style(ax, *, xlabel=None, ylabel=None, title=None, xtick_labels=None):
    darkgray = "#4B4B4B"
    from matplotlib.ticker import MultipleLocator

    # Axis labels and title
    if title:
        ax.set_title(title, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontdict={'fontsize': 11, 'color': darkgray})
    if ylabel:
        ax.set_ylabel(ylabel, fontdict={'fontsize': 11, 'color': darkgray})

    # Tick styling
    ax.tick_params(axis='x', labelsize=9, colors=darkgray)
    ax.tick_params(axis='y', labelsize=9, colors=darkgray)

    # Optional: Set y-axis spacing
    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    # Optional: Set x-axis labels
    if xtick_labels is not None:
        ax.set_xticks(range(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    # Grid and spines
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', linestyle='-', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # Legend (if any)
    legend = ax.get_legend()
    if legend:
        legend.facecolor = 'white',
        legend.edgecolor = 'lightgray',
        legend.alpha = 0.2,
        legend.fontsize = 5

def get_image_download_link(fig, filename="plot.png"):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Download as PNG (Matplotlib)</a>'
    return href

def export_matplotlib_png(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf

def export_plotly_png(fig):
    from plotly.io import write_image
    buf = BytesIO()
    write_image(fig, buf, format="png", scale=3)
    buf.seek(0)
    return buf

def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'


def assign_ari_groups(ari_df, low_th, high_th):
    def assign(score):
        if pd.isna(score):
            return None
        if np.isclose(low_th, high_th):
            return "Low" if score < low_th else "High"
        if score < low_th:
            return "Low"
        elif score < high_th:
            return "Medium"
        else:
            return "High"
    df = ari_df.copy()
    df = df.rename(columns={"participant number": "participant_code", "score": "score"})
    df["participant_code"] = df["participant_code"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["group"] = df["score"].apply(assign)
    return df

# === Plotting Functions ===

def plot_aggression(df, ari_df):
    AGGRESSION_COLS = [
        "C_Agr_other", "C_Agr_slam", "C_Agr_throw_smt", "C_Agr_throw_twd", "C_Agr_yelled"
    ]
    pie2_labels = {
        "C_Agr_other": "Other",
        "C_Agr_slam": "Slammed something",
        "C_Agr_throw_smt": "Threw object",
        "C_Agr_throw_twd": "Threw toward someone",
        "C_Agr_yelled": "Yelled"
    }

    def prepare_ari_df(ari_df):
        ari_df = ari_df.rename(columns={"participant number": "participant_code", "score": "score"})
        ari_df["participant_code"] = ari_df["participant_code"].astype(str).str.strip()
        ari_df["group"] = pd.to_numeric(ari_df["score"], errors="coerce").apply(
            lambda x: "Irritable" if pd.notna(x) and x >= 3 else "Non-Irritable"
        )
        return ari_df

    def prepare_df(df):
        df["participant_code"] = df["participant_code_parent"].astype(str).str.strip()
        df["C_Agr_other"] = df["C_Agr_other"].apply(lambda x: 1 if isinstance(x, str) and x.strip() != "" else 0)
        for col in AGGRESSION_COLS[1:]:
            df[col] = df[col].replace({"Yes": 1, "No": 0})
        return df

    ari_df = prepare_ari_df(ari_df)
    df = prepare_df(df)
    df = df.merge(ari_df[["participant_code", "group"]], on="participant_code", how="inner")

    avg_df = pd.read_csv(AVG_PATH)
    avg_df["participant_code"] = avg_df["participant_code_parent"].astype(str).str.strip()
    avg_df = avg_df.merge(ari_df[["participant_code", "group"]], on="participant_code", how="inner")
    avg_df["C_Agr_none"] = pd.to_numeric(avg_df["C_Agr_none"], errors="coerce")
    avg_df["aggression_rate"] = 1 - avg_df["C_Agr_none"]
    aggr_share = avg_df.groupby("group")["aggression_rate"].mean() * 100

    participant_aggr = (
        df.groupby("participant_code")[AGGRESSION_COLS]
        .mean()
        .multiply(100)
        .reset_index()
        .merge(df[["participant_code", "group"]].drop_duplicates(), on="participant_code")
    )
    participant_aggr = participant_aggr[participant_aggr[AGGRESSION_COLS].sum(axis=1) > 0]
    grouped_means = participant_aggr.groupby("group")[AGGRESSION_COLS].mean()

    figs = []
    for group in ["Irritable", "Non-Irritable"]:
        if group not in grouped_means.index:
            continue

        values = grouped_means.loc[group].values
        labels = [pie2_labels[c] for c in AGGRESSION_COLS]
        colors = [AGGRESSION_COLORS[c] for c in AGGRESSION_COLS]
        n = (participant_aggr["group"] == group).sum()
        n_total = (avg_df["group"] == group).sum()
        aggr_pct = aggr_share.get(group, 0.0)

        # Plotly figure
        fig_plotly = go.Figure(
            data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo="label+percent"
            )]
        )
        fig_plotly.update_layout(
            title=(
            f"Aggression Breakdown – {group} Group<br>"
            f"{n} participants (of {n_total}) reported aggression, Average Aggression Rate: {aggr_pct:.1f}%"
        ),
            showlegend=False
        )

        # Matplotlib figure
        fig_mat, ax = plt.subplots(figsize=(6, 6))
        ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        ax.set_title(
            f"Aggression Breakdown – {group} Group\n"
            f"{n} participants (of {n_total}) reported aggression, Average Aggression Rate: {aggr_pct:.1f}%",
            pad=10
        )

        figs.append((group, fig_plotly, fig_mat))

    return figs


def plot_means_by_irritability(df, ari_df, selected_questions, color_map):
    avg_df = pd.read_csv("output/child_parent_averages.csv")
    avg_df["participant_code"] = avg_df["participant_code_parent"].astype(str).str.strip()
    merged = avg_df.merge(ari_df[["participant_code", "group"]], on="participant_code", how="inner")

    figs = []

    for role, label in zip(["child", "parent"], ["Children", "Parents"]):
        role_prefix = "C_" if role == "child" else "P_"
        role_cols = [col for col in selected_questions if col.startswith(role_prefix)]
        if not role_cols:
            continue

        df_role = merged[["group"] + role_cols].copy()
        melted = df_role.melt(id_vars="group", var_name="Question", value_name="Mean Score")

        # Compute group means and N
        grouped = melted.groupby(["Question", "group"])["Mean Score"].mean().reset_index()
        df_role["participant_code"] = merged["participant_code"]
        n_per_group = df_role.groupby("group")["participant_code"].nunique().to_dict()

        print("Correct group counts:", n_per_group)
        print("Sum of participants:", sum(n_per_group.values()))  # Should equal 81

        # Plotly interactive chart
        grouped["group_with_n"] = grouped["group"].apply(
            lambda g: f"{g} (N={n_per_group.get(g, 0)})"
        )

        grouped["group"] = pd.Categorical(grouped["group"], categories=["Low", "Medium", "High"], ordered=True)
        grouped = grouped.sort_values(["Question", "group"])
        fig_plotly = px.bar(
            grouped,
            x="Question",
            y="Mean Score",
            color="group_with_n",
            barmode="group",
            title=f"{label} Question Means by Irritability",
            color_discrete_map = {
                gw: color_map.get(gw.split(" ")[0], "#cccccc")  # extract 'Low', 'Medium', etc.
                for gw in grouped["group_with_n"].unique()
            }
        )
        fig_plotly.update_layout(
            xaxis_tickangle=-45,
            margin=dict(t=60, b=100),
            width=900,
            height=300
        )

        # Plotly export version
        fig_export = go.Figure(fig_plotly.to_dict())
        fig_export.update_layout(
            width=1100,
            height=600,
            margin=dict(l=100, r=100, t=100, b=150)
        )

        # Matplotlib version
        grouped_mat = melted.groupby(["Question", "group"])["Mean Score"].mean().unstack()
        x = np.arange(len(grouped_mat.index))
        width = 0.22  # Narrower bars to fit nicely

        fig_mat, ax = plt.subplots(figsize=(max(10, len(grouped_mat.index) * 0.6), 5))

        # Ensure consistent group order
        grouped_mat = grouped_mat.reindex(columns=[g for g in ["Low", "Medium", "High"] if g in grouped_mat.columns])
        n_groups = len(grouped_mat.columns)

        # Compute fixed offsets per group (centered)
        bar_offsets = [-width * n_groups / 2 + width * (i + 0.5) for i in range(n_groups)]

        # Plot bars for each group
        for i, g in enumerate(grouped_mat.columns):
            ax.bar(
                x + bar_offsets[i],
                grouped_mat[g].values,
                width=width,
                label=f"{g} (N={n_per_group.get(g, 0)})",
                color=color_map.get(g, "#cccccc"),
                align="center"
            )

        # X-ticks centered under question clusters
        ax.set_xticks(x)
        ax.set_xticklabels(grouped_mat.index, rotation=45, ha="right")

        ax.legend(loc="upper right")
        apply_modern_mpl_style(
            ax,
            xlabel="Item",
            ylabel="Mean Score",
            title=f"{label} Question Means by Irritability"
        )
        plt.tight_layout()
        figs.append((label, fig_plotly, fig_export, fig_mat))

    return figs


def plot_questions_over_time(df, ari_df, question_labels, color_map):
    import streamlit as st  # ensure available if needed for debug output

    if isinstance(question_labels, str):
        question_labels = [question_labels]

    # Prepare data
    df = df.copy()
    df["participant_code"] = df["participant_code_parent"].astype(str).str.strip()
    ari_df = ari_df.copy()
    ari_df["participant_code"] = ari_df["participant_code"].astype(str).str.strip()

    merged = df.merge(ari_df[["participant_code", "group"]], on="participant_code", how="inner")
    merged["factor_score"] = merged[question_labels].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

    # Construct ordered timepoints
    full_timepoints = [f"{d} {t}" for d in range(1, 8) for t in ["AM", "PM"]]
    merged["timepoint"] = pd.Categorical(
        merged["day_child"].astype(str) + " " + merged["time_of_day_child"],
        categories=full_timepoints,
        ordered=True
    )

    # Compute group-level stats
    grouped = merged.groupby(["timepoint", "group"])["factor_score"]
    means = grouped.mean().unstack()
    sems = grouped.sem().unstack()

    # Ensure full index even if some timepoints are missing
    means = means.reindex(full_timepoints)
    sems = sems.reindex(full_timepoints)

    # 🔧 Filter out timepoints where all groups have no data
    valid_timepoints = means.dropna(how="all").index
    means = means.loc[valid_timepoints]
    sems = sems.loc[valid_timepoints]

    ordered_groups = [g for g in ["Low", "Medium", "High"] if g in means.columns]

    # === Plotly plot ===
    fig_plotly = go.Figure()
    for grp in ordered_groups:
        color = color_map.get(grp, "#cccccc")
        fig_plotly.add_trace(go.Scatter(
            x=means.index,
            y=means[grp],
            mode="lines+markers",
            name=f"{grp} (N={merged[merged['group'] == grp]['participant_code'].nunique()})",
            line=dict(color=color),
            connectgaps=True
        ))
        upper = (means[grp] + sems[grp]).interpolate(limit_direction='both')
        lower = (means[grp] - sems[grp]).interpolate(limit_direction='both')
        fig_plotly.add_trace(go.Scatter(
            x=means.index.tolist() + means.index[::-1].tolist(),
            y=upper.tolist() + lower[::-1].tolist(),
            fill='toself',
            fillcolor=hex_to_rgba(color, 0.2),
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

    fig_plotly.update_layout(
        title=f"{' + '.join(question_labels)} Over Time by Irritability",
        xaxis_title="Timepoint",
        yaxis_title="Average Score",
        xaxis_tickangle=-45,
        width=1000
    )

    fig_export = go.Figure(fig_plotly.to_dict())
    fig_export.update_layout(
        margin=dict(l=120, r=100, t=100, b=120),
        width=1000,
        height=600
    )

    # === Matplotlib plot ===
    fig_mat, ax = plt.subplots(figsize=(6, 4))
    x_vals = range(len(means.index))
    for grp in ordered_groups:
        color = color_map.get(grp, "#cccccc")
        ax.plot(
            x_vals,
            means[grp],
            label=f"{grp} (N={merged[merged['group'] == grp]['participant_code'].nunique()})",
            marker='o',
            color=color,
            linewidth=1.5,
            markersize=4
        )
        ax.fill_between(
            x_vals,
            (means[grp] - sems[grp]).interpolate(limit_direction='both'),
            (means[grp] + sems[grp]).interpolate(limit_direction='both'),
            color=color,
            alpha=0.15,
            edgecolor=None
        )

    ax.set_title(f"{' + '.join(question_labels)} Over Time by Irritability", pad=10)
    ax.set_xlabel("Timepoint", fontdict={'fontsize': 11, 'color': 'dimgray'})
    ax.set_ylabel("Average Score", fontdict={'fontsize': 11, 'color': 'dimgray'})
    ax.set_xticks(x_vals)
    ax.set_xticklabels(means.index, rotation=45, ha="right")
    ax.tick_params(axis='x', labelsize=9, colors='dimgray')
    ax.tick_params(axis='y', labelsize=9, colors='dimgray')
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(True, axis='y', linestyle='-', alpha=0.15)
    leg = ax.legend(loc="upper right", frameon=True, fontsize='7', facecolor='white', edgecolor='lightgray')
    leg.get_frame().set_alpha(0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    plt.tight_layout()

    return fig_plotly, fig_export, fig_mat



def plot_correlation_matrix(df, ari_df, selected_questions):
    df["participant_code"] = df["participant_code_parent"].astype(str).str.strip()
    ari_df = ari_df.rename(columns={"participant number": "participant_code", "score": "score"})
    ari_df["participant_code"] = ari_df["participant_code"].astype(str).str.strip()
    merged = df.merge(ari_df[["participant_code"]], on="participant_code", how="inner")

    selected = [
        col for col in selected_questions
        if col in merged.columns and pd.api.types.is_numeric_dtype(merged[col])
    ]
    if not selected:
        raise ValueError("No valid numeric question columns selected.")

    corr = merged[selected].corr().round(2)

    # === Plotly version (unchanged) ===
    fig_plotly = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="equal",
        title=f"Questions Correlation Matrix (N={merged['participant_code'].nunique()})"
    )

    fig_plotly.update_layout(
        xaxis=dict(
            tickangle=90,
            tickfont=dict(size=10, color="black"),
            showgrid=False,
            showline=False
        ),
        yaxis=dict(
            tickfont=dict(size=10, color="black"),
            showgrid=False,
            showline=False,
            autorange="reversed"
        ),
        coloraxis_colorbar=dict(
            title="Correlation",
            thickness=12,
            x=1.01,
            xpad=10
        ),
        margin=dict(l=40, r=40, t=60, b=20),
        width=700,
        height=700
    )

    # === Plotly export figure (for PNG saving) ===
    fig_export = go.Figure(fig_plotly.to_dict())
    fig_export.update_layout(
        margin=dict(l=120, r=100, t=100, b=120),
        width=1000,
        height=900
    )

    # === Matplotlib version with text ===
    fig_mat, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_aspect('equal')

    # Show all ticks and label them
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=270, ha="right", fontsize=9, color="#4B4B4B")
    ax.set_yticklabels(corr.index, fontsize=9, color="#4B4B4B")
    

    # Show values inside the heatmap
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.5 else "#333333", fontsize=7)

    # Title and colorbar
    ax.set_title(f"Correlation Matrix (N={merged['participant_code'].nunique()})", pad=10)
    
    # Colorbar
    cbar = fig_mat.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8, colors='dimgray')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.outline.set_edgecolor('dimgray')
    cbar.outline.set_linewidth(0.8)

    # Clean up spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Layout
    fig_mat.patch.set_facecolor("white")
    plt.tight_layout(pad=1)

    return fig_plotly, fig_export, fig_mat

def plot_sync_comparison(sync_df, metric="pearson", plot_type="box", questions=None, color_map=None, show_pvalues=False):
    

    print("🔍 Starting plot_sync_comparison")
    print(f"🔍 Metric: {metric}, Plot type: {plot_type}, Show p-values: {show_pvalues}")
    print(f"🔍 Available columns: {list(sync_df.columns)}")

    if questions is not None:
        sync_df = sync_df[sync_df["question"].isin(questions)]
        print(f"🔍 Filtered questions: {questions}")

    df = sync_df.dropna(subset=[metric])
    print(f"🔍 Remaining rows after dropping NA in {metric}: {len(df)}")

    title_map = {
        "pearson": "Pearson Correlation",
        "spearman": "Spearman Correlation",
        "mad": "Mean Absolute Difference (MAD)"
    }
    y_title = title_map.get(metric, metric)

    if "group" not in df.columns:
        df["group"] = "All"
        print("⚠️ 'group' column missing, set to 'All'")

    if color_map is None:
        color_map = {
            "Low": "#cbdfe0",
            "Medium": "#88babc",
            "High": "#114f52",
            "All": "#888888"
        }

    counts = df.groupby(["question", "group"]).size().unstack(fill_value=0)
    question_labels = {
        q: f"{q} (N={','.join(str(counts.loc[q].get(g, 0)) for g in ['Low', 'Medium', 'High'] if g in counts.columns)})"
        for q in counts.index
    }

    group_order = [g for g in ["Low", "Medium", "High", "All"] if g in df["group"].unique() and pd.notna(g)]
    df["group"] = pd.Categorical(df["group"], categories=group_order, ordered=True)
    df["question_label"] = df["question"].map(question_labels)

    print(f"🔍 Group order: {group_order}")
    print(f"🔍 Color map: {color_map}")

    fig_func = px.box if plot_type == "box" else px.violin
    fig_kwargs = dict(
        x="group", y=metric, color="group",
        facet_col="question_label", facet_col_wrap=4,
        title=f"Parent-Child Synchronization by Group ({y_title})",
        points="all", height=600,
        color_discrete_map=color_map,
        category_orders={"group": group_order}
    )
    if plot_type == "violin":
        fig_kwargs["box"] = True
    fig_plotly = fig_func(df, **fig_kwargs)

    if show_pvalues:
        print("🔍 Adding p-value annotations to Plotly plot...")
        for question_label, group_df in df.groupby("question_label"):
            safe_order = [g for g in group_df["group"].cat.categories if g in group_df["group"].unique()]
            for i, (g1, g2) in enumerate(itertools.combinations(safe_order, 2)):
                d1 = group_df[group_df["group"] == g1][metric]
                d2 = group_df[group_df["group"] == g2][metric]
                if len(d1) < 2 or len(d2) < 2:
                    continue
                _, pval = ttest_ind(d1, d2, equal_var=False)
                if pval < 0.001:
                    stars = "***"
                elif pval < 0.01:
                    stars = "**"
                elif pval < 0.05:
                    stars = "*"
                else:
                    continue

                x0, x1 = group_order.index(g1), group_order.index(g2)
                fig_plotly.add_annotation(
                    text=stars,
                    x=(x0 + x1) / 2,
                    y=group_df[metric].max() + 0.05 + i * 0.03,
                    showarrow=False,
                    xref="x domain",
                    yref="y",
                    font=dict(size=12, color="#444")
                )

    fig_plotly.update_layout(
        yaxis_title=y_title,
        font=dict(size=11, color="#4B4B4B"),
        title_font_size=16,
        margin=dict(t=80, b=60),
        showlegend=False,
    )
    fig_plotly.for_each_xaxis(lambda axis: axis.update(title=None, tickangle=0))
    fig_plotly.for_each_annotation(lambda a: a.update(text=a.text.replace("question_label=", "").strip()))

    fig_export = go.Figure(fig_plotly.to_dict())
    fig_export.update_layout(width=700, height=550, margin=dict(l=120, r=100, t=100, b=120))
    fig_export.update_yaxes(zeroline=False)

    # Matplotlib
    num_questions = df["question"].nunique()
    n_cols = 4
    n_rows = -(-num_questions // n_cols)
    fig_mat, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.8 * n_rows), squeeze=False)

    grouped = df.groupby("question")
    global_y_max = df[metric].max()
    global_y_lim = max(1.05, global_y_max + 0.15)

    i = 0
    for question, group_data in grouped:
        if group_data.empty:
            continue
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        safe_order = [g for g in ["Low", "Medium", "High"] if g in group_data["group"].unique()]
        group_data["group"] = pd.Categorical(group_data["group"], categories=safe_order, ordered=True)
        palette = {g: color_map.get(g, "#cccccc") for g in safe_order}

        n_counts = group_data["group"].value_counts().reindex(safe_order).fillna(0).astype(int).tolist()
        title = f"{question} (N={','.join(map(str, n_counts))})"

        # === Use compact positions ===
        positions = {
            2: [0.3, 0.7],
            3: [0.2, 0.5, 0.8]
        }.get(len(safe_order), list(range(len(safe_order))))

        # Create custom colored boxes
        bp = ax.boxplot(
            [group_data[group_data["group"] == g][metric] for g in safe_order],
            positions=positions,
            widths=0.2,
            patch_artist=True,
            boxprops=dict(linewidth=1.5),
            medianprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker='o', markersize=4, alpha=0)
        )

        for j, group in enumerate(safe_order):
            base_color = palette[group]
            edge_color = to_rgba(base_color, alpha=1)
            face_color = to_rgba(base_color, alpha=0.5)

            bp['boxes'][j].set(facecolor=face_color, edgecolor=edge_color, linewidth=1.2)
            bp['whiskers'][2*j].set_color(edge_color)
            bp['whiskers'][2*j+1].set_color(edge_color)
            bp['caps'][2*j].set_color(edge_color)
            bp['caps'][2*j+1].set_color(edge_color)
            bp['medians'][j].set_color(edge_color)
            bp['medians'][j].set_linewidth(1.2)

            # Scatter dots to the left
            group_vals = group_data[group_data["group"] == group][metric]
            jitter_x = np.random.normal(positions[j] - 0.14, 0.03, size=len(group_vals))
            ax.scatter(jitter_x, group_vals, color=base_color, alpha=0.4, s=10, zorder=3)

        ax.set_ylim(-1, global_y_lim)
        ax.set_xlim(0, 1)

        # P-value stars between box pairs
        if show_pvalues and len(safe_order) > 1:
            y_base = group_data[metric].max() + 0.03
            offset = 0.04
            for idx, (g1, g2) in enumerate(itertools.combinations(safe_order, 2)):
                d1 = group_data[group_data["group"] == g1][metric]
                d2 = group_data[group_data["group"] == g2][metric]
                if len(d1) < 2 or len(d2) < 2:
                    continue
                _, pval = ttest_ind(d1, d2, equal_var=False)
                x1 = positions[safe_order.index(g1)]
                x2 = positions[safe_order.index(g2)]
                add_pvalue_manual(ax, x1, x2, y_base + idx * offset, h=0.01, pval=pval)

        ax.set_xticks(positions)
        ax.set_xticklabels(safe_order)
        ax.set_xlabel("")

        if col == 0:
            apply_modern_mpl_style(ax, title=title, ylabel=y_title)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(axis='y', left=False)
            ax.grid(False, axis='y')
            ax.spines['left'].set_visible(False)
            apply_modern_mpl_style(ax, title=title)

        i += 1

    for j in range(i, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row][col].axis("off")

    fig_mat.tight_layout(w_pad=0.8, h_pad=1.5)
    fig_mat.subplots_adjust(wspace=0.02, hspace=0.4)
    return fig_plotly, fig_export, fig_mat


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    ari_df = pd.read_excel(ARI_PATH)
    ari_df = assign_ari_groups(ari_df, low_th=3, high_th=5)

    color_map = {
        "Low": PALETTE[4],
        "Medium": "#ff7f0e",
        "High": "#2e6c70"
    }

    plot_aggression(df.copy(), ari_df)
    plot_means_by_irritability(df.copy(), ari_df, ["C_Irr_Frustration"], color_map=color_map)
    plot_questions_over_time(df.copy(), ari_df, "C_Irr_Frustration", color_map=color_map)
    plot_correlation_matrix(df.copy(), ari_df)

    # New: Synchronization comparison
    sync_path = os.path.join(PROJECT_ROOT, "output", "sync_df.csv")
    sync_df = pd.read_csv(sync_path)

    fig = plot_sync_comparison(sync_df, metric="pearson", plot_type="box", color_map=color_map)
    fig.savefig(os.path.join(OUTPUT_DIR, "sync_comparison_pearson_box.png"), dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()

