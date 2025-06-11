from io import BytesIO
import sys
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import base64

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

# === Preprocessing Helper ===
def preprocess_ari_df(ari_df):
    df = ari_df.rename(columns={"participant number": "participant_code", "score": "score"}).copy()
    df["participant_code"] = df["participant_code"].astype(str).str.strip()
    df["group"] = pd.to_numeric(df["score"], errors="coerce").apply(
        lambda x: "Irritable" if pd.notna(x) and x >= 3 else "Non-Irritable"
    )
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
            title=f"{group} Group (n={n}) – Avg Aggression: {aggr_pct:.1f}% (N={n_total})",
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
        ax.set_title(f"{group} Group (n={n}) – Avg Aggression: {aggr_pct:.1f}% (N={n_total})")
        plt.tight_layout()

        figs.append((group, fig_plotly, fig_mat))

    return figs


def plot_means_by_irritability(df, ari_df, selected_questions):
    avg_df = pd.read_csv("output/child_parent_averages.csv")

    ari_df = ari_df.rename(columns={"participant number": "participant_code", "score": "score"})
    ari_df["participant_code"] = ari_df["participant_code"].astype(str).str.strip()
    ari_df["group"] = pd.to_numeric(ari_df["score"], errors="coerce").apply(
        lambda x: "Irritable" if pd.notna(x) and x >= 3 else "Non-Irritable"
    )

    avg_df["participant_code"] = avg_df["participant_code_parent"].astype(str).str.strip()
    merged = avg_df.merge(ari_df[["participant_code", "group"]], on="participant_code", how="inner")

    figs = []

    for role, label in zip(["child", "parent"], ["Child", "Parent"]):
        role_prefix = "C_" if role == "child" else "P_"
        role_cols = [col for col in selected_questions if col.startswith(role_prefix)]
        if not role_cols:
            continue

        df_role = merged[["group"] + role_cols].copy()
        melted = df_role.melt(id_vars="group", var_name="Question", value_name="Mean Score")

        # Compute group means and N
        grouped = melted.groupby(["Question", "group"])["Mean Score"].mean().reset_index()
        n_per_group = df_role.groupby("group")["group"].count().to_dict()

        # Plotly interactive chart
        grouped["group_with_n"] = grouped["group"].apply(
            lambda g: f"{g} (N={n_per_group.get(g, 0)})"
        )
        fig_plotly = px.bar(
            grouped,
            x="Question",
            y="Mean Score",
            color="group_with_n",
            barmode="group",
            title=f"{label} Question Means by Irritability",
            color_discrete_map={
                f"Irritable (N={n_per_group.get('Irritable', 0)})": PALETTE[0],
                f"Non-Irritable (N={n_per_group.get('Non-Irritable', 0)})": PALETTE[4]
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
        width = 0.35

        fig_mat, ax = plt.subplots(figsize=(max(10, len(grouped_mat) * 0.6), 5))
        ax.bar(x - width / 2, grouped_mat.get("Irritable", [0]*len(x)),
               width, label=f"Irritable (N={n_per_group.get('Irritable', 0)})", color=PALETTE[0])
        ax.bar(x + width / 2, grouped_mat.get("Non-Irritable", [0]*len(x)),
               width, label=f"Non-Irritable (N={n_per_group.get('Non-Irritable', 0)})", color=PALETTE[4])

        ax.set_ylabel("Mean Score")
        ax.set_xticks(x)
        ax.set_xticklabels(grouped_mat.index, rotation=45, ha="right")
        ax.set_title(f"{label} Question Means by Irritability")
        ax.legend()
        plt.tight_layout()

        figs.append((label, fig_plotly, fig_export, fig_mat))

    return figs


def plot_questions_over_time(df, ari_df, question_labels):
    if isinstance(question_labels, str):
        question_labels = [question_labels]

    # Prepare data
    df["participant_code"] = df["participant_code_parent"].astype(str).str.strip()
    ari_df = ari_df.rename(columns={"participant number": "participant_code", "score": "score"})
    ari_df["participant_code"] = ari_df["participant_code"].astype(str).str.strip()
    ari_df["group"] = pd.to_numeric(ari_df["score"], errors="coerce").apply(
        lambda x: "Irritable" if pd.notna(x) and x >= 3 else "Non-Irritable"
    )

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

    # 🔧 Filter out timepoints where both groups have no data
    valid_timepoints = means.dropna(how="all").index
    means = means.loc[valid_timepoints]
    sems = sems.loc[valid_timepoints]

    # === Plotly plot ===
    fig_plotly = go.Figure()
    for grp, color in zip(["Irritable", "Non-Irritable"], ["#2e6c70", PALETTE[4]]):
        if grp in means:
            fig_plotly.add_trace(go.Scatter(
                x=means.index,
                y=means[grp],
                mode="lines+markers",
                name=f"{grp} (N={merged[merged['group'] == grp]['participant_code'].nunique()})",
                line=dict(color=color),
                connectgaps=True
            ))

            # Interpolate SEM values for shading
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

    # === Plotly export figure (for PNG saving) ===
    fig_export = go.Figure(fig_plotly.to_dict())
    fig_export.update_layout(
        margin=dict(l=120, r=100, t=100, b=120),
        width=1000,
        height=600
    )

    # === Matplotlib plot ===
    fig_mat, ax = plt.subplots(figsize=(6, 4))
    x_vals = range(len(means.index))
    for grp, color in zip(["Irritable", "Non-Irritable"], ["#2e6c70", PALETTE[4]]):
        if grp in means:
            ax.plot(x_vals, means[grp], label=f"{grp} (N={merged[merged['group'] == grp]['participant_code'].nunique()})", marker='o', color=color)
            ax.fill_between(
                x_vals,
                (means[grp] - sems[grp]).interpolate(limit_direction='both'),
                (means[grp] + sems[grp]).interpolate(limit_direction='both'),
                color=color,
                alpha=0.2
            )

    ax.set_title(f"{' + '.join(question_labels)} Over Time by Irritability")
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Average Score")
    ax.set_xticks(x_vals)
    ax.set_xticklabels(means.index, rotation=45, ha="right")
    ax.legend()
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

    # Show all ticks and label them
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)

    # Show values inside the heatmap
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    color="white" if abs(value) > 0.5 else "black", fontsize=7)

    ax.set_title(f"Correlation Matrix (N={merged['participant_code'].nunique()})")
    fig_mat.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    return fig_plotly, fig_export, fig_mat



def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    ari_df = pd.read_excel(ARI_PATH)
    ari_df = preprocess_ari_df(ari_df)

    plot_aggression(df.copy(), ari_df)
    plot_means_by_irritability(df.copy(), ari_df)
    plot_questions_over_time(df.copy(), ari_df, "C_Irr_Frustration")
    plot_correlation_matrix(df.copy(), ari_df)

if __name__ == "__main__":
    main()

