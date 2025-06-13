import streamlit as st
import pandas as pd
import os
import sys
from included_questions import INCLUDED_QUESTIONS
from io import BytesIO
import plotly.io as pio
import matplotlib.pyplot as plt
from visualization.colors_config import PALETTE
from visualize_all import assign_ari_groups

# === Path Config ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSING_DIR = os.path.join(PROJECT_ROOT, "processing")
sys.path.insert(0, PROCESSING_DIR)

from labeling import build_label_to_hebrew_map

DATA_PATH = os.path.join(PROJECT_ROOT, "output", "merged_surveys.csv")
ARI_PATH = os.path.join(PROJECT_ROOT, "data", "ARI.xlsx")
CHILD_LABELING_PATH = os.path.join(PROJECT_ROOT, "data", "children_labeling.xlsx")
PARENT_LABELING_PATH = os.path.join(PROJECT_ROOT, "data", "parent_labeling.xlsx")
SYNC_DF_PATH = os.path.join(PROJECT_ROOT, "output", "sync_df.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHILD_LABEL_MAP = build_label_to_hebrew_map(CHILD_LABELING_PATH)
PARENT_LABEL_MAP = build_label_to_hebrew_map(PARENT_LABELING_PATH)

# Make project root importable (for visualize_all.py)
sys.path.append(PROJECT_ROOT)

from visualize_all import (
    plot_aggression,
    plot_means_by_irritability,
    plot_questions_over_time,
    plot_correlation_matrix,
    plot_sync_comparison
)

# === Valid scale questions for both child and parent ===
SCALE_QUESTIONS = [
    "ADHD_Distracted", "ADHD_Restless", "Agr_NotAsWant",
    "Angry_now", "Anx_Worry", "Anx_now", "Discipline",
    "IC_CantStop", "IC_FirstOnMind",
    "Inv_Fun", "Inv_Help", "Inv_Talk",
    "Irr_Frustration", "Mood_Good", "Mood_Sad",
    "PC_Annoy", "PC_Criticism", "PC_Sharing",
    "PS_Agree", "PS_GotAngry", "PS_Patient", "Positive"
]

# === Load data ===
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    ari_df = pd.read_excel(ARI_PATH)
    return df, ari_df

df, ari_df = load_data()


# === Sidebar controls ===
st.sidebar.title("EMA Visualization")
vis_type = st.sidebar.selectbox("Choose Visualization Type", [
    "Aggression Pie Charts",
    "Mean Scores by Group",
    "Question Over Time",
    "Questions Correlation Matrix",
    "Dyadic Synchronization Comparison"
])

# === Utility functions ===
def download_buttons(fig_plotly=None, fig_mat=None, filename_prefix="plot"):
    if fig_plotly:
        buf = BytesIO()
        pio.write_image(fig_plotly, buf, format="png", scale=3)
        st.download_button(
            label="Download as PNG (Plotly)",
            data=buf.getvalue(),
            file_name=f"{filename_prefix}_plotly.png",
            mime="image/png"
        )
    if fig_mat:
        buf = BytesIO()
        fig_mat.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            label="Download as PNG (Matplotlib)",
            data=buf.getvalue(),
            file_name=f"{filename_prefix}_matplotlib.png",
            mime="image/png"
        )
        plt.close(fig_mat)

# === Visualization logic ===
if vis_type == "Aggression Pie Charts":
    for group, fig_plotly, fig_mat in plot_aggression(df, ari_df):
        st.subheader(f"{group} Group")
        st.plotly_chart(fig_plotly, use_container_width=True)
        download_buttons(fig_plotly, fig_mat, filename_prefix=f"aggression_{group.lower()}")

elif vis_type == "Mean Scores by Group":
    st.subheader("Mean Scores by Irritability Group")

    with st.sidebar.expander("⚙️ Settings", expanded=True):
        st.markdown("### 🔧 ARI Thresholds")
        th_range = st.slider(
            "Set thresholds",
            min_value=1, max_value=7, value=(3, 5), step=1,
            help="If thresholds overlap, two groups will be used. Otherwise, three groups"
        )

        st.markdown("### 🎨 Plot Group Colors")
        color_low = st.color_picker("Low", PALETTE[3])
        color_med = st.color_picker("Medium", PALETTE[4])
        color_high = st.color_picker("High", PALETTE[1])

        apply_changes = st.button("✅ Apply")

    if "mean_scores_config" not in st.session_state:
        st.session_state.mean_scores_config = {
            "low_th": th_range[0],
            "high_th": th_range[1],
            "colors": {
                "Low": color_low,
                "Medium": color_med,
                "High": color_high
            }
        }

    if apply_changes:
        st.session_state.mean_scores_config = {
            "low_th": th_range[0],
            "high_th": th_range[1],
            "colors": {
                "Low": color_low,
                "Medium": color_med,
                "High": color_high
            }
        }

    config = st.session_state.mean_scores_config
    low_th, high_th = config["low_th"], config["high_th"]
    color_map = config["colors"]
    ari_df_grouped = assign_ari_groups(ari_df, low_th, high_th)

    # === Group & Question Selection ===
    group = st.radio("Select group", ["Children", "Parents"])
    prefix = "C_" if group == "Children" else "P_"

    all_scale_questions = [f"{prefix}{q}" for q in SCALE_QUESTIONS if f"{prefix}{q}" in df.columns]
    default_included = [q for q in INCLUDED_QUESTIONS if q.startswith(prefix) and q in all_scale_questions]

    previous = st.session_state.get("mean_scores_selector", default_included)
    stripped_previous = {q[2:] for q in previous}
    converted_selection = [f"{prefix}{q}" for q in stripped_previous if f"{prefix}{q}" in all_scale_questions]
    st.session_state["mean_scores_selector"] = converted_selection

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Choose All", key="choose_all_means"):
            st.session_state["mean_scores_selector"] = all_scale_questions
    with col2:
        if st.button("Default", key="choose_default_means"):
            st.session_state["mean_scores_selector"] = default_included

    selected = st.multiselect(
        "Select questions to include:",
        options=all_scale_questions,
        default=st.session_state["mean_scores_selector"],
        key="mean_scores_selector"
    )

    if selected:
        figs = plot_means_by_irritability(df, ari_df_grouped, selected, color_map)
        for label, fig_plotly, fig_export, fig_mat in figs:
            if label == group:
                st.plotly_chart(fig_plotly, use_container_width=True)
                download_buttons(fig_plotly, fig_mat, filename_prefix=f"mean_scores_{group.lower()}")
    else:
        st.warning("Please select at least one question.")

elif vis_type == "Question Over Time":
    st.subheader("Trends Over Time by Irritability Group")

    group = st.radio("Choose data group", ["Child", "Parent"])
    prefix = "C_" if group == "Child" else "P_"

    group_columns = [f"{prefix}{q}" for q in SCALE_QUESTIONS if f"{prefix}{q}" in df.columns]

    if group_columns:
        question = st.selectbox("Choose a question to visualize", group_columns)
        label_map = CHILD_LABEL_MAP if group == "Child" else PARENT_LABEL_MAP
        hebrew_text = label_map.get(question, "שאלה בעברית לא זמינה").replace("אמא", "ההורה").replace("אבא", "ההורה")
        st.markdown(f"**Hebrew question:** {hebrew_text}")

        # === Sidebar: ARI thresholds and colors ===
        with st.sidebar.expander("⚙️ Settings", expanded=True):
            st.markdown("### 🔧 ARI Thresholds")
            th_range = st.slider(
                "Set thresholds",
                min_value=1, max_value=7, value=(3, 5), step=1,
                help="If thresholds overlap, two groups will be used. Otherwise, three groups"
            )

            st.markdown("### 🎨 Plot Group Colors")
            default_colors = {
                "Low": PALETTE[3],
                "Medium": PALETTE[4],
                "High": PALETTE[1]
            }

            # Reset colors if requested
            if "qot_color_low" not in st.session_state:
                st.session_state.qot_color_low = default_colors["Low"]
            if "qot_color_med" not in st.session_state:
                st.session_state.qot_color_med = default_colors["Medium"]
            if "qot_color_high" not in st.session_state:
                st.session_state.qot_color_high = default_colors["High"]

            col_apply, col_reset = st.columns([1, 1])
            with col_reset:
                if st.button("↩️ Default Colors", key="reset_qot_colors"):
                    st.session_state.qot_color_low = default_colors["Low"]
                    st.session_state.qot_color_med = default_colors["Medium"]
                    st.session_state.qot_color_high = default_colors["High"]

            # Draw color pickers using session state
            color_low = st.color_picker("Low", key="qot_color_low")
            color_med = st.color_picker("Medium", key="qot_color_med")
            color_high = st.color_picker("High", key="qot_color_high")

            with col_apply:
                if st.button("✅ Apply", key="apply_qot_config"):
                    st.session_state.qot_config = {
                        "low_th": th_range[0],
                        "high_th": th_range[1],
                        "colors": {
                            "Low": st.session_state.qot_color_low,
                            "Medium": st.session_state.qot_color_med,
                            "High": st.session_state.qot_color_high,
                        }
                    }

            if "qot_config" not in st.session_state:
                st.session_state.qot_config = {
                    "low_th": th_range[0],
                    "high_th": th_range[1],
                    "colors": {
                        "Low": color_low,
                        "Medium": color_med,
                        "High": color_high
                    }
                }

        # Extract config
        config = st.session_state.qot_config
        low_th, high_th = config["low_th"], config["high_th"]
        color_map = config["colors"]

        # Assign ARI groups dynamically
        ari_df_grouped = assign_ari_groups(ari_df.copy(), low_th, high_th)

        # Plot
        fig_plotly, fig_export, fig_mat = plot_questions_over_time(
            df, ari_df_grouped, question, color_map=color_map
        )
        st.plotly_chart(fig_plotly, use_container_width=True)
        download_buttons(fig_export, fig_mat, filename_prefix=f"question_over_time_{question}")
    else:
        st.warning("No valid questions found for this group.")



elif vis_type == "Questions Correlation Matrix":
    st.subheader("Questions Correlation Matrix")

    all_numeric_questions = [
        col for col in df.columns
        if (col.startswith("C_") or col.startswith("P_")) and pd.api.types.is_numeric_dtype(df[col])
    ]
    default_questions = [q for q in INCLUDED_QUESTIONS if q in all_numeric_questions]

    if "correlation_matrix_selected" not in st.session_state:
        st.session_state.correlation_matrix_selected = default_questions

    st.markdown("### 📊 Filter by Correlation Strength")
    row1, row2 = st.columns([3, 1])
    with row1:
        threshold = st.slider(
            "Select minimum correlation between questions:",
            min_value=0.1,
            max_value=0.95,
            value=0.6,
            step=0.05,
            key="corr_slider",
            help="Includes questions that have at least one strong correlation (positive or negative)."
        )
    with row2:
        if st.button("🔍 Apply Filter"):
            corr_matrix = df[all_numeric_questions].corr().abs()
            corr_mask = (corr_matrix > threshold) & (corr_matrix < 1.0)
            filtered_questions = sorted(set(
                col for col in corr_mask.columns if corr_mask[col].any()
            ))
            st.session_state.correlation_matrix_selected = filtered_questions
            if filtered_questions:
                st.success(f"Selected {len(filtered_questions)} questions with |correlation| ≥ {threshold}")
            else:
                st.warning("No questions met the correlation threshold.")

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("✅ Choose All"):
            st.session_state.correlation_matrix_selected = all_numeric_questions
    with btn_col2:
        if st.button("↩️ Default"):
            st.session_state.correlation_matrix_selected = default_questions

    valid_defaults = [q for q in st.session_state.correlation_matrix_selected if q in all_numeric_questions]

    selected_corr_questions = st.multiselect(
        "Select questions to include in Correlation Matrix",
        options=all_numeric_questions,
        default=valid_defaults,
        key="correlation_matrix_selector"
    )

    if selected_corr_questions:
        fig_plotly, fig_export, fig_mat = plot_correlation_matrix(df, ari_df, selected_corr_questions)
        st.plotly_chart(fig_plotly, use_container_width=True)
        download_buttons(fig_export, fig_mat, filename_prefix="correlation_matrix")
    else:
        st.warning("Please select at least one question.")

elif vis_type == "Dyadic Synchronization Comparison":
    st.subheader("Dyadic Synchronization: Parent-Child Agreement")

    @st.cache_data
    def load_sync_df():
        return pd.read_csv(SYNC_DF_PATH, encoding="utf-8-sig")

    sync_df = load_sync_df()

    # === Sidebar: ARI thresholds and colors ===
    with st.sidebar.expander("⚙️ Settings", expanded=True):
        st.markdown("### 🔧 ARI Thresholds")
        th_range = st.slider(
            "Set thresholds",
            min_value=1, max_value=7, value=(3, 5), step=1,
            help="If thresholds overlap, two groups will be used. Otherwise, three groups"
        )

        st.markdown("### 🎨 Plot Group Colors")
        color_low = st.color_picker("Low", PALETTE[3])
        color_med = st.color_picker("Medium", PALETTE[4])
        color_high = st.color_picker("High", PALETTE[1])

        apply_changes = st.button("✅ Apply", key="sync_apply")

    if "sync_config" not in st.session_state:
        st.session_state.sync_config = {
            "low_th": th_range[0],
            "high_th": th_range[1],
            "colors": {
                "Low": color_low,
                "Medium": color_med,
                "High": color_high
            }
        }

    if apply_changes:
        st.session_state.sync_config = {
            "low_th": th_range[0],
            "high_th": th_range[1],
            "colors": {
                "Low": color_low,
                "Medium": color_med,
                "High": color_high
            }
        }

    config = st.session_state.sync_config
    low_th, high_th = config["low_th"], config["high_th"]
    color_map = config["colors"]

    # Assign ARI groups
    ari_df_grouped = assign_ari_groups(ari_df.copy(), low_th, high_th)
    
    # Merge with sync_df
    if "participant_code_parent" in sync_df.columns:
        sync_df = sync_df.drop(columns=["group"], errors="ignore")  # Drop if exists

        sync_df["participant_code_parent"] = sync_df["participant_code_parent"].astype(str).str.strip()
        ari_df_grouped["participant_code"] = ari_df_grouped["participant_code"].astype(str).str.strip()

        merged_df = sync_df.merge(
            ari_df_grouped[["participant_code", "group"]],
            left_on="participant_code_parent",
            right_on="participant_code",
            how="left"
        )
        merged_df.drop(columns=["participant_code"], inplace=True)

    else:
        st.warning("⚠️ participant_code_parent not found in sync_df. Defaulting group to 'All'.")
        sync_df["group"] = "All"
        merged_df = sync_df

    # === Plot controls ===
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        correlation_type = st.selectbox("Select correlation type", ["pearson", "spearman", "mad"], key="sync_corr_type")
    with col2:
        plot_type = st.selectbox("Select plot type", ["box", "violin"], key="sync_plot_type")
    with col3:
        group_mode = st.radio("Grouping", ["Separate groups", "Combine groups"], key="sync_group_mode")

    all_sync_questions = SCALE_QUESTIONS + ["Agr_none"]
    default_questions = ["Inv_Fun", "PS_Agree", "PS_Patient", "Agr_none"]

    if "sync_questions_selected" not in st.session_state:
        st.session_state.sync_questions_selected = default_questions

    btn1, btn2 = st.columns([1, 1])
    with btn1:
        if st.button("✅ Choose All", key="choose_all_sync"):
            st.session_state.sync_questions_selected = all_sync_questions
    with btn2:
        if st.button("↩️ Default", key="default_sync"):
            st.session_state.sync_questions_selected = default_questions

    selected_sync_questions = st.multiselect(
        "Select questions to include:",
        options=all_sync_questions,
        default=st.session_state.sync_questions_selected,
        key="sync_questions_selector"
    )

    if selected_sync_questions:
        df_to_plot = merged_df.copy()
        if group_mode == "Combine groups":
            df_to_plot["group"] = "All"

        fig_plotly, fig_export, fig_mat = plot_sync_comparison(
            df_to_plot,
            metric=correlation_type,
            plot_type=plot_type,
            questions=selected_sync_questions,
            color_map=color_map
        )
        st.plotly_chart(fig_plotly, use_container_width=True)
        download_buttons(fig_export, fig_mat, filename_prefix=f"sync_comparison_{correlation_type}_{plot_type}")
    else:
        st.warning("Please select at least one question to display.")

