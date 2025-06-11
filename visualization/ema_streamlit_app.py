import streamlit as st
import pandas as pd
import os
import sys

# === Path Config ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSING_DIR = os.path.join(PROJECT_ROOT, "processing")
sys.path.insert(0, PROCESSING_DIR)

from labeling import build_label_to_hebrew_map

DATA_PATH = os.path.join(PROJECT_ROOT, "output", "merged_surveys.csv")
ARI_PATH = os.path.join(PROJECT_ROOT, "data", "ARI.xlsx")
CHILD_LABELING_PATH = os.path.join(PROJECT_ROOT, "data", "children_labeling.xlsx")
PARENT_LABELING_PATH = os.path.join(PROJECT_ROOT, "data", "parent_labeling.xlsx")


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHILD_LABEL_MAP = build_label_to_hebrew_map(CHILD_LABELING_PATH)
PARENT_LABEL_MAP = build_label_to_hebrew_map(PARENT_LABELING_PATH)

# Make project root importable (for visualize_all.py)
sys.path.append(PROJECT_ROOT)

# Import plotting functions
from visualize_all import (
    plot_aggression,
    plot_means_by_irritability,
    plot_questions_over_time,
    plot_correlation_matrix
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
    "Factor Correlation Matrix"
])

# === Visualization logic ===
if vis_type == "Aggression Pie Charts":
    st.subheader("Aggression Type Distribution by Group")
    plot_aggression(df, ari_df)
    st.image(os.path.join(OUTPUT_DIR, "aggression_pie_by_group.png"))

elif vis_type == "Mean Scores by Group":
    st.subheader("Mean Scores for Child and Parent Questions")
    plot_means_by_irritability(df, ari_df)
    st.image(os.path.join(OUTPUT_DIR, "question_means_child_from_average.png"))
    st.image(os.path.join(OUTPUT_DIR, "question_means_parent_from_average.png"))

elif vis_type == "Question Over Time":
    st.subheader("Trends Over Time by Irritability Group")

    group = st.radio("Choose data group", ["Child", "Parent"])
    prefix = "C_" if group == "Child" else "P_"

    group_columns = [f"{prefix}{q}" for q in SCALE_QUESTIONS if f"{prefix}{q}" in df.columns]

    if group_columns:
        question = st.selectbox("Choose a question to visualize", group_columns)

        label_map = CHILD_LABEL_MAP if group == "Child" else PARENT_LABEL_MAP
        hebrew_text = label_map.get(question, "שאלה בעברית לא זמינה")

        if group == "Child":
            hebrew_text = hebrew_text.replace("אמא", "ההורה").replace("אבא", "ההורה")

        st.markdown(f"**Hebrew question:** {hebrew_text}")

        # Plot and show image
        plot_questions_over_time(df, ari_df, question)
        fname = f"question_trend_{question.lower()}.png"
        st.image(os.path.join(OUTPUT_DIR, fname))
    else:
        st.warning("No valid questions found for this group.")

elif vis_type == "Factor Correlation Matrix":
    st.subheader("Parent-Child Factor Correlation Matrix")
    plot_correlation_matrix(df, ari_df)
    st.image(os.path.join(OUTPUT_DIR, "factor_correlation_matrix.png"))
