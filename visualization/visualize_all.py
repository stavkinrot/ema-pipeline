import pandas as pd
import matplotlib.pyplot as plt
import os
from colors_config import PALETTE, AGGRESSION_COLORS
from factor_config import FACTOR_QUESTIONS

# === Shared Config ===
DATA_PATH = "output/merged_surveys.csv"
ARI_PATH = "data/ARI.xlsx"
OUTPUT_DIR = "output"
FACTOR_QUESTIONS = {
    "Anxiety": ["C_Anx_Worry", "C_Anx_now"],
    "Irritability": ["C_Irr_Frustration", "C_Angry_now", "C_triggers"],
    # Add more factors here...
}


def plot_aggression(df):
    AGGRESSION_COLS = [
        "C_Agr_other",
        "C_Agr_slam",
        "C_Agr_throw_smt",
        "C_Agr_throw_twd",
        "C_Agr_yelled"
    ]

    aggr_df = df[df["C_Agr_none"].isin(["Yes", "No"])]

    pie1_counts = aggr_df["C_Agr_none"].value_counts().reindex(["Yes", "No"], fill_value=0)
    pie1_labels = ["No Aggression", "Some Aggression"]
    pie1_values = [pie1_counts.get("Yes", 0), pie1_counts.get("No", 0)]
    pie1_colors = [AGGRESSION_COLORS["none"], AGGRESSION_COLORS["some"]]

    aggr_only_df = aggr_df[aggr_df["C_Agr_none"] == "No"]
    pie2_labels = {
        "C_Agr_other": "Other",
        "C_Agr_slam": "Slammed something",
        "C_Agr_throw_smt": "Threw object",
        "C_Agr_throw_twd": "Threw toward someone",
        "C_Agr_yelled": "Yelled"
    }

    pie2_labels_ordered = []
    pie2_values = []
    pie2_colors = []

    for col in AGGRESSION_COLS:
        pie2_labels_ordered.append(pie2_labels[col])
        if col == "C_Agr_other":
            value = aggr_only_df[col].apply(lambda x: isinstance(x, str) and x.strip() != "").sum()
        else:
            value = (aggr_only_df[col] == "Yes").sum()
        pie2_values.append(value)
        pie2_colors.append(AGGRESSION_COLORS[col])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].pie(pie1_values, labels=pie1_labels, autopct="%1.1f%%", startangle=90, colors=pie1_colors)
    axes[0].set_title("Aggression Presence in Surveys")

    axes[1].pie(pie2_values, labels=pie2_labels_ordered, autopct="%1.1f%%", startangle=90, colors=pie2_colors)
    axes[1].set_title("Types of Aggression (among 'Yes')")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "aggression_pie_charts.png"), dpi=300)
    plt.close()
    print("[INFO] Saved aggression plot to output/aggression_pie_charts.png")


def plot_means_by_irritability(df, ari_df):
    ari_df = ari_df.rename(columns={"participant number": "participant_code", "score": "score"})
    ari_df["participant_code"] = ari_df["participant_code"].astype(str).str.strip()
    ari_df["group"] = ari_df["score"].apply(lambda x: "Irritable" if x >= 3 else "Non-Irritable")

    df["participant_code"] = df["participant_code_parent"].astype(str).str.strip()
    merged = df.merge(ari_df[["participant_code", "group"]], on="participant_code", how="inner")

    numeric_cols = [
        col for col in merged.columns
        if (col.startswith("C_") or col.startswith("P_"))
        and pd.api.types.is_numeric_dtype(merged[col])
    ]
    grouped_means = merged.groupby("group")[numeric_cols].mean().T

    fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols) * 0.5), 6))
    bar_width = 0.35
    x = range(len(numeric_cols))

    ax.bar([i - bar_width/2 for i in x], grouped_means["Irritable"], width=bar_width, label="Irritable", color=PALETTE[0])
    ax.bar([i + bar_width/2 for i in x], grouped_means["Non-Irritable"], width=bar_width, label="Non-Irritable", color=PALETTE[3])

    ax.set_xticks(x)
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_ylabel("Mean Score")
    ax.set_title("Mean Survey Responses by Irritability Group")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mean_by_irritability.png"), dpi=300)
    plt.close()
    print("[INFO] Saved mean comparison plot to output/mean_by_irritability.png")


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    ari_df = pd.read_excel(ARI_PATH)
    plot_aggression(df)
    plot_means_by_irritability(df, ari_df)


if __name__ == "__main__":
    main()
