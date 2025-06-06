import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
from colors_config import PALETTE, AGGRESSION_COLORS
from factor_config import FACTOR_QUESTIONS
from included_questions import INCLUDED_QUESTIONS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === Shared Config ===
DATA_PATH = "output/merged_surveys.csv"
ARI_PATH = "data/ARI.xlsx"
AVG_PATH = "output/child_parent_averages.csv"
OUTPUT_DIR = "output/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_aggression(df, ari_df):
    AGGRESSION_COLS = [
        "C_Agr_other",
        "C_Agr_slam",
        "C_Agr_throw_smt",
        "C_Agr_throw_twd",
        "C_Agr_yelled"
    ]
    pie2_labels = {
        "C_Agr_other": "Other",
        "C_Agr_slam": "Slammed something",
        "C_Agr_throw_smt": "Threw object",
        "C_Agr_throw_twd": "Threw toward someone",
        "C_Agr_yelled": "Yelled"
    }

    # Label irritability group from ARI file
    ari_df = ari_df.rename(columns={"participant number": "participant_code", "score": "score"})
    ari_df["participant_code"] = ari_df["participant_code"].astype(str).str.strip()
    ari_df["group"] = pd.to_numeric(ari_df["score"], errors="coerce").apply(
        lambda x: "Irritable" if pd.notna(x) and x >= 3 else "Non-Irritable"
    )

    df["participant_code"] = df["participant_code_parent"].astype(str).str.strip()
    df = df.merge(ari_df[["participant_code", "group"]], on="participant_code", how="inner")

    # Load averaged survey data for aggression presence
    avg_df = pd.read_csv(AVG_PATH)
    avg_df["participant_code"] = avg_df["participant_code_parent"].astype(str).str.strip()
    avg_df = avg_df.merge(ari_df[["participant_code", "group"]], on="participant_code", how="inner")
    avg_df["C_Agr_none"] = pd.to_numeric(avg_df["C_Agr_none"], errors="coerce")

    # Calculate average aggression rate per group: 1 - C_Agr_none
    avg_df["aggression_rate"] = 1 - avg_df["C_Agr_none"]
    aggr_share = avg_df.groupby("group")["aggression_rate"].mean() * 100

    # Handle binary and text aggression columns
    df["C_Agr_other"] = df["C_Agr_other"].apply(lambda x: 1 if isinstance(x, str) and x.strip() != "" else 0)
    for col in AGGRESSION_COLS[1:]:
        df[col] = df[col].replace({"Yes": 1, "No": 0})

    # Filter only aggressive participants for the type pie charts
    participant_aggr = (
        df.groupby("participant_code")[AGGRESSION_COLS]
        .mean()
        .multiply(100)
        .reset_index()
        .merge(df[["participant_code", "group"]].drop_duplicates(), on="participant_code")
    )
    participant_aggr = participant_aggr[participant_aggr[AGGRESSION_COLS].sum(axis=1) > 0]

    grouped_means = participant_aggr.groupby("group")[AGGRESSION_COLS].mean()

    # Create 1x2 plot for aggression type pies
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for idx, group in enumerate(["Irritable", "Non-Irritable"]):
        if group in grouped_means.index:
            axes[idx].pie(
                grouped_means.loc[group],
                labels=[pie2_labels[c] for c in AGGRESSION_COLS],
                autopct="%1.1f%%",
                startangle=90,
                colors=[AGGRESSION_COLORS[c] for c in AGGRESSION_COLS]
            )
            n = (participant_aggr['group'] == group).sum()
            aggression_pct = aggr_share.get(group, 0.0)
            n_total = (avg_df['group'] == group).sum()
            axes[idx].set_title(f"{group} Group\n(N={n_total}, Aggressive={n}, Avg Agg: {aggression_pct:.1f}%)")

    plt.suptitle("Aggression Type Distribution by Irritability Group")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "aggression_pie_by_group.png"), dpi=300)
    plt.close()
    print("[INFO] Saved compact aggression pie chart to output/plots/aggression_pie_by_group.png")


def plot_means_by_irritability(df, ari_df):
    # Load averaged per-participant data
    avg_df = pd.read_csv(AVG_PATH)

    # Prepare ARI grouping
    ari_df = ari_df.rename(columns={"participant number": "participant_code", "score": "score"})
    ari_df["participant_code"] = ari_df["participant_code"].astype(str).str.strip()
    ari_df["group"] = pd.to_numeric(ari_df["score"], errors="coerce").apply(
        lambda x: "Irritable" if pd.notna(x) and x >= 3 else "Non-Irritable"
    )

    avg_df["participant_code"] = avg_df["participant_code_parent"].astype(str).str.strip()
    merged = avg_df.merge(ari_df[["participant_code", "group"]], on="participant_code", how="inner")

    def plot_grouped_means(gmeans, title, filename, group_counts):
        fig, ax = plt.subplots(figsize=(max(10, len(gmeans) * 0.4), 6))
        x = range(len(gmeans))
        ax.bar([i - 0.175 for i in x], gmeans.get("Irritable", [0]*len(x)), width=0.35,
               label=f"Irritable (N={group_counts.get('Irritable', 0)})", color=PALETTE[0])
        ax.bar([i + 0.175 for i in x], gmeans.get("Non-Irritable", [0]*len(x)), width=0.35,
               label=f"Non-Irritable (N={group_counts.get('Non-Irritable', 0)})", color=PALETTE[4])
        ax.set_xticks(x)
        ax.set_xticklabels(gmeans.index, rotation=45, ha="right")
        ax.set_ylabel("Mean Score")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        plt.close()
        print(f"[INFO] Saved plot to output/plots/{filename}")

    for role, label in zip(["child", "parent"], ["Child", "Parent"]):
        # Pick columns that belong to this role
        role_cols = [
        col for col in merged.columns
        if col in INCLUDED_QUESTIONS and col.startswith(role[0].upper() + "_")
    ]

        if not role_cols:
            print(f"[WARN] No valid columns for {label}")
            continue

        df_role = merged[["group"] + role_cols].copy()
        melted = df_role.melt(id_vars="group", var_name="Question", value_name="Score")
        grouped = melted.groupby(["Question", "group"])["Score"].mean().unstack()
        group_counts = df_role["group"].value_counts().to_dict()

        title = f"{label} Question Means by Irritability (from averaged data)"
        filename = f"question_means_{label.lower()}_from_average.png"
        plot_grouped_means(grouped, title, filename, group_counts)


def plot_questions_over_time(df, ari_df, question_labels):
    if isinstance(question_labels, str):
        question_labels = [question_labels]

    df["participant_code"] = df["participant_code_parent"].astype(str).str.strip()
    ari_df = ari_df.rename(columns={"participant number": "participant_code", "score": "score"})
    ari_df["participant_code"] = ari_df["participant_code"].astype(str).str.strip()
    ari_df["group"] = pd.to_numeric(ari_df["score"], errors="coerce").apply(
        lambda x: "Irritable" if pd.notna(x) and x >= 3 else "Non-Irritable"
    )

    merged = df.merge(ari_df[["participant_code", "group"]], on="participant_code", how="inner")

    # Check if all question labels exist
    missing = [q for q in question_labels if q not in merged.columns]
    if missing:
        raise ValueError(f"The following question labels are missing in the data: {missing}")

    merged["factor_score"] = merged[question_labels].select_dtypes(include='number').mean(axis=1, skipna=True)

    merged["timepoint"] = pd.Categorical(
        merged["day_child"].astype(str) + " " + merged["time_of_day_child"],
        categories=[f"{d} {t}" for d in range(1, 8) for t in ["AM", "PM"]],
        ordered=True
    )

    def plot_with_sem(data, title, fname):
        grouped = data.groupby(["timepoint", "group"])["factor_score"]
        means, sems = grouped.mean().unstack(), grouped.sem().unstack()
        group_counts = data.groupby("group")["participant_code"].nunique()

        plt.figure(figsize=(12, 6))
        for grp, clr in zip(["Irritable", "Non-Irritable"], [PALETTE[0], PALETTE[4]]):
            if grp in means:
                label = f"{grp} (N={group_counts.get(grp, 0)})"
                plt.plot(means.index, means[grp], label=label, marker='o', color=clr)
                plt.fill_between(means.index, means[grp] - sems[grp], means[grp] + sems[grp], color=clr, alpha=0.2)

        plt.title(title)
        plt.ylabel("Average Response")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[INFO] Saved plot to {path}")

    # Title and filename based on question(s)
    joined_labels = "_".join([q.lower() for q in question_labels])
    title = f"{' + '.join(question_labels)} Over Time by Irritability"
    filename = f"question_trend_{joined_labels}.png"

    plot_with_sem(merged, title, filename)

    # norm_df = merged.copy()
    # norm_df["factor_score"] = norm_df.groupby("participant_code")["factor_score"].transform(
    #     lambda x: (x - x.mean()) / x.std(ddof=0))
    # plot_with_sem(
    #     norm_df,
    #     f"{factor_name} Over Time (Normalized) by Irritability",
    #     f"factor_trend_{factor_name.replace(' ', '_').lower()}_normalized.png"
    # )


def plot_correlation_matrix(df, ari_df):
    df["participant_code"] = df["participant_code_parent"].astype(str).str.strip()
    ari_df = ari_df.rename(columns={"participant number": "participant_code", "score": "score"})
    ari_df["participant_code"] = ari_df["participant_code"].astype(str).str.strip()

    merged = df.merge(ari_df[["participant_code"]], on="participant_code", how="inner")

    def compute_factors(data, role):
        out = pd.DataFrame(index=data.index)
        for factor, mapping in FACTOR_QUESTIONS.items():
            cols = [col for col in mapping.get(role, []) if col in data.columns]
            if cols:
                out[f"{factor}_{role}"] = data[cols].select_dtypes(include='number').mean(axis=1, skipna=True)
        return out

    factors_child = compute_factors(merged, "child")
    factors_parent = compute_factors(merged, "parent")
    combined = pd.concat([factors_child, factors_parent], axis=1)
    corr = combined.corr()

    plt.figure(figsize=(14, 12))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            text = f"{corr.iloc[i, j]:.2f}" if not pd.isna(corr.iloc[i, j]) else ""
            plt.text(j, i, text, ha="center", va="center", fontsize=7, color="black")

    plt.title(f"Correlation Matrix: Child & Parent Factor Scores (N={merged['participant_code'].nunique()})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "factor_correlation_matrix.png"), dpi=300)
    plt.close()
    print("[INFO] Saved correlation matrix to output/plots/factor_correlation_matrix.png")


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    ari_df = pd.read_excel(ARI_PATH)
    plot_aggression(df, ari_df)
    plot_means_by_irritability(df, ari_df)
    plot_questions_over_time(df, ari_df, "C_Irr_Frustration")
    plot_correlation_matrix(df, ari_df)


if __name__ == "__main__":
    main()

