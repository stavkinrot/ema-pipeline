import pandas as pd
import matplotlib.pyplot as plt
import os
from colors_config import PALETTE, AGGRESSION_COLORS
from factor_config import FACTOR_QUESTIONS

# === Shared Config ===
DATA_PATH = "output/merged_surveys.csv"
ARI_PATH = "data/ARI.xlsx"
OUTPUT_DIR = "output/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_aggression(df):
    AGGRESSION_COLS = ["C_Agr_other", "C_Agr_slam", "C_Agr_throw_smt", "C_Agr_throw_twd", "C_Agr_yelled"]
    pie2_labels = {
        "C_Agr_other": "Other",
        "C_Agr_slam": "Slammed something",
        "C_Agr_throw_smt": "Threw object",
        "C_Agr_throw_twd": "Threw toward someone",
        "C_Agr_yelled": "Yelled"
    }

    aggr_df = df[df["C_Agr_none"].isin(["Yes", "No"])]
    pie1_counts = aggr_df["C_Agr_none"].value_counts().reindex(["Yes", "No"], fill_value=0)
    pie1_labels = ["No Aggression", "Some Aggression"]
    pie1_values = [pie1_counts.get("Yes", 0), pie1_counts.get("No", 0)]
    pie1_colors = [AGGRESSION_COLORS["none"], AGGRESSION_COLORS["some"]]

    aggr_only_df = aggr_df[aggr_df["C_Agr_none"] == "No"]
    pie2_labels_ordered, pie2_values, pie2_colors = [], [], []
    for col in AGGRESSION_COLS:
        pie2_labels_ordered.append(pie2_labels[col])
        val = aggr_only_df[col].apply(lambda x: isinstance(x, str) and x.strip() != "").sum() if col == "C_Agr_other" else (aggr_only_df[col] == "Yes").sum()
        pie2_values.append(val)
        pie2_colors.append(AGGRESSION_COLORS[col])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].pie(pie1_values, labels=pie1_labels, autopct="%1.1f%%", startangle=90, colors=pie1_colors)
    axes[0].set_title("Aggression Presence in Surveys")
    axes[1].pie(pie2_values, labels=pie2_labels_ordered, autopct="%1.1f%%", startangle=90, colors=pie2_colors)
    axes[1].set_title("Types of Aggression (among 'Yes')")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "aggression_pie_charts.png"), dpi=300)
    plt.close()
    print("[INFO] Saved aggression plot to output/plots/aggression_pie_charts.png")


def plot_means_by_irritability(df, ari_df):
    ari_df = ari_df.rename(columns={"participant number": "participant_code", "score": "score"})
    ari_df["participant_code"] = ari_df["participant_code"].astype(str).str.strip()
    ari_df["group"] = pd.to_numeric(ari_df["score"], errors="coerce").apply(lambda x: "Irritable" if pd.notna(x) and x >= 3 else "Non-Irritable")

    df["participant_code"] = df["participant_code_parent"].astype(str).str.strip()
    merged = df.merge(ari_df[["participant_code", "group"]], on="participant_code", how="inner")

    def compute_factor_scores(data, role):
        out = pd.DataFrame(index=data.index)
        for factor, mapping in FACTOR_QUESTIONS.items():
            cols = [col for col in mapping.get(role, []) if col in data.columns]
            if cols:
                out[factor] = data[cols].select_dtypes(include='number').mean(axis=1, skipna=True)
        return out

    def plot_grouped_means(gmeans, title, filename):
        fig, ax = plt.subplots(figsize=(max(10, len(gmeans) * 0.5), 6))
        x = range(len(gmeans))
        ax.bar([i - 0.175 for i in x], gmeans.get("Irritable", [0]*len(x)), width=0.35, label="Irritable", color=PALETTE[0])
        ax.bar([i + 0.175 for i in x], gmeans.get("Non-Irritable", [0]*len(x)), width=0.35, label="Non-Irritable", color=PALETTE[3])
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
        factor_df = compute_factor_scores(merged, role=role)
        factor_df["group"] = merged["group"]
        grouped = factor_df.groupby("group").mean().T
        plot_grouped_means(grouped, f"{label} Factor Means by Irritability", f"factor_means_{label.lower()}.png")


def plot_factor_over_time(df, ari_df, factor_name):
    df["participant_code"] = df["participant_code_parent"].astype(str).str.strip()
    ari_df = ari_df.rename(columns={"participant number": "participant_code", "score": "score"})
    ari_df["participant_code"] = ari_df["participant_code"].astype(str).str.strip()
    ari_df["group"] = pd.to_numeric(ari_df["score"], errors="coerce").apply(lambda x: "Irritable" if pd.notna(x) and x >= 3 else "Non-Irritable")

    merged = df.merge(ari_df[["participant_code", "group"]], on="participant_code", how="inner")
    factor_items = [col for col in FACTOR_QUESTIONS.get(factor_name, {}).get("child", []) if col in merged.columns]
    merged["factor_score"] = merged[factor_items].select_dtypes(include='number').mean(axis=1, skipna=True)
    merged["timepoint"] = pd.Categorical(
        merged["day_child"].astype(str) + " " + merged["time_of_day_child"],
        categories=[f"{d} {t}" for d in range(1, 8) for t in ["AM", "PM"]],
        ordered=True)

    def plot_with_sem(data, title, fname):
        grouped = data.groupby(["timepoint", "group"])["factor_score"]
        means, sems = grouped.mean().unstack(), grouped.sem().unstack()
        plt.figure(figsize=(12, 6))
        for grp, clr in zip(["Irritable", "Non-Irritable"], [PALETTE[0], PALETTE[3]]):
            if grp in means:
                plt.plot(means.index, means[grp], label=grp, marker='o', color=clr)
                plt.fill_between(means.index, means[grp] - sems[grp], means[grp] + sems[grp], color=clr, alpha=0.2)
        plt.title(title)
        plt.ylabel("Average Factor Score")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, fname)
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"[INFO] Saved plot to {path}")

    plot_with_sem(
        merged,
        f"{factor_name} Over Time by Irritability",
        f"factor_trend_{factor_name.replace(' ', '_').lower()}_raw.png"
    )

    norm_df = merged.copy()
    norm_df["factor_score"] = norm_df.groupby("participant_code")["factor_score"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0))
    plot_with_sem(
        norm_df,
        f"{factor_name} Over Time (Normalized) by Irritability",
        f"factor_trend_{factor_name.replace(' ', '_').lower()}_normalized.png"
    )


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

    plt.title("Correlation Matrix: Child & Parent Factor Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "factor_correlation_matrix.png"), dpi=300)
    plt.close()
    print("[INFO] Saved correlation matrix to output/plots/factor_correlation_matrix.png")


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    ari_df = pd.read_excel(ARI_PATH)
    plot_aggression(df)
    plot_means_by_irritability(df, ari_df)
    plot_factor_over_time(df, ari_df, factor_name="Anxiety")
    plot_correlation_matrix(df, ari_df)


if __name__ == "__main__":
    main()

