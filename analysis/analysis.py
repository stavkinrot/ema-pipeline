import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import pearsonr

# === Config ===
DATA_PATH = "output/merged_surveys.csv"
OUTPUT_DIR = "output/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parallel question mapping
PARALLEL_ITEMS = {
    "Help Given": ("C_Inv_Help", "P_Inv_Help"),
    "Fun Time Together": ("C_Inv_Fun", "P_Inv_Fun"),
    "Talked About Day": ("C_Inv_Talk", "P_Inv_Talk"),
    "Positive Feedback": ("C_Positive", "P_Positive"),
    "Worried or Afraid": ("C_Anx_Worry", "P_Anx_Worry"),
    "Feeling Anxiety Now": ("C_Anx_now", "P_Anx_now"),
    "Anger Expression - Yelled": ("C_Agr_Yelled", "P_Agr_Yelled"),
    "Anger Expression - Slammed": ("C_Agr_slam", "P_Agr_slam"),
    "Anger Expression - Throw Something": ("C_Agr_throw_smt", "P_Agr_throw_smt"),
    "Anger Expression - Throw Toward": ("C_Agr_throw_twd", "P_Agr_throw_twd"),
    "Anger Expression - Hit": ("C_Agr_hit", "P_Agr_hit"),
    "No Anger Expression": ("C_Agr_none", "P_Agr_none"),
    "Angry When Things Didn't Go as Wanted": ("C_Agr_NotAsWant", "P_Agr_NotAsWant"),
    "Frustration": ("C_Irr_Frustration", "P_Irr_Frustration"),
    "Anger Right Now": ("C_Angry_now", "P_Angry_now"),
    "Annoyed Parent": ("C_PC_Annoy", "P_PC_Annoy"),
    "Criticism": ("C_PC_Criticism", "P_PC_Criticism"),
    "Sharing Emotions": ("C_PC_Sharing", "P_PC_Sharing"),
    "Feeling Sad or Depressed": ("C_Mood_Sad", "P_Mood_Sad"),
    "Feeling Good Now": ("C_Mood_Good", "P_Mood_Good"),
    "ADHD - Distracted": ("C_ADHD_Distracted", "P_ADHD_Distracted"),
    "ADHD - Restless": ("C_ADHD_Restless", "P_ADHD_Restless"),
    "Inhibitory Control - Spoke Without Thinking": ("C_IC_FirstOnMind", "P_IC_FirstOnMind"),
    "Inhibitory Control - Couldn't Stop": ("C_IC_CantStop", "P_IC_CantStop"),
    "Parent Got Angry": ("C_PS_GotAngry", "P_PS_GotAngry"),
    "Parent Was Patient": ("C_PS_Patient", "P_PS_Patient"),
    "Parent Always Agreed": ("C_PS_Agree", "P_PS_Agree"),
}


def analyze_parallel_correlations(df):
    results = []
    valid_plots = []

    for label, (child_col, parent_col) in PARALLEL_ITEMS.items():
        if child_col not in df.columns or parent_col not in df.columns:
            continue

        subset = df[[child_col, parent_col]].dropna()
        x = pd.to_numeric(subset[child_col], errors='coerce')
        y = pd.to_numeric(subset[parent_col], errors='coerce')
        x, y = x.dropna(), y.dropna()
        common_index = x.index.intersection(y.index)
        if len(common_index) < 3:
            print(f"[SKIP] Not enough valid responses for {label}")
            continue

        r, p = pearsonr(x.loc[common_index], y.loc[common_index])
        n = len(common_index)
        results.append({"Label": label, "r": round(r, 3), "n": n, "p-value": round(p, 4)})
        valid_plots.append((label, x.loc[common_index], y.loc[common_index], r, n))

    # Plotting
    cols = 5
    rows = math.ceil(len(valid_plots) / cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for i, (label, x_vals, y_vals, r, n) in enumerate(valid_plots):
        ax = axes[i]
        sns.regplot(x=x_vals, y=y_vals, ax=ax, scatter_kws={"alpha": 0.5})
        ax.set_title(f"{label}\nr={r:.2f}, n={n}", fontsize=9)
        ax.set_xlabel("Parent", fontsize=8)
        ax.set_ylabel("Child", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)

    for j in range(len(valid_plots), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "parallel_sync_grid.png"), dpi=300)
    plt.close()
    print("[INFO] Saved sync grid to output/plots/parallel_sync_grid.png")

    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(OUTPUT_DIR, "parallel_sync_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved summary to {summary_path}")


def analyze_dyadic_agreement(df):
    results = []
    used_pairs = [(c, p) for (c, p) in PARALLEL_ITEMS.values() if c in df.columns and p in df.columns]

    for code, group in df.groupby("participant_code_child"):
        child_responses = group[[c for c, _ in used_pairs]].mean(skipna=True, numeric_only=True)
        parent_responses = group[[p for _, p in used_pairs]].mean(skipna=True, numeric_only=True)

        if child_responses.empty or parent_responses.empty:
            continue

        common = child_responses.index.intersection(parent_responses.index)
        x = child_responses[common]
        y = parent_responses[common]

        if len(common) < 3:
            continue

        r, _ = pearsonr(x, y)
        results.append({
            "participant_code": code,
            "n_items": len(common),
            "r": round(r, 3)
        })

    df_result = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "dyadic_agreement.csv")
    df_result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved dyadic agreement results to {out_path}")

    # Histogram plot
    if "r" in df_result.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df_result["r"], bins=20, kde=True, color="steelblue")
        plt.axvline(0, color='gray', linestyle='--')
        plt.title("Distribution of Parent-Child Dyadic Agreement (r)")
        plt.xlabel("Pearson Correlation (r)")
        plt.ylabel("Number of Dyads")
        plt.tight_layout()
        hist_path = os.path.join(OUTPUT_DIR, "dyadic_agreement_hist.png")
        plt.savefig(hist_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved dyadic agreement histogram to {hist_path}")
    else:
        print("[WARNING] Column 'r' not found in dyadic agreement results")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    analyze_parallel_correlations(df)
    analyze_dyadic_agreement(df)
