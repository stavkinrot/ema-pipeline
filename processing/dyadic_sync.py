import pandas as pd
import numpy as np

def compute_dyadic_sync(merged_df: pd.DataFrame, ari_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    # Work on a copy to avoid modifying input
    merged_df = merged_df.copy()

    # Prepare ARI group info
    ari_df = ari_df.rename(columns={"participant number": "participant_code", "score": "score"})
    ari_df["participant_code"] = ari_df["participant_code"].astype(str).str.strip()
    ari_df["group"] = pd.to_numeric(ari_df["score"], errors="coerce").apply(
        lambda x: "Irritable" if pd.notna(x) and x >= 3 else "Non-Irritable"
    )

    # Convert C_Agr_None and P_Agr_None from Yes/No to numeric
    yn_map = {"Yes": 1, "No": 0}
    for col in ["C_Agr_none", "P_Agr_none"]:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].map(yn_map)

    # Identify numeric column pairs where both C_ and P_ exist
    numeric_cols = merged_df.select_dtypes(include='number').columns
    question_pairs = [
        (c, c.replace("C_", "P_"))
        for c in numeric_cols if c.startswith("C_") and c.replace("C_", "P_") in numeric_cols
    ]

    sync_records = []

    for child_code in merged_df["participant_code_child"].dropna().unique():
        df_child = merged_df[merged_df["participant_code_child"] == child_code]
        parent_code = df_child["participant_code_parent"].iloc[0]
        group_row = ari_df.loc[ari_df["participant_code"] == str(parent_code), "group"]
        if group_row.empty:
            continue
        group = group_row.values[0]

        for c_q, p_q in question_pairs:
            paired = df_child[[c_q, p_q]].dropna()
            if len(paired) < 2:
                continue

            # Check variance before calculating Pearson/Spearman
            if paired[c_q].nunique() < 2 or paired[p_q].nunique() < 2:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = np.corrcoef(paired[c_q], paired[p_q])[0, 1]
                spearman = paired.corr(method='spearman').iloc[0, 1]

            mad = np.mean(np.abs(paired[c_q] - paired[p_q]))

            sync_records.append({
                "participant_code_child": child_code,
                "participant_code_parent": parent_code,
                "group": group,
                "question": c_q.replace("C_", ""),
                "pearson": pearson,
                "spearman": spearman,
                "mad": mad
            })

    # Save and return
    sync_df = pd.DataFrame(sync_records)
    sync_df.to_csv(output_path, index=False)
    print(f"Saved dyadic sync to {output_path}")
    return sync_df

# Example usage
if __name__ == "__main__":
    merged = pd.read_csv("output/merged_surveys.csv")
    ari = pd.read_excel("data/ARI.xlsx")
    compute_dyadic_sync(merged, ari, output_path="output/sync_df.csv")
