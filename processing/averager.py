import pandas as pd

def average_responses(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # Use child code as main identifier
    group_key = 'participant_code_child'
    parent_code_col = 'participant_code_parent'

    # Metadata and administrative columns to exclude
    exclude_cols = {
        'Participant ID_child', 'first_day_child', 'day_child', 'time_of_day_child', 'timestamp_child',
        'Participant ID_parent', 'first_day_parent', 'day_parent', 'time_of_day_parent', 'timestamp_parent',
        'merge_key', 'PC_Time_Gap'
    }

    # Exclude free-text fields
    free_text_cols = df.select_dtypes(include='object').columns
    long_text_cols = [
        col for col in free_text_cols
        if df[col].apply(lambda x: isinstance(x, str) and len(x.strip()) > 20).any()
    ]
    exclude_cols.update(long_text_cols)

    # Determine question columns to include
    question_cols = [col for col in df.columns if col not in exclude_cols and col != group_key and col != parent_code_col]

    # Convert Yes/No to 1/0 for averaging
    df[question_cols] = df[question_cols].replace({'Yes': 1, 'No': 0})

    # Include parent code for clarity by selecting the first seen per group
    parent_codes = df[[group_key, parent_code_col]].drop_duplicates(subset=[group_key])

    # Compute mean per child participant
    avg_df = df.groupby(group_key)[question_cols].mean().reset_index()
    avg_df = avg_df.round(3)

    # Merge back parent codes for reference
    avg_df = avg_df.merge(parent_codes, on=group_key, how='left')

    # Reorder columns to place parent_code after C_Positive
    col_list = avg_df.columns.tolist()
    if 'C_Positive' in col_list and parent_code_col in col_list:
        idx = col_list.index('C_Positive') + 1
        col_list.remove(parent_code_col)
        col_list.insert(idx, parent_code_col)
        avg_df = avg_df[col_list]

    # Save to CSV
    avg_df.to_csv(output_path, index=False)
    print(f"Saved averages to {output_path}")

# Example usage
if __name__ == "__main__":
    average_responses(
        input_path="output/merged_surveys.csv",
        output_path="output/child_parent_averages.csv"
    )
