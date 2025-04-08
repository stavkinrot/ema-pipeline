import os
import pandas as pd
from processing.extract import extract_zips
from processing.labeling import load_labeling_map
from processing.survey_parser import parse_survey_folder, save_other_text_mapping

# Define paths
input_zip_folder = "data/input_zips"
extracted_folder = "data/extracted"
labeling_path = "data/labeling.xlsx"
output_children_csv = "output/children_surveys.csv"
output_parents_csv = "output/parents_surveys.csv"
output_other_texts_csv = "output/other_text_responses.csv"

def main():
    # Step 1: Extract all zip files
    # extract_zips(input_zip_folder, extracted_folder)

    # Step 2: Load question label mappings
    parent_map, child_map = load_labeling_map(labeling_path)

    # Step 3: Process all extracted folders
    children_dfs = []
    parent_dfs = []

    for folder_name in os.listdir(extracted_folder):
        folder_path = os.path.join(extracted_folder, folder_name)

        if not os.path.isdir(folder_path):
            continue

        print(f"[INFO] Processing folder: {folder_name}")

        if "EMA_Parent" in folder_name:
            df = parse_survey_folder(folder_path, parent_map, is_child=False)
            parent_dfs.append(df)
        else:
            df = parse_survey_folder(folder_path, child_map, is_child=True)
            children_dfs.append(df)

    # Step 4: Concatenate and save
    if children_dfs:
        children_df = pd.concat(children_dfs, ignore_index=True)
        children_df.to_csv(output_children_csv, index=False)
        print(f"[INFO] Saved children data to {output_children_csv}")

    if parent_dfs:
        parents_df = pd.concat(parent_dfs, ignore_index=True)
        parents_df.to_csv(output_parents_csv, index=False)
        print(f"[INFO] Saved parent data to {output_parents_csv}")

    # Step 5: Save free-text 'other' responses
    save_other_text_mapping(output_other_texts_csv)

if __name__ == "__main__":
    main()