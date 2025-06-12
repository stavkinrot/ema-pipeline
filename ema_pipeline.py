# --- ema_pipeline.py ---
import os
import pandas as pd
from processing.extract import extract_zips
from processing.labeling import load_labeling_map
from processing.survey_parser import parse_survey_folder, save_other_text_mapping, merge_surveys
from processing.outlier_detector import OutlierDetector
from processing.dyadic_sync import run_dyadic_sync
from processing.averager import run_averager



input_zip_folder = "data/input_zips"
extracted_folder = "data/extracted"
p_labeling_path = "data/parent_labeling.xlsx"
c_labeling_path = "data/children_labeling.xlsx"
output_children_csv = "output/children_surveys.csv"
output_parents_csv = "output/parents_surveys.csv"
output_other_texts_csv = "output/other_text_responses.csv"
output_merged_csv = "output/merged_surveys.csv"
output_children_xlsx = output_children_csv.replace(".csv", ".xlsx")
output_parents_xlsx = output_parents_csv.replace(".csv", ".xlsx")

def main():
    extract_zips(input_zip_folder, extracted_folder)  # Optional step

    parent_map = load_labeling_map(p_labeling_path)
    child_map = load_labeling_map(c_labeling_path)

    children_dfs = []
    parent_dfs = []

    for folder_name in os.listdir(extracted_folder):
        folder_path = os.path.join(extracted_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        print(f"[INFO] Processing folder: {folder_name}")
        if "EMA_Parent" in folder_name:
            df = parse_survey_folder(folder_path, parent_map)
            parent_dfs.append(df)
        else:
            df = parse_survey_folder(folder_path, child_map)
            children_dfs.append(df)

    # Handle children data
    if children_dfs:
        children_df = pd.concat(children_dfs, ignore_index=True)
        children_df.to_csv(output_children_csv, index=False, encoding="utf-8-sig")
        children_df.to_excel(output_children_xlsx, index=False)
        print(f"[INFO] Saved children data to {output_children_csv} and {output_children_xlsx}")

        detector = OutlierDetector(children_df)
        detector.detect_outliers()
        detector.highlight_in_excel(output_children_xlsx, output_children_xlsx)

    # Handle parent data
    if parent_dfs:
        parents_df = pd.concat(parent_dfs, ignore_index=True)
        parents_df.to_csv(output_parents_csv, index=False, encoding="utf-8-sig")
        parents_df.to_excel(output_parents_xlsx, index=False)
        print(f"[INFO] Saved parent data to {output_parents_csv} and {output_parents_xlsx}")

        detector = OutlierDetector(parents_df)
        detector.detect_outliers()
        detector.highlight_in_excel(output_parents_xlsx, output_parents_xlsx)
    
    if parent_dfs and children_dfs:
        merged_df = merge_surveys(children_df, parents_df)
        merged_df.to_csv(output_merged_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved merged wide-format survey to {output_merged_csv}")

    print("[INFO] Running post-processing: dyadic_sync")
    run_dyadic_sync()

    print("[INFO] Running post-processing: averager")
    run_averager()

    # Save free-text "Other" answers
    save_other_text_mapping(output_other_texts_csv)

if __name__ == "__main__":
    main()
