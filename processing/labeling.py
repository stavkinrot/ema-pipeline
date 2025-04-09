import pandas as pd

def load_labeling_map(labeling_path):
    parent_map = {}
    child_map = {}

    def load_sheet_map(path_or_df):
        if isinstance(path_or_df, str):
            df = pd.read_excel(path_or_df, engine="openpyxl")
        else:
            df = path_or_df

        mapping = {}
        for _, row in df.iterrows():
            hebrew_text = str(row.get("hebrew_text", "")).strip()
            hebrew_option = str(row.get("hebrew_option", "")).strip()
            label = str(row.get("label", "")).strip()
            if hebrew_text and label:
                key = (hebrew_text, hebrew_option if hebrew_option else None)
                mapping[key] = label
        return mapping

    xl = pd.ExcelFile(labeling_path, engine="openpyxl")
    for sheet_name in xl.sheet_names:
        sheet_df = xl.parse(sheet_name)
        if "parent" in sheet_name.lower():
            parent_map = load_sheet_map(sheet_df)
        elif "child" in sheet_name.lower():
            child_map = load_sheet_map(sheet_df)

    return parent_map, child_map
