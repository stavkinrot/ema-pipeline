import os
import pandas as pd
import re
from datetime import datetime, timedelta
from processing.labeling import relabel_columns
import logging

# --- Logging setup ---
os.makedirs("logs", exist_ok=True)

unmatched_participants_logger = logging.getLogger("unmatched_participants")
unmatched_participants_logger.setLevel(logging.WARNING)
up_handler = logging.FileHandler("logs/unmatched_participants.log", mode="w", encoding="utf-8")
up_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
unmatched_participants_logger.addHandler(up_handler)
unmatched_participants_logger.propagate = False

unmatched_questions_logger = logging.getLogger("unmatched_questions")
unmatched_questions_logger.setLevel(logging.WARNING)
uq_handler = logging.FileHandler("logs/unmatched_questions.log", mode="w", encoding="utf-8")
uq_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
unmatched_questions_logger.addHandler(uq_handler)
unmatched_questions_logger.propagate = False

# Global list to store other free-text values
other_text_participants_by_day = []

def is_example_column(column_name):
    return "אפשר גם להקליט תשובה" in column_name

def best_partial_match(column_name, question_map):
    for hebrew, english in question_map.items():
        if hebrew.strip() in column_name.strip():
            return english
    return None

def get_expected_dates(start_date, skip_weekends):
    expected_dates = []
    date = start_date
    while len(expected_dates) < 7:
        if skip_weekends and date.weekday() in [4, 5]:  # Friday=4, Saturday=5
            date += timedelta(days=1)
            continue
        expected_dates.append(date)
        date += timedelta(days=1)
    return expected_dates

def get_first_experiment_day(raw_start_date, skip_weekends):
    date = raw_start_date + timedelta(days=1)
    while skip_weekends and date.weekday() in [4, 5]:
        date += timedelta(days=1)
    return date

def parse_survey_folder(folder_path, question_map):
    global other_text_participants_by_day

    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    code_file = [f for f in files if "קוד" in f][0]
    survey_files = [f for f in files if f != code_file]

    code_path = os.path.join(folder_path, code_file)
    code_df = pd.read_csv(code_path, header=[1, 3], encoding="utf-8-sig")

    id_col = [col for col in code_df.columns if col[1] == "Participant ID"]
    code_col = [col for col in code_df.columns if "#" in str(col[0])]
    date_col = [col for col in code_df.columns if col[1] == "Start Date"]

    if not id_col or not code_col or not date_col:
        raise ValueError("Could not find valid ID, Code, or Start Date columns in participant code file.")

    id_col = id_col[0]
    code_col = code_col[0]
    date_col = date_col[0]

    raw_code_series = code_df[code_col].astype(str).str.strip()
    code_df["_raw_code_column_for_logging"] = raw_code_series

    cleaned_codes = []
    for idx, original in raw_code_series.items():
        match = re.search(r"\d{4}", original)
        cleaned = f"#{match.group()}" if match else ""

        if cleaned and not re.fullmatch(r"#\d{4}", original.strip()):
            unmatched_participants_logger.warning(
                f"[FIXED CODE] Row {idx}: original='{original}' -> cleaned='{cleaned}'"
            )

        cleaned_codes.append(cleaned)

    code_df[code_col] = cleaned_codes

    valid_code_df = code_df[
        code_df[id_col].notna() & 
        code_df[code_col].str.match(r"#\d{4}")
    ]

    raw_codes = code_df[code_col].astype(str)

    unmatched_participants = code_df[
        code_df[id_col].isna() |
        raw_codes.isna() |
        ~raw_codes.str.fullmatch(r"#\d{4}")
    ]

    unmatched_participants_logger.warning(f"--- Unmatched participants in folder: {folder_path} ---")
    for _, row in unmatched_participants.iterrows():
        pid = str(row.get(id_col, "")).strip()
        code = str(row.get("_raw_code_column_for_logging", "")).strip()

        reason = []
        if not pid or pd.isna(pid):
            reason.append("missing ID")
        if not code or not re.fullmatch(r"#\d{4}", code):
            reason.append("invalid code")

        unmatched_participants_logger.warning(
            f"ID: {pid}, Code: {code}, Reason: {', '.join(reason)}"
        )

    id_to_code = dict(zip(valid_code_df[id_col], valid_code_df[code_col]))
    code_to_id = {v: k for k, v in id_to_code.items()}

    skip_weekends = "בלי_שישי_שבת" in folder_path

    code_to_start_day = {
        row[code_col]: get_first_experiment_day(
            pd.to_datetime(row[date_col], errors="coerce").date(),
            skip_weekends
        )
        for _, row in valid_code_df.iterrows()
    }

    all_rows = []
    for file in survey_files:
        full_path = os.path.join(folder_path, file)
        headers_df = pd.read_csv(full_path, nrows=4, header=None, encoding="utf-8-sig")
        question_texts = headers_df.iloc[1]
        question_types = headers_df.iloc[2]
        subquestion_texts = headers_df.iloc[3]

        column_map, unmatched = relabel_columns(question_texts, question_types, subquestion_texts, question_map)

        if unmatched:
            unmatched_questions_logger.warning(f"--- Unmatched questions in file: {file} ---")
            for qtext, opt in unmatched:
                unmatched_questions_logger.warning(f"Unmatched: '{qtext}' -> Option: '{opt}'")

        response_df = pd.read_csv(full_path, skiprows=3, encoding="utf-8-sig")

        time_of_day = "AM" if "בוקר" in file or "morning" in file.lower() else "PM"

        df = pd.DataFrame()
        df["Participant ID"] = response_df["Participant ID"]
        df["Start Date"] = pd.to_datetime(response_df["Start Date"], errors="coerce", format="%m/%d/%Y %I:%M%p")
        df["End Date"] = pd.to_datetime(response_df["End Date"], errors="coerce", format="%m/%d/%Y %I:%M%p")
        df["timestamp"] = df[["Start Date", "End Date"]].mean(axis=1, numeric_only=False)
        df["participant_code"] = df["Participant ID"].map(id_to_code)
        df["time_of_day"] = time_of_day

        for idx, label in column_map.items():
            column_name = response_df.columns[idx]
            df[label] = response_df[column_name] if column_name in response_df else ""

        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    result_rows = []

    for participant, p_df in combined.groupby("participant_code"):
        p_df = p_df.sort_values("timestamp")
        if participant not in code_to_start_day:
            continue

        start_date = code_to_start_day[participant]
        expected_dates = get_expected_dates(start_date, skip_weekends)
        expected_timepoints = {(d, tod): False for d in expected_dates for tod in ["AM", "PM"]}

        for _, row in p_df.iterrows():
            survey_date = row["Start Date"].date()
            tod = row["time_of_day"]
            if skip_weekends and survey_date.weekday() in [4, 5]:
                continue
            if survey_date in expected_dates:
                expected_timepoints[(survey_date, tod)] = True
                row_dict = {
                    k: v for k, v in row.items()
                    if k not in ["Start Date", "End Date"]
                }
                row_dict["day"] = expected_dates.index(survey_date) + 1
                row_dict["first_day"] = start_date
                result_rows.append(row_dict)

        for i, date in enumerate(expected_dates):
            for tod in ["AM", "PM"]:
                if not expected_timepoints[(date, tod)]:
                    participant_id = code_to_id.get(participant, None)
                    result_rows.append({
                        "Participant ID": participant_id,
                        "participant_code": participant,
                        "first_day": start_date,
                        "day": i + 1,
                        "time_of_day": tod,
                        "timestamp": pd.NaT
                    })

    result_df = pd.DataFrame(result_rows)

    metadata_cols = {"Participant ID", "participant_code", "first_day", "day", "time_of_day", "timestamp"}

    for colname, label in [("C_Agr_other", "Children"), ("P_Agr_other", "Parent")]:
        if colname in result_df.columns:
            result_df[colname] = result_df[colname].apply(lambda x: "" if pd.isna(x) else str(x).strip())

            for _, row in result_df.iterrows():
                text_value = row[colname]
                if text_value:
                    other_text_participants_by_day.append({
                        "Participant ID": row["Participant ID"],
                        "participant_code": row["participant_code"],
                        "Parent/Children": label,
                        "Day of experiment": row["day"],
                        "text": text_value
                    })

    base_cols = ["Participant ID", "participant_code", "first_day", "day", "time_of_day", "timestamp"]
    other_cols = [col for col in result_df.columns if col not in base_cols]
    result_df = result_df[base_cols + sorted(other_cols)]
    result_df = result_df.sort_values(["participant_code", "day", "time_of_day"])

    return result_df

def save_other_text_mapping(output_csv_path):
    records = []
    for record in other_text_participants_by_day:
        text = record.get("text")
        if isinstance(text, float) and pd.isna(text):
            continue
        if not text or str(text).strip() == "":
            continue
        records.append({
            "Participant ID": record["Participant ID"],
            "participant_code": record["participant_code"],
            "Parent/Children": record["Parent/Children"],
            "Day of experiment": record["Day of experiment"],
            "text": text
        })

    if records:
        df = pd.DataFrame(records)
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] Saved {len(df)} valid free-text responses to {output_csv_path}")
    else:
        print("[INFO] No valid free-text 'Other' responses found.")

def merge_surveys(children_df: pd.DataFrame, parents_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges children and parent survey responses into a single wide-format DataFrame.
    Each row represents a matched survey from a child and their corresponding parent
    (same day, time of day, and matching participant codes).

    Returns:
        A DataFrame with suffixes `_child` and `_parent` for each side's responses.
    """
    children_df["merge_key"] = children_df.apply(
        lambda row: (row["participant_code"], row["day"], row["time_of_day"]), axis=1
    )

    parents_df["merge_key"] = parents_df.apply(
        lambda row: (f"#{int(row['participant_code'][1:]) + 1:04}", row["day"], row["time_of_day"]), axis=1
    )

    merged_df = pd.merge(
        children_df,
        parents_df,
        on="merge_key",
        suffixes=("_child", "_parent")
    )

    return merged_df
