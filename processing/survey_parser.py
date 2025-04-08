import os
import pandas as pd
import re
from datetime import datetime, timedelta

# Global mapping to store other free-text values
other_text_participants_by_day = {}

def is_example_column(column_name):
    return "אפשר גם להקליט תשובה" in column_name

def is_anger_other_column(column_name):
    keywords = ["אחר", "בצורה אחרת"]
    base_questions = [
        "מאז הסקר האחרון, הילד/ה שלי התפרץ/ה בכעס באחת או יותר מהדרכים הבאות",
        "מאז הסקר האחרון, התפרצתי בכעס באחת או יותר מהדרכים הבאות"
    ]
    return any(k in column_name for k in keywords) and any(b in column_name for b in base_questions)

def best_partial_match(column_name, question_map):
    for hebrew, english in question_map.items():
        if hebrew.strip() in column_name.strip():
            return english
    return None

def parse_survey_folder(folder_path, question_map, is_child):
    global other_text_participants_by_day

    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    code_file = [f for f in files if "קוד" in f][0]
    survey_files = [f for f in files if f != code_file]

    # --- Load participant code map with multi-header ---
    code_path = os.path.join(folder_path, code_file)

    # Read both 2nd and 4th rows as headers (Hebrew label + Participant ID)
    code_df = pd.read_csv(code_path, header=[1, 3])

    # Find the correct column names
    id_col = [col for col in code_df.columns if col[1] == "Participant ID"]
    code_col = [col for col in code_df.columns if "#" in str(col[0])]

    if not id_col or not code_col:
        raise ValueError("Could not find valid ID or Code columns in participant code file.")

    id_col = id_col[0]
    code_col = code_col[0]

    # Filter rows where code is valid (e.g., "#1234")
    valid_code_df = code_df[
        code_df[id_col].notna() &
        code_df[code_col].astype(str).str.match(r"#\d{4}$")
    ]

    # Build clean ID → code mapping
    id_to_code = dict(zip(valid_code_df[id_col], valid_code_df[code_col]))

    # === Process each morning/evening survey ===
    all_rows = []
    for file in survey_files:
        full_path = os.path.join(folder_path, file)
        df = pd.read_csv(full_path, header=3)
        print("[DEBUG] Survey file columns:", df.columns.tolist())

        time_of_day = "AM" if "בוקר" in file or "morning" in file.lower() else "PM"

        base_cols = ["Participant ID", "Day of Survey", "Start Date"]
        question_cols = [col for col in df.columns if col not in base_cols and not is_example_column(col)]
        df = df[base_cols + question_cols].copy()

        relabeled = {}
        other_free_text_cols = []

        for col in question_cols:
            if is_anger_other_column(col):
                relabeled[col] = "anger_other_freetext"
                other_free_text_cols.append(col)
            else:
                match = best_partial_match(col, question_map)
                if match:
                    relabeled[col] = match
                else:
                    relabeled[col] = col

        df.rename(columns=relabeled, inplace=True)
        df = df[[c for c in df.columns if c in base_cols or relabeled.get(c)]]

        df["participant_code"] = df["Participant ID"].map(id_to_code)
        df["time_of_day"] = time_of_day
        df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")

        for col in other_free_text_cols:
            df["anger_other_specified"] = df[col].notnull() & df[col].astype(str).str.strip().ne("")
            for i, row in df.iterrows():
                if row.get("anger_other_specified"):
                    key = (row["participant_code"], row["Start Date"].date(), row["time_of_day"])
                    other_text_participants_by_day[key] = row[col]

        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    skip_weekends = "בלי_שישי_שבת" in folder_path
    result_rows = []

    for participant, p_df in combined.groupby("participant_code"):
        p_df = p_df.sort_values("Start Date")
        if p_df["Start Date"].isnull().all():
            continue
        start_date = p_df["Start Date"].min()

        day_map = {}
        day_counter = 1
        date = start_date
        while day_counter <= 7:
            weekday = date.weekday()
            if skip_weekends and weekday in [4, 5]:
                date += timedelta(days=1)
                continue
            day_map[date.date()] = day_counter
            date += timedelta(days=1)
            day_counter += 1

        for _, row in p_df.iterrows():
            survey_date = row["Start Date"].date()
            if survey_date in day_map:
                row_dict = {
                    k: v for k, v in row.items()
                    if k not in ["Start Date", "Day of Survey", "Participant ID"]
                }
                row_dict["day"] = day_map[survey_date]
                result_rows.append(row_dict)

        for day in range(1, 8):
            for tod in ["AM", "PM"]:
                exists = any(
                    r.get("participant_code") == participant and
                    r.get("day") == day and
                    r.get("time_of_day") == tod
                    for r in result_rows if isinstance(r, dict)
                )
                if not exists:
                    result_rows.append({
                        "participant_code": participant,
                        "day": day,
                        "time_of_day": tod
                    })

    cleaned_rows = [r for r in result_rows if isinstance(r, dict)]
    print(f"[DEBUG] Total valid rows: {len(cleaned_rows)}")
    result_df = pd.DataFrame(cleaned_rows)

    base_cols = ["participant_code", "day", "time_of_day"]
    other_cols = [col for col in result_df.columns if col not in base_cols]
    result_df = result_df[base_cols + sorted(other_cols)]
    result_df = result_df.sort_values(["participant_code", "day", "time_of_day"])

    return result_df

def save_other_text_mapping(output_csv_path):
    records = []
    for (participant_code, day_or_date, time_of_day), text in other_text_participants_by_day.items():
        day = day_or_date
        if isinstance(day_or_date, datetime):
            day = day_or_date.strftime("%Y-%m-%d")
        records.append({
            "participant_code": participant_code,
            "day": day,
            "time_of_day": time_of_day,
            "text": text
        })
    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"[INFO] Saved other free-text responses to {output_csv_path}")
