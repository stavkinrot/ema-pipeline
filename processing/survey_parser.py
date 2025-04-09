import os
import pandas as pd
import re
from datetime import datetime, timedelta
from .labeling import relabel_columns

other_text_participants_by_day = {}

def get_expected_dates(start_date, skip_weekends):
    expected_dates = []
    date = start_date
    while len(expected_dates) < 7:
        if skip_weekends and date.weekday() in [4, 5]:
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

def parse_survey_folder(folder_path, question_map, is_child):
    global other_text_participants_by_day

    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    code_file = [f for f in files if "קוד" in f][0]
    survey_files = [f for f in files if f != code_file]

    code_path = os.path.join(folder_path, code_file)
    code_df = pd.read_csv(code_path, header=[1, 3])

    id_col = [col for col in code_df.columns if col[1] == "Participant ID"]
    code_col = [col for col in code_df.columns if "#" in str(col[0])]
    date_col = [col for col in code_df.columns if col[1] == "Start Date"]

    id_col = id_col[0]
    code_col = code_col[0]
    date_col = date_col[0]

    valid_code_df = code_df[
        code_df[id_col].notna() &
        code_df[code_col].astype(str).str.match(r"#\d{4}$")
    ]

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
    if os.path.exists("unmatched_questions.log"):
        os.remove("unmatched_questions.log")

    for file in survey_files:
        full_path = os.path.join(folder_path, file)
        headers_df = pd.read_csv(full_path, nrows=4, header=None)
        question_texts = headers_df.iloc[1]
        question_types = headers_df.iloc[2]
        subquestion_texts = headers_df.iloc[3]

        column_map, unmatched = relabel_columns(question_texts, question_types, subquestion_texts, question_map)
        print(f"[DEBUG] Mapped {len(column_map)} columns in file: {file}")

        if unmatched:
            with open("unmatched_questions.log", "a", encoding="utf-8") as log:
                log.write(f"\n--- Unmatched in {file} ---\n")
                for qtext, opt in unmatched:
                    log.write(f"Unmatched: {qtext} → {opt}\n")

        # Load the actual response data
        response_df = pd.read_csv(full_path, skiprows=3)

        time_of_day = "AM" if "בוקר" in file or "morning" in file.lower() else "PM"

        # Build DataFrame with only relevant columns
        df = pd.DataFrame()
        df["Participant ID"] = response_df["Participant ID"]
        df["Start Date"] = pd.to_datetime(response_df["Start Date"], errors="coerce")
        df["participant_code"] = df["Participant ID"].map(id_to_code)
        df["time_of_day"] = time_of_day

        for idx, label in column_map.items():
            column_name = response_df.columns[idx]
            df[label] = response_df[column_name] if column_name in response_df else ""

        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    result_rows = []

    for participant, p_df in combined.groupby("participant_code"):
        p_df = p_df.sort_values("Start Date")
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
                row_dict = {k: v for k, v in row.items() if k not in ["Start Date"]}
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
                        "time_of_day": tod
                    })

    cleaned_rows = [r for r in result_rows if isinstance(r, dict)]
    result_df = pd.DataFrame(cleaned_rows)

    base_cols = ["Participant ID", "participant_code", "first_day", "day", "time_of_day"]
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