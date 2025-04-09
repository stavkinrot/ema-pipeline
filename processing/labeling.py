import pandas as pd
import unicodedata
import re

def clean(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u200f", "")           # RTL
    text = text.replace("\xa0", " ")            # non-breaking space
    text = text.replace("\u202c", "").replace("\u202a", "")  # directional
    text = re.sub(r"^[^\wא-ת]+", "", text)      # remove leading punctuation like colons
    text = re.sub(r"[\[\]\"\'”“׳׳]", "", text)  # brackets, quotes
    text = re.sub(r"\s+", " ", text)            # collapse multiple spaces
    text = text.strip()

    # Normalize unbalanced parentheses
    if text.count("(") > text.count(")"):
        text += ")"
    elif text.count(")") > text.count("("):
        text = "(" + text

    return text

def load_labeling_map(labeling_path):
    df = pd.read_excel(labeling_path, engine="openpyxl")
    mapping = {}

    for _, row in df.iterrows():
        raw_text = row.get("hebrew_text", "")
        raw_option = row.get("hebrew_option", "")
        hebrew_text = clean(raw_text)
        hebrew_option = clean(raw_option)
        label = str(row.get("label", "")).strip()

        if hebrew_option == "אחר (בבקשה לפרט)":
            print(f"[LABEL] loaded option = '{hebrew_option}' (len={len(hebrew_option)})")

        if hebrew_text and label:
            key = (hebrew_text, hebrew_option if hebrew_option else None)
            mapping[key] = label

    return mapping

def relabel_columns(question_texts, question_types, subquestion_texts, question_map):
    mapping = {}
    unmatched = []

    for i in range(len(question_texts)):
        q_text = clean(question_texts[i])
        q_type = clean(question_types[i])
        sub_text = clean(subquestion_texts[i])

        if sub_text == "אחר (בבקשה לפרט)":
            print(f"[RELAB] matched option = '{sub_text}' (len={len(sub_text)})")
        
        if "התפרצתי בכעס" in q_text:
            print(f"[CHILD DEBUG] q_text: '{q_text}', sub_text: '{sub_text}'")

        if not q_text or "Start Date" in q_text or "Location" in q_text:
            continue

        if "שיתפתי את" in q_text or "עזר לי" in q_text or "קשה להפסיק" in q_text:
            print(f"[CHILD TRACK] q_text: '{q_text}' -> type: '{q_type}'")


        key = (q_text, sub_text) if q_type == "Multiple selection" else (q_text, None)
        label = question_map.get(key)

        if label:
            mapping[i] = label
        else:
            if key[0] == "מאז הסקר האחרון, הילד/ה שלי התפרץ/ה בכעס באחת או יותר מהדרכים הבאות":
                with open("debug_subtext.log", "a", encoding="utf-8") as f:
                    f.write("=== Raw Char Comparison ===\n")
                    for i, (a, b) in enumerate(zip(sub_text, "אחר (בבקשה לפרט)")):
                        f.write(f"pos {i}: actual '{a}' ({ord(a)}) vs expected '{b}' ({ord(b)})\n")
                    f.write("\n")
            unmatched.append(key)

    return mapping, unmatched
