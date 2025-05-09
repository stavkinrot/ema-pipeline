# 📊 EMA Survey Data Pipeline
This project provides a robust and reproducible pipeline to preprocess, unify, and analyze ecological momentary assessment (EMA) survey data from parents and children. It handles multiple survey versions, time-based tracking, Hebrew-English question mapping, outlier detection, and survey synchronization.

## 📁 Project Structure
Here’s what’s in the project folder:
```bash
├── children_example.xlsx
├── ema_pipeline.py
├── generate_tree.py
├── pipleline_structure.txt
└── project_structure.txt
├── data/
│   ├── children_labeling.xlsx
│   └── parent_labeling.xlsx
├── logs/
│   ├── unmatched_participants.log
│   └── unmatched_questions.log
├── output/
│   ├── children_surveys.csv
│   ├── children_surveys.xlsx
│   ├── merged_surveys.xlsx
│   ├── other_text_responses.csv
│   ├── parents_surveys.csv
│   └── parents_surveys.xlsx
├── processing/
│   ├── contradiction_config.py
│   ├── extract.py
│   ├── labeling.py
│   ├── outlier_detector.py
│   └── survey_parser.py
```
## ⚙️ Installation

Before running the project, make sure you have Python 3.8 or newer.

Then open a terminal or command prompt and run:

```bash
pip install pandas numpy openpyxl
```

## 🚀 Running the Pipeline
1. Put Your Survey Files In the Right Place:
   * Make a folder: data/input_zips/
   * Put all your EMA survey .zip files into that folder.
```bash
data/input_zips/
```
2. Run the Project
   In your terminal, run:
  ```bash
  python ema_pipeline.py
  ```
  That’s it! The program will:

  * Open and read the surveys

  * Combine and organize the answers

  * Check for suspicious answers (like always choosing the same number)

## 📦 What You’ll Get in the Output
After running the script, check the output/ folder. You’ll find:
- **children_surveys.xlsx** – Survey responses from children (includes outlier detection).
- **parents_surveys.xlsx** – Survey responses from parents (includes outlier detection).
- **merged_surveys.xlsx** – A combined file with both parent and child data, useful for analysis.
- **children_surveys.csv** / **parents_surveys.csv** – Same as above, but in CSV format.
- **other_text_responses.csv** – Free-text answers written in “Other” options (e.g., "Other behavior").
