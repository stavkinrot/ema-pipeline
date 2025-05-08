# 📊 EMA Survey Data Pipeline
This project provides a robust and reproducible pipeline to preprocess, unify, and analyze ecological momentary assessment (EMA) survey data from parents and children. It handles multiple survey versions, time-based tracking, Hebrew-English question mapping, outlier detection, and survey synchronization.

## Project Structure
```bash
├── children_example.xlsx
├── ema_pipeline.py
├── generate_tree.py
├── pipleline_structure.txt
└── project_structure.txt
├── data/
│   ├── children_labeling.xlsx
│   ├── labeling.xlsx
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

Ensure you have Python 3.8+ and install the required packages:

```bash
pip install pandas numpy openpyxl
```

## 🚀 Running the Pipeline
1. Place all EMA survey .zip files into the folder:
```bash
data/input_zips/
```
2. Run the main script:
```bash
python ema_pipeline.py
```
  This will:

  * Extract survey files

  * Parse and align them into structured data

  * Apply outlier detection

  * Save clean output files to the output/ folder:
    * merged_surveys.csv (for convenient MLM analysis).
    * parents_surveys.xlsx (to detect outliers in parents).
    * children_surveys.xlsx (to detect outliers in children).
