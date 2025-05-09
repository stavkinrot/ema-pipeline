# 📊 EMA Survey Data Pipeline

This project provides a simple way to process, clean, and organize survey data collected from parents and children as part of an Ecological Momentary Assessment (EMA) study. It supports multiple versions of the survey (e.g., parent/child, morning/evening), Hebrew-English question mapping, time-based alignment, and automatic detection of suspicious answers.

---

> 🧭 **This guide is written for people who do NOT have VS Code or coding tools installed.**  
> You’ll use tools already built into your computer like Command Prompt or Terminal.

---

## 📁 Project Structure

After downloading and unzipping the project, you will see:

```bash
├── children_example.xlsx
├── ema_pipeline.py
├── generate_tree.py
├── pipleline_structure.txt
└── project_structure.txt
├── data/
│   ├── children_labeling.xlsx
│   └── parent_labeling.xlsx
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

---

## 🧰 Getting Started (No Coding or VS Code Needed)

### ✅ Step 1: Download the Project

1. Go to the GitHub page of this project.
2. Click the green **"Code"** button → **"Download ZIP"**.
3. Open your **Downloads** folder and unzip the file (right-click → **Extract All**).
4. Open the unzipped folder — you’ll see files like `ema_pipeline.py` and a `data/` folder.

---

### ✅ Step 2: Install Python

1. Visit: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Click **"Download Python"** (version 3.8 or newer).
3. Run the installer:
   - **Important**: Check the box that says **"Add Python to PATH"**
   - Then click **Install Now**

---

### ✅ Step 3: Install Required Packages

1. Open **Command Prompt** (Windows) or **Terminal** (Mac).
2. Type the following and press Enter:

```
pip install pandas numpy openpyxl
```

---

### ✅ Step 4: Add Your Survey Files

1. Open the unzipped project folder.
2. Go into the `data/` folder.
3. Create a **new folder** called:

```
input_zips
```

4. Put all your `.zip` survey files into that folder.

---

### ✅ Step 5: Run the Pipeline

1. Go back to your Command Prompt or Terminal.
2. Navigate to the folder where you unzipped the project. For example:

```
cd Desktop/ema-pipeline-main
```

> 💡 Tip: You can type `cd ` (with a space), then **drag the folder** from File Explorer (Windows) or Finder (Mac) into the terminal, and press Enter.

3. Run the script:

```
python ema_pipeline.py
```

---

## 📦 What You’ll Get in the Output

After the script finishes, open the `output/` folder inside the project. You will find:

- **children_surveys.xlsx** – Survey responses from children, with suspicious answers flagged.
- **parents_surveys.xlsx** – Survey responses from parents, with suspicious answers flagged.
- **merged_surveys.xlsx** – A combined file with both parent and child data, for easier analysis.
- **children_surveys.csv** / **parents_surveys.csv** – Same data in CSV format.
- **other_text_responses.csv** – Written responses where participants selected an "Other" option.

---

## 🆘 Having Trouble?

- Make sure you installed Python and checked **"Add Python to PATH"**
- Make sure you typed folder names and commands exactly as shown
- Double-check that your `.zip` survey files are inside `data/input_zips/`
- If needed, ask someone familiar with Python to walk you through the first run

---

## 💻 For VS Code Users

If you already use **Visual Studio Code**, you can:

1. Open the project folder in VS Code.
2. Open a terminal inside VS Code (from the top menu: **Terminal → New Terminal**).
3. Install the required packages if you haven’t already:

```
pip install pandas numpy openpyxl
```

4. Run the script with:

```
python ema_pipeline.py
```

This method gives you error messages and logs directly inside the VS Code interface.
