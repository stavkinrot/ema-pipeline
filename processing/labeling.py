import pandas as pd

def load_labeling_map(labeling_path):
    """
    Loads the question labeling Excel and returns two dictionaries:
    - parent_map: Hebrew → English (parent surveys)
    - child_map: Hebrew → English (child surveys)
    """
    df = pd.read_excel(labeling_path, sheet_name=0)

    parent_map = dict(zip(df['EMA Item - Parent'].dropna(), df['Parent Label']))
    child_map = dict(zip(df['EMA Item - Child'].dropna(), df['Child Label']))

    return parent_map, child_map