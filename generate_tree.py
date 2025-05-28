# File: generate_tree.py

import os

EXCLUDE_DIRS = {'.git', '__pycache__', '.venv', 'venv', '.mypy_cache', '.pytest_cache'}
EXCLUDE_FILES = {'.DS_Store'}

def generate_tree(root_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        # Files in root
        root_files = [f for f in os.listdir(root_path)
                      if os.path.isfile(os.path.join(root_path, f)) and f not in EXCLUDE_FILES]
        for i, file in enumerate(sorted(root_files)):
            marker = '└── ' if i == len(root_files) - 1 else '├── '
            f.write(f"{marker}{file}\n")

        # First-level directories
        dirs = [d for d in os.listdir(root_path)
                if os.path.isdir(os.path.join(root_path, d)) and d not in EXCLUDE_DIRS]
        for d in sorted(dirs):
            f.write(f"├── {d}/\n")
            dir_path = os.path.join(root_path, d)
            sub_files = [f for f in os.listdir(dir_path)
                         if os.path.isfile(os.path.join(dir_path, f)) and f not in EXCLUDE_FILES]
            for i, file in enumerate(sorted(sub_files)):
                marker = '└── ' if i == len(sub_files) - 1 else '├── '
                f.write(f"│   {marker}{file}\n")

if __name__ == "__main__":
    generate_tree(".", "project_structure.txt")
