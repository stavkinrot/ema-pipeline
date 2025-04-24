# File: generate_tree.py

import os

EXCLUDE_DIRS = {'.git', '__pycache__', '.venv', 'venv', '.mypy_cache', '.pytest_cache'}

def generate_tree(root_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(root_path):
            rel_root = os.path.relpath(root, root_path)
            if any(part in EXCLUDE_DIRS for part in rel_root.split(os.sep)):
                continue

            level = rel_root.count(os.sep) if rel_root != '.' else 0
            indent = '│   ' * level + '├── '
            f.write(f"{indent}{os.path.basename(root)}/\n")

            sub_indent = '│   ' * (level + 1)
            visible_files = [f for f in files if not f.startswith('.')]
            for i, file in enumerate(visible_files):
                marker = '└── ' if i == len(visible_files) - 1 else '├── '
                f.write(f"{sub_indent}{marker}{file}\n")

if __name__ == "__main__":
    generate_tree(".", "project_structure.txt")
