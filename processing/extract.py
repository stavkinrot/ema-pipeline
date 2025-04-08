import os
import zipfile

def extract_zips(input_folder, output_folder):
    """
    Extract each .zip file from `input_folder` into its own subfolder in `output_folder`.
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".zip"):
            zip_path = os.path.join(input_folder, filename)
            folder_name = os.path.splitext(filename)[0]
            extract_path = os.path.join(output_folder, folder_name)

            os.makedirs(extract_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            print(f"[EXTRACTED] {filename} to {extract_path}")