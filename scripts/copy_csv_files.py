from pathlib import Path
import os
import shutil

# Define source and destination paths
parent = Path(__file__).parent.resolve().parent.resolve()
outputs_folder =  parent / 'outputs'
if not outputs_folder.exists():
    raise FileNotFoundError(f"Outputs folder does not exist: {outputs_folder}")

cities_output_text_folder = outputs_folder / 'cities_output_text'
input_file = os.path.join(parent, 'input', 'gridlist_with_idx_cities.csv')
pls_table = Path(os.path.join(parent, 'src', 'PLS_MAIN', 'pls_attrs-9999.csv'))

print(f"Outputs folder: {outputs_folder}")
print(f"Cities output text folder: {cities_output_text_folder}")
print(f"Input file: {input_file}")
print(f"PLS table: {pls_table}")


# Create the destination folder if it doesn't exist
os.makedirs(cities_output_text_folder, exist_ok=True)

# Function to copy CSV files while maintaining folder structure
def copy_csv_files(src_folder, dest_folder):
    for root, _, files in os.walk(src_folder):
        # Skip the destination folder and its subfolders to prevent recursive copying
        if os.path.commonpath([os.path.abspath(root), os.path.abspath(dest_folder)]) == os.path.abspath(dest_folder):
            continue

        for file in files:
            if file.endswith('.csv'):
                # Recreate the folder structure in the destination
                relative_path = os.path.relpath(root, src_folder)
                dest_path = os.path.join(dest_folder, relative_path)
                os.makedirs(dest_path, exist_ok=True)

                # Copy the file
                shutil.copy2(os.path.join(root, file), dest_path)

# Copy all CSV files from outputs folder to cities_output_text folder
copy_csv_files(outputs_folder, cities_output_text_folder)

# Copy the specific file from input folder to cities_output_text folder
shutil.copy2(input_file, cities_output_text_folder)

# Copy the specific file from PLS_MAIN folder to cities_output_text folder
shutil.copy2(pls_table, cities_output_text_folder)

print(f"All CSV files have been copied to {cities_output_text_folder}")

shutil.make_archive(str(cities_output_text_folder), 'zip', root_dir=str(cities_output_text_folder))
# Move files to /Documents/TMP folder
tmp_folder = Path.home() / 'Onedrive' / 'Documentos' / 'TMP'
if not tmp_folder.exists():
    os.makedirs(tmp_folder, exist_ok=True)
shutil.move(str(cities_output_text_folder) + '.zip', tmp_folder)
print(f"All CSV files have been zipped and moved to {tmp_folder}")