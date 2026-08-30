import zipfile
import csv
import os

def append_to_csv(zip_path, csv_path):
    if not os.path.exists(zip_path):
        return
        
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_names = [f for f in z.namelist() if not f.endswith('/')]
        file_names = [os.path.basename(f) for f in file_names if os.path.basename(f)]
        
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for name in file_names:
            writer.writerow([name])
            
    print(f"Appended {len(file_names)} new data rows from {zip_path} to {csv_path}.")

if __name__ == "__main__":
    append_to_csv('new_labels.zip', 'data.csv')
