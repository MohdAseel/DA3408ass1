import zipfile
import csv
import os

def create_csv_from_zip(zip_path, csv_path):
    if not os.path.exists(zip_path):
        return
        
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_names = [f for f in z.namelist() if not f.endswith('/')]
        # Extract just the filenames without paths
        file_names = [os.path.basename(f) for f in file_names if os.path.basename(f)]
        
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename']) # Header line
        for name in file_names:
            writer.writerow([name])
            
    print(f"Created {csv_path} with {len(file_names)} data row")

if __name__ == "__main__":
    create_csv_from_zip('data.zip', 'data.csv')
