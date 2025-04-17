import os
import pandas as pd
from tqdm import tqdm

DATASET_PATH = os.path.join(".", "dataset")
DATASET_FILE = "dataset.csv"


def main():
    csv_file = pd.read_csv(DATASET_FILE)
    duplicated_lines = csv_file.loc[csv_file.duplicated(subset="hash", keep="first")]

    indexes = duplicated_lines.index.to_list()
    files = [os.path.join(DATASET_PATH, file) for file in duplicated_lines["file"].to_list()]
    
    print("Dropping invalid rows")
    csv_file = csv_file.drop(indexes)
    print("Deleting duplicated files")
    for file in tqdm(files):
        os.remove(file)
    
    csv_file.to_csv(DATASET_FILE, index=False)

if __name__ == "__main__":
    main()
