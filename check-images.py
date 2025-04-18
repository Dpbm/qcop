from multiprocessing import Pool
import hashlib
import os
from tqdm import tqdm

from constants import THREADS, DATASET_PATH


def check_image(image):
    with open(image, "rb") as file:
        file_data = file.read()
        hash_data = hashlib.md5(file_data).hexdigest()
        return hash_data

def main():
    files = os.listdir(DATASET_PATH)
    hashes = []
   
    for starting_point in tqdm(range(0, len(files), THREADS)):
        files_to_check = files[starting_point:starting_point+THREADS]
        files_to_check_with_path = (os.path.join(DATASET_PATH, file) for file in files_to_check)

        with Pool(processes=len(files_to_check)) as pool:
            results = pool.map(check_image, files_to_check_with_path)

        hashes = [*hashes, *results]

    print(f"Total hashes: {len(hashes)}")
    print(f"Total unique hahes: {len(set(hashes))}")
            






if __name__ == "__main__":
    main()
