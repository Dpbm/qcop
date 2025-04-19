import os
import pandas as pd

SHOTS = 1000

N_QUBITS = 5

MIN_DEPTH = 1
MAX_DEPTH = 10

TOTAL_THREADS = THREADS = 10

DATASET_SIZE = 1000
DATASET_PATH = os.path.join('.', 'dataset')
DATASET_FILE = "dataset.csv"
NPY_IMAGES_FILE = "images.npy"

def get_new_dim():
    df = pd.read_csv(DATASET_FILE)
    return  df["img_width"].max(), df["img_height"].max()
