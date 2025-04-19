import os
import pandas as pd

DEBUG = False

SHOTS = 1000

N_QUBITS = 5

MIN_DEPTH = 1
MAX_DEPTH = 10

TOTAL_THREADS = THREADS = 10

DATASET_SIZE = 1000
DATASET_PATH = os.path.join('.', 'dataset')
DATASET_FILE = "dataset.csv"
IMAGES_ARRAY_FILE = "images.h5"

def get_new_dim():
    return 1324, 2631

def debug(inp):
    if(not DEBUG):
        return
    print(inp)
