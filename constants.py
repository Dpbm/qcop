import os

EPOCHS = 60

DEBUG = os.environ.get("DEBUG") or False

SHOTS = 1000

N_QUBITS = 5

MIN_DEPTH = 1
MAX_DEPTH = 10

TOTAL_THREADS = THREADS = 10

DATASET_SIZE = 5000
DATASET_PATH = os.path.join('.', 'dataset')
DATASET_FILE = "dataset.csv"
IMAGES_TRAIN = "images_train.h5"
IMAGES_TEST = "images_test.h5"

new_dim = (600, 1200)

