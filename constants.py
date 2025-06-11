"""Constant values"""

import os

EPOCHS = 80

DEBUG = os.environ.get("DEBUG") or False

SHOTS = 1000

N_QUBITS = 5

MAX_TOTAL_GATES = 20

TOTAL_THREADS = THREADS = 10

BATCH_SIZE = 10

DATASET_SIZE = 20000
TARGET_FOLDER = os.environ.get("TARGET_FOLDER") or "."

DATASET_PATH = os.path.join(TARGET_FOLDER, 'dataset')
DATASET_FILE = os.path.join(TARGET_FOLDER, "dataset.csv")

IMAGES_H5_FILE = os.path.join(TARGET_FOLDER, "images.h5")

GHZ_FILE = os.path.join(TARGET_FOLDER, "ghz.pth")
GHZ_IMAGE_FILE = os.path.join(TARGET_FOLDER, "ghz.jpeg")
GHZ_PRED_FILE = os.path.join(TARGET_FOLDER,"ghz-prediction.pth")

HISTORY_FILE = os.path.join(TARGET_FOLDER, "history.json")
OUTPUT_PLOT_FILE = os.path.join(TARGET_FOLDER, "training_progress.png")

NEW_DIM = (500, 500)

TRAIN_PERCENTAGE = 0.7
TEST_PERCENTAGE = 0.2
EVAL_PERCENTAGE = 1 - (TRAIN_PERCENTAGE + TEST_PERCENTAGE)
