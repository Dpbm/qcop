"""Constant values"""

import os
import time


MODEL = "google/vit-base-patch16-384"

DEFAULT_DATASET_NAME="dqcop"
DEFAULT_MODEL_NAME="qcop"

DEFAULT_RANDOM_SEED = 32

DEFAULT_EPOCHS = 60

DEFAULT_SHOTS = 1000
DEFAULT_NUM_QUBITS = 5
DEFAULT_MAX_TOTAL_GATES = 20
DEFAULT_THREADS = 10
DEFAULT_AMOUNT_OF_CIRCUITS = 2000  # this one doesn't reflect exactly the size of the dataset, once the dataset might get either bigger, due to the different combinations of mesurements, or smaller due to duplicated circuits

SCALE_CIRCUIT_SIZE = 0.5

DATASET_FILE = "dataset.csv"
IMAGES_PATH = "images"

# ----- RANDOMNESS --------

DEFAULT_BATCH_SIZE = 10


DEFAULT_TARGET_FOLDER = "."

DEFAULT_NEW_DIM = (1353, 193)

DEFAULT_TRAIN_PERCENTAGE = 0.7
DEFAULT_TEST_PERCENTAGE = 0.2
# The remaining 0.1 is for Evaluation

DEFAULT_CHECKPOINT = None

MODEL_FILE_PREFIX = "model_"
CHECKPOINT_FILE_PREFIX = "checkpoint_"

DEFAULT_EARLY_STOP_PATIENCE=5
DEFAULT_EARLY_STOP_THRESHOLD=0.01
