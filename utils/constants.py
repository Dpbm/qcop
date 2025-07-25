"""Constant values"""

import os

DEBUG = os.environ.get("DEBUG") or False

DEFAULT_EPOCHS = 100

DEFAULT_SHOTS = 1000
DEFAULT_NUM_QUBITS = 5
DEFAULT_MAX_TOTAL_GATES = 20

DEFAULT_THREADS = 10

DEFAULT_BATCH_SIZE = 10

DEFAULT_DATASET_SIZE = 20000 # this one doesn't reflect exactly the size of the dataset, once the dataset might get either bigger, due to the different combinations of mesurements, or smaller due to duplicated circuits

DEFAULT_TARGET_FOLDER = "."

DEFAULT_NEW_DIM = (500, 500)

DEFAULT_TRAIN_PERCENTAGE = 0.7
DEFAULT_TEST_PERCENTAGE = 0.2

DEFAULT_CHECKPOINT = None

MODEL_FILE_PREFIX = "model_"
CHECKPOINT_FILE_PREFIX = "checkpoint_"

# ruff: noqa: E731
dataset_path = lambda target_folder: os.path.join(target_folder, "dataset")
dataset_file = lambda target_folder: os.path.join(target_folder, "dataset.csv")
images_h5_file = lambda target_folder: os.path.join(target_folder, "images.h5")
ghz_file = lambda target_folder: os.path.join(target_folder, "ghz.pth")
ghz_image_file = lambda target_folder: os.path.join(target_folder, "ghz.jpeg")
ghz_pred_file = lambda target_folder: os.path.join(target_folder, "ghz-prediction.pth")
ghz_image_file = lambda target_folder: os.path.join(target_folder, "ghz.jpeg")
history_file = lambda target_folder: os.path.join(target_folder, "history.json")
output_plot_file = lambda target_folder: os.path.join(
    target_folder, "training_progress.png"
)
images_gen_checkpoint_file = lambda target_folder: os.path.join(
    target_folder, "gen_checkpoint.json"
)
