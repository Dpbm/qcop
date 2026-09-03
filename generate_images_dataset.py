""""Generate Dataset Images"""

import argparse
import os
import sys

from generate.images import Images,DF,Checkpoint
from utils.datatypes import *
from utils.constants import (
    DEFAULT_SHOTS,
    DEFAULT_NUM_QUBITS,
    DEFAULT_MAX_TOTAL_GATES,
    DEFAULT_THREADS,
    DEFAULT_AMOUNT_OF_CIRCUITS,
    DEFAULT_DATASET_NAME,
    DATASET_FILE,
    IMAGES_PATH
)

IMAGES_CHECKPOINT_FILE = "images_checkpoint.json"

def update_rows_callback(rows:DFRows, checkpoint:Checkpoint, df:DF, inc: int):
    df.append_rows_to_file(rows)
    checkpoint.index += inc
    checkpoint.save()

def main(args:argparse.Namespace):
    images_data_path = os.path.join(args.target_folder, IMAGES_PATH)

    checkpoint = Checkpoint.get_checkpoint(os.path.join(args.target_folder, IMAGES_CHECKPOINT_FILE))
    img_handler = Images(images_data_path)
    df = DF(os.path.join(args.target_folder, DATASET_FILE))

    os.makedirs(images_data_path, exist_ok=True)

    img_handler.generate_images(
                args.n_qubits, 
                args.amount_circuits, 
                args.max_gates,
                args.shots,
                lambda rows,inc: update_rows_callback(rows, checkpoint, df, inc),
                args.threads,
                checkpoint,
                circuit_format_counter=checkpoint.index
    )

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
        parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
        parser.add_argument("--n-qubits", type=int, default=DEFAULT_NUM_QUBITS)
        parser.add_argument("--max-gates", type=int, default=DEFAULT_MAX_TOTAL_GATES)
        parser.add_argument(
            "--amount-circuits", type=int, default=DEFAULT_AMOUNT_OF_CIRCUITS
        )
        parser.add_argument("--target-folder", type=FilePath, required=True)

        if len(sys.argv) <= 2:
            parser.print_usage()
            exit()
            
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        exit()

