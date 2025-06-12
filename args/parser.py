"""Parse CLI arguments"""

from typing import TypedDict, Optional

import sys
import argparse

from utils.constants import (
    DEFAULT_EPOCHS,
    DEFAULT_SHOTS,
    DEFAULT_NUM_QUBITS,
    DEFAULT_MAX_TOTAL_GATES,
    DEFAULT_THREADS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_SIZE,
    DEFAULT_NEW_DIM,
    DEFAULT_TRAIN_PERCENTAGE,
    DEFAULT_TEST_PERCENTAGE,
    DEFAULT_TARGET_FOLDER,
)
from utils.datatypes import Dimensions, FilePath


class Arguments(TypedDict):
    """Parsed args types"""

    epochs: int
    batch_size: int
    train_size: int
    test_size: int
    threads: int
    shots: int
    n_qubits: int
    max_gates: int
    dataset_size: int
    target_folder: FilePath
    checkpoint: Optional[FilePath]
    new_image_dim: Dimensions


def parse_args() -> Arguments:
    """
    Use argparse to parse CLI arguments for all scripts
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--train-size", type=float, default=DEFAULT_TRAIN_PERCENTAGE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_PERCENTAGE)
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)

    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--n-qubits", type=int, default=DEFAULT_NUM_QUBITS)
    parser.add_argument("--max-gates", type=int, default=DEFAULT_MAX_TOTAL_GATES)

    parser.add_argument("--dataset-size", type=int, default=DEFAULT_DATASET_SIZE)
    parser.add_argument("--target-folder", type=str, default=DEFAULT_TARGET_FOLDER)
    parser.add_argument("--new-image-dim", type=int, nargs=2, default=DEFAULT_NEW_DIM)

    args = parser.parse_args(sys.argv[1:])

    parsed_arguments: Arguments = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "checkpoint": args.checkpoint,
        "threads": args.threads,
        "shots": args.shots,
        "n_qubits": args.n_qubits,
        "max_gates": args.max_gates,
        "dataset_size": args.dataset_size,
        "target_folder": args.target_folder,
        "new_image_dim": args.new_image_dim,
    }

    return parsed_arguments
