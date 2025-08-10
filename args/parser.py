"""Parse CLI arguments"""

import sys
import argparse
from typing import Optional

from utils.constants import (
    DEFAULT_EPOCHS,
    DEFAULT_SHOTS,
    DEFAULT_NUM_QUBITS,
    DEFAULT_MAX_TOTAL_GATES,
    DEFAULT_THREADS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_AMOUNT_OF_CIRCUITS,
    DEFAULT_NEW_DIM,
    DEFAULT_TRAIN_PERCENTAGE,
    DEFAULT_TEST_PERCENTAGE,
    DEFAULT_TARGET_FOLDER,
    DEFAULT_CHECKPOINT,
)
from utils.datatypes import Dimensions, FilePath


class Arguments:
    """Parsed args types"""

    __slots__ = [
        "_epochs",
        "_batch_size",
        "_train_size",
        "_test_size",
        "_threads",
        "_shots",
        "_n_qubits",
        "_max_gates",
        "_amount_circuits",
        "_target_folder",
        "_checkpoint",
        "_new_image_dim",
    ]

    def __init__(self):
        """set default arguments"""

        self._epochs = DEFAULT_EPOCHS
        self._batch_size = DEFAULT_BATCH_SIZE
        self._train_size = DEFAULT_TRAIN_PERCENTAGE
        self._test_size = DEFAULT_TEST_PERCENTAGE
        self._threads = DEFAULT_THREADS
        self._shots = DEFAULT_SHOTS
        self._n_qubits = DEFAULT_NUM_QUBITS
        self._max_gates = DEFAULT_MAX_TOTAL_GATES
        self._amount_circuits = DEFAULT_AMOUNT_OF_CIRCUITS
        self._target_folder = DEFAULT_TARGET_FOLDER
        self._checkpoint = DEFAULT_CHECKPOINT
        self._new_image_dim = DEFAULT_NEW_DIM

    def parse(self, args: argparse.Namespace):
        """Parse arguments from argparse"""
        self._epochs = args.epochs
        self._batch_size = args.batch_size
        self._train_size = args.train_size
        self._test_size = args.test_size
        self._threads = args.threads
        self._shots = args.shots
        self._n_qubits = args.n_qubits
        self._max_gates = args.max_gates
        self._amount_circuits = args.amount_circuits
        self._target_folder = args.target_folder
        self._checkpoint = args.checkpoint
        self._new_image_dim = args.new_image_dim

    @property
    def epochs(self) -> int:
        """Get epochs data"""
        return self._epochs  # type: ignore

    @epochs.setter
    def epochs(self, value: int):
        """Set epochs data"""
        self._epochs = value

    @property
    def batch_size(self) -> int:
        """Get batch_size data"""
        return self._batch_size  # type: ignore

    @batch_size.setter
    def batch_size(self, value: int):
        """Set batch_size data"""
        self._batch_size

    @property
    def train_size(self) -> int:
        """Get train_size data"""
        return self._train_size  # type: ignore

    @train_size.setter
    def train_size(self, value: int):
        """Set train_size data"""
        self._train_size = value

    @property
    def test_size(self) -> int:
        """Get test_size data"""
        return self._test_size  # type: ignore

    @test_size.setter
    def test_size(self, value: int):
        """Set test_size data"""
        self._test_size = value

    @property
    def threads(self) -> int:
        """Get threads data"""
        return self._threads  # type: ignore

    @threads.setter
    def threads(self, value: int):
        """Set threads data"""
        self._threads = value

    @property
    def shots(self) -> int:
        """Get shots data"""
        return self._shots  # type: ignore

    @shots.setter
    def shots(self, value: int):
        """Set shots data"""
        self._shots = value

    @property
    def n_qubits(self) -> int:
        """Get n_qubits data"""
        return self._n_qubits  # type: ignore

    @n_qubits.setter
    def n_qubits(self, value: int):
        """Set n_qubits data"""
        self._n_qubits = value

    @property
    def max_gates(self) -> int:
        """Get max_gates data"""
        return self._max_gates  # type: ignore

    @max_gates.setter
    def max_gates(self, value: int):
        """Set max_gates data"""
        self._max_gates = value

    @property
    def amount_circuits(self) -> int:
        """Get amount_circuits data"""
        return self._amount_circuits  # type: ignore

    @amount_circuits.setter
    def amount_circuits(self, value: int):
        """Set amount_circuits data"""
        self._amount_circuits = value

    @property
    def target_folder(self) -> FilePath:
        """Get target_folder data"""
        return self._target_folder  # type: ignore

    @target_folder.setter
    def target_folder(self, value: FilePath):
        """Set target_folder data"""
        self._target_folder = value

    @property
    def checkpoint(self) -> Optional[FilePath]:
        """Get checkpoint data"""
        return self._checkpoint  # type: ignore

    @checkpoint.setter
    def checkpoint(self, value: Optional[FilePath]):
        """Set checkpoint data"""
        self._checkpoint = value

    @property
    def new_image_dim(self) -> Dimensions:
        """Get new_image_dim data"""
        return self._new_image_dim  # type: ignore

    @new_image_dim.setter
    def new_image_dim(self, value: Dimensions):
        """Set new_image_dim data"""
        self._new_image_dim = value

    def __str__(self) -> str:
        string = f"epochs: {self._epochs}\n"
        string += f"batch size: {self._batch_size}\n"
        string += f"train size: {self._train_size}\n"
        string += f"teste size: {self._test_size}\n"
        string += f"threads: {self._threads}\n"
        string += f"shots: {self._shots}\n"
        string += f"n qubits: {self._n_qubits}\n"
        string += f"max gates: {self._max_gates}\n"
        string += f"amount circuits: {self._amount_circuits}\n"
        string += f"target_folder: {self._target_folder}\n"
        string += f"checkpoint: {self._checkpoint}\n"
        string += f"new image dim: {self._new_image_dim}\n"

        return string


def parse_args() -> Arguments:
    """
    Use argparse to parse CLI arguments for all scripts
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--train-size", type=float, default=DEFAULT_TRAIN_PERCENTAGE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_PERCENTAGE)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)

    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)

    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--n-qubits", type=int, default=DEFAULT_NUM_QUBITS)
    parser.add_argument("--max-gates", type=int, default=DEFAULT_MAX_TOTAL_GATES)

    parser.add_argument(
        "--amount-circuits", type=int, default=DEFAULT_AMOUNT_OF_CIRCUITS
    )
    parser.add_argument("--target-folder", type=str, default=DEFAULT_TARGET_FOLDER)
    parser.add_argument("--new-image-dim", type=int, nargs=2, default=DEFAULT_NEW_DIM)

    args = parser.parse_args(sys.argv[1:])

    parsed_arguments = Arguments()
    parsed_arguments.parse(args)

    return parsed_arguments
