"""Parse CLI arguments"""

import sys
import argparse
from typing import Optional

from utils.constants import (
    DEFAULT_SHOTS,
    DEFAULT_NUM_QUBITS,
    DEFAULT_MAX_TOTAL_GATES,
    DEFAULT_THREADS,
    DEFAULT_AMOUNT_OF_CIRCUITS,
    DEFAULT_TARGET_FOLDER,
)
from utils.datatypes import Dimensions, FilePath

DEFAULT_DATASET_NAME="dqcop"
DEFAULT_MODEL_NAME="qcop"

class ArgumentsGenerate:
    """Parsed args types for dataset generation"""

    __slots__ = [
            "_threads",
            "_shots",
            "_n_qubits",
            "_max_gates",
            "_amount_circuits",
            "_target_folder",
            "_dataset_name_kaggle",
            "_dataset_name_hf",
            "_model_name_kaggle",
            "_model_name_hf",
    ]
    def __init__(self):
        """set default arguments"""

        self._threads = DEFAULT_THREADS
        self._shots = DEFAULT_SHOTS
        self._n_qubits = DEFAULT_NUM_QUBITS
        self._max_gates = DEFAULT_MAX_TOTAL_GATES
        self._amount_circuits = DEFAULT_AMOUNT_OF_CIRCUITS
        self._target_folder = DEFAULT_TARGET_FOLDER
        self._dataset_name_kaggle = DEFAULT_DATASET_NAME
        self._dataset_name_hf = DEFAULT_DATASET_NAME
        self._model_name_kaggle = DEFAULT_MODEL_NAME
        self._model_name_hf = DEFAULT_MODEL_NAME

    def parse(self, args: argparse.Namespace):
        """Parse arguments from argparse"""
        self._threads = args.threads
        self._shots = args.shots
        self._n_qubits = args.n_qubits
        self._max_gates = args.max_gates
        self._amount_circuits = args.amount_circuits
        self._target_folder = args.target_folder
        self._dataset_name_kaggle = args.dataset_name_kaggle
        self._dataset_name_hf = args.dataset_name_hf
        self._model_name_kaggle = args.model_name_kaggle
        self._model_name_hf = args.model_name_hf


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
    def dataset_name_kaggle(self) -> str:
        """Get dataset_name_kaggle data"""
        return self._dataset_name_kaggle  # type: ignore

    @dataset_name_kaggle.setter
    def dataset_name_kaggle(self, value: str):
        """Set dataset_name_kaggle data"""
        self._dataset_name_kaggle = value
    
    @property
    def dataset_name_hf(self) -> str:
        """Get dataset_name_hf data"""
        return self._dataset_name_hf  # type: ignore

    @dataset_name_hf.setter
    def dataset_name_hf(self, value: str):
        """Set dataset_name_hf data"""
        self._dataset_name_hf = value
    
    @property
    def model_name_kaggle(self) -> str:
        """Get model_name_kaggle data"""
        return self._model_name_kaggle  # type: ignore

    @model_name_kaggle.setter
    def model_name_kaggle(self, value: str):
        """Set model_name_kaggle data"""
        self._model_name_kaggle = value
    
    @property
    def model_name_hf(self) -> str:
        """Get model_name_hf data"""
        return self._model_name_hf  # type: ignore

    @model_name_hf.setter
    def model_name_hf(self, value: str):
        """Set model_name_hf data"""
        self._model_name_hf = value
    

def parse_args_generate() -> ArgumentsGenerate:
    """
    Use argparse to parse CLI arguments for all scripts
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name-kaggle", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-name-hf", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--model-name-kaggle", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-name-hf", type=str, default=DEFAULT_MODEL_NAME)

    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)

    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--n-qubits", type=int, default=DEFAULT_NUM_QUBITS)
    parser.add_argument("--max-gates", type=int, default=DEFAULT_MAX_TOTAL_GATES)

    parser.add_argument(
        "--amount-circuits", type=int, default=DEFAULT_AMOUNT_OF_CIRCUITS
    )
    parser.add_argument("--target-folder", type=str, default=DEFAULT_TARGET_FOLDER)

    if len(sys.argv) <= 2:
        parser.print_usage()
        raise SystemExit
        
    args = parser.parse_args()
    parsed_arguments = ArgumentsGenerate()
    parsed_arguments.parse(args)

    return parsed_arguments
