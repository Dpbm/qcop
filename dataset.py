"""Generate dataset"""

from typing import Dict, List, TypedDict, Any, Optional
from enum import Enum
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from itertools import product, combinations
import gc
import csv

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler

import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import polars as pl
from PIL import Image
import h5py

from args.parser import parse_args, Arguments
from utils.constants import (
    dataset_path,
    dataset_file,
    images_h5_file,
    images_gen_checkpoint_file,
    dataset_file_tmp,
    SCALE_CIRCUIT_SIZE
)
from utils.datatypes import FilePath, df_schema, Dimensions
from utils.image import transform_image
from utils.colors import Colors
from generate.random_circuit import get_random_circuit

Schema = Dict[str, Any]
Dist = Dict[int, float]
States = List[int]
Measurements = List[int]
MeasurementsCombinations = List[Measurements]
Rows = List[List[Any]]


class Stages(Enum):
    """Enum for dataset generation stages"""

    GEN_IMAGES = "gen"
    SHUFFLE = "shuffle"
    DUPLICATES = "duplicates"
    TRANSFORM = "transform"


class Checkpoint:
    """Class to handle generate data checkpoints"""

    __slots__ = ["_path", "_stage", "_index", "_files"]

    def __init__(self, path: Optional[FilePath]):
        self._path = path

        self._stage = Stages.GEN_IMAGES
        self._index = 0
        self._files: List[FilePath] = []

        # Check file and get the data

        if self._path is None:
            print("%sNo Checkpoint was provided!%s" % (Colors.YELLOWFG, Colors.ENDC))
            return

        if not os.path.exists(self._path):
            print(
                "%sCheckpoint file %s doesn't exists!%s"
                % (Colors.YELLOWFG, self._path, Colors.ENDC)
            )
            return

        print(
            "%sLoading checkpoint from: %s...%s"
            % (Colors.MAGENTABG, self._path, Colors.ENDC)
        )

        with open(self._path, "r") as file:
            data = json.load(file)
            stage = data.get("stage")
            self._stage = Stages.GEN_IMAGES if stage is None else Stages(stage)
            self._index = data.get("index") or 0
            self._files = data.get("files") or []

    @property
    def stage(self) -> Stages:
        """get checkpoint generation stage"""
        return self._stage

    @stage.setter
    def stage(self, value: Stages):
        """Update stage"""
        self._stage = value

    @property
    def index(self) -> int:
        """get checkpoint generation index"""
        return self._index

    @index.setter
    def index(self, value: int):
        """update index"""
        self._index = value

    @property
    def files(self) -> List[FilePath]:
        """get duplicated files to remove"""
        return self._files

    @files.setter  # type: ignore
    def files(self, value: List[FilePath]):
        """set files to delete"""
        self._files = value

    def save(self):
        """Saves checkpoint to a json file"""
        print(
            "%sSaving checkpoint at: %s%s" % (Colors.GREENBG, self._path, Colors.ENDC)
        )
        with open(self._path, "w") as file:
            data = {
                "stage": self._stage.value,
                "index": self._index,
                "files": self._files,
            }
            json.dump(data, file)

    def __str__(self) -> str:
        return "Checkpoint: %s; stage: %s; index: %d; total_files: %d" % (
            self._path,
            self._stage.value,
            self._index,
            len(self._files),
        )



def shuffle_csv(target_folder:FilePath):
    """
    Shuffle CSV rows.
    """
    print("%sShuffling DF....%s"%(Colors.GREENBG,Colors.ENDC))
    file_path = dataset_file(target_folder)
    df = pl.read_csv(file_path)
    df = shuffle_df(df)
    df.write_csv(file_path)




def main(args: Arguments):
    """generate, clean and save dataset and images"""

    crate_dataset_folder(dataset_file(args.target_folder))
    start_df(args.target_folder)
    checkpoint = Checkpoint(images_gen_checkpoint_file(args.target_folder))

    if checkpoint.stage == Stages.GEN_IMAGES:
        generate_images(
            args.target_folder,
            args.n_qubits,
            args.max_gates,
            args.shots,
            args.amount_circuits,
            args.threads,
            checkpoint,
        )
        checkpoint.stage = Stages.DUPLICATES
        checkpoint.index = 0

    if checkpoint.stage == Stages.SHUFFLE:
        shuffle_csv(args.target_folder)
        checkpoint.stage = Stages.DUPLICATES
        checkpoint.index = 0

    if checkpoint.stage == Stages.DUPLICATES:
        remove_duplicated_files(args.target_folder, checkpoint)
        checkpoint.stage = Stages.TRANSFORM
        checkpoint.index = 0

    transform_images(args.target_folder, args.new_image_dim, checkpoint)


if __name__ == "__main__":
    try:
        args = parse_args()
        main(args)
    except KeyboardInterrupt:
        sys.exit(0)
