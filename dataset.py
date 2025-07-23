"""Generate dataset"""

from typing import Dict, List, TypedDict, Tuple, Any, Optional
from enum import Enum
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from itertools import product
import random
import gc
import csv

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager, StagedPassManager

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
)
from utils.datatypes import FilePath, df_schema, Dimensions
from utils.image import transform_image
from utils.colors import Colors
from generate.random_circuit import get_random_circuit
from utils.helpers import get_measurements

Schema = Dict[str, Any]
Dist = Dict[int, float]
States = List[int]
Measurements = List[int]


class Stages(Enum):
    """Enum for dataset generation stages"""

    GEN_IMAGES = "gen"
    DUPLICATES = "duplicates"
    TRANSFORM = "transform"


class Checkpoint:
    """Class to handle generate data checkpoints"""

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


class CircuitResult(TypedDict):
    """Type for circuit results"""

    index: int
    depth: int
    file: str
    measurements: str  # JSON string
    result: str  # JSON string
    hash: str


def generate_circuit(
    circuit_image_path: FilePath, pm: StagedPassManager, n_qubits: int, total_gates: int
) -> Tuple[QuantumCircuit, int, Measurements]:
    """Generate circuit and return the isa version of the circuit, its depth and the qubits that were measured"""

    # non-interactive backend
    matplotlib.use("Agg")

    qc = get_random_circuit(n_qubits, total_gates)

    type_of_meas = random.randint(0, 1)
    measurements = list(range(n_qubits))

    if type_of_meas == 0:
        measurements = get_measurements(n_qubits)
        total_measurements = len(measurements)

        classical_register = ClassicalRegister(total_measurements)
        qc.add_register(classical_register)
        qc.measure(measurements, classical_register)
    else:
        qc.measure_all()

    drawing = qc.draw("mpl", filename=circuit_image_path)
    plt.close(drawing)

    depth = qc.depth()
    isa_qc = pm.run(qc)

    # release quantum circuit
    del qc
    del drawing
    gc.collect()

    return isa_qc, depth, measurements


def get_circuit_results(qc: QuantumCircuit, sampler: Sampler, shots: int) -> Dist:
    """Execute cirucit on sampler. Returns its quasi dist"""
    return sampler.run([qc], shots=shots).result().quasi_dists[0]  # type: ignore


def fix_dist_gaps(dist: Dist, states: States):
    """Auxiliary function to fill the remaining bitstrings with 0"""
    for state in states:
        result_value = dist.get(state)
        if result_value is None:
            dist[state] = 0


def generate_image(
    index: int,
    states: States,
    image_path: FilePath,
    n_qubits: int,
    total_gates: int,
    shots: int,
) -> CircuitResult:
    """Run an experiment, save its image and return its results"""

    sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=sim, optimization_level=0)
    isa_qc, depth, measurements = generate_circuit(
        image_path, pm, n_qubits, total_gates
    )

    with open(image_path, "rb") as file:
        file_hash = hashlib.md5(file.read()).hexdigest()

    sampler = Sampler()
    result = get_circuit_results(isa_qc, sampler, shots)
    fix_dist_gaps(result, states)

    # clear data
    del sim
    del pm
    del sampler
    del isa_qc
    gc.collect()

    return {
        "index": index,
        "depth": depth,
        "file": image_path,
        "result": json.dumps(list(result.values())),
        "hash": file_hash,
        "measurements": json.dumps(measurements),
    }


def generate_images(
    target_folder: FilePath,
    n_qubits: int,
    total_gates: int,
    shots: int,
    dataset_size: int,
    total_threads: int,
    checkpoint: Checkpoint,
):
    """
    Generate multiple images and saves a dataframe with information about them.
    It runs in multiple threads(processes in this case) to speed up.
    """

    dataset_file_path = dataset_file(target_folder)

    bitstrings_to_int = [
        int("".join(comb), 2) for comb in product("01", repeat=n_qubits)
    ]

    base_dataset_path = dataset_path(target_folder)

    index = checkpoint.index
    with tqdm(total=dataset_size, initial=index) as progress:
        while index < dataset_size:
            args = []

            for i in range(total_threads):
                filename = "%d.jpeg" % (index)
                circuit_image_path = os.path.join(base_dataset_path, filename)

                args.append(
                    (
                        index,
                        bitstrings_to_int,
                        circuit_image_path,
                        n_qubits,
                        total_gates,
                        shots,
                    )
                )
                index += 1

            with ThreadPoolExecutor(max_workers=total_threads) as pool:
                threads = [pool.submit(generate_image, *arg) for arg in args]  # type:ignore

                # The best would be using the polars scan_csv and sink_csv to
                # write memory efficient queries easily.
                # However, it's an experimental feature, and for some reason they don't work
                # well together.
                # https://github.com/pola-rs/polars/issues/22845
                # https://github.com/pola-rs/polars/issues/20468
                # to solve that, we gonna use the built-in python's csv library
                # to append the new lines without loading the whole csv into memory.

                # df = open_csv(dataset_file_path)

                rows = []
                for future in as_completed(threads):  # type: ignore
                    values = list(future.result().values())
                    rows.append(values)

                append_rows_to_df(dataset_file_path, rows)

                del rows
                del threads
                del args
                gc.collect()

                # save_df(df, dataset_file_path)

                # remove df from memory to open avoid excessive
                # of memory usage
                # del df
                # gc.collect()

            progress.update(total_threads)

            checkpoint.index = index
            checkpoint.save()


def remove_duplicated_files(target_folder: FilePath, checkpoint: Checkpoint):
    """Remove images that are duplicated based on its hash"""
    print("%sRemoving duplicated images%s" % (Colors.GREENBG, Colors.ENDC))

    if not checkpoint.files:  # empty list
        dataset_file_path = dataset_file(target_folder)

        df = open_csv(dataset_file_path)

        # once if we stop at some point the dataset generation,
        # when we resume it, there's some chance of have another row with the same
        # file index. The file is overwritten, but another line will be added and
        # it can raise inconsistency. So before checking distinct hashes, we check
        # for distinct file paths and set is at the default.
        df = df.filter(pl.col("file").is_first_distinct())

        clean_df = df.filter(pl.col("hash").is_first_distinct())
        # even though using the combination scan_csv + sink_csv is not a good idea, using
        # it interchanged with a filter
        save_df(clean_df, dataset_file_path)

        duplicated_files = (
            df.join(clean_df, on=df.columns, how="anti").collect().get_column("file")
        )

        checkpoint.files = duplicated_files.to_list()

        # to avoid useless memory usage
        del duplicated_files
        del clean_df
        del df
        gc.collect()

        checkpoint.save()

    print("%sDeleting duplicated files%s" % (Colors.GREENBG, Colors.ENDC))
    for file in tqdm(checkpoint.files):
        os.remove(file)

        checkpoint.files.remove(file)
        checkpoint.save()


def transform_images(
    target_folder: FilePath, new_dim: Dimensions, checkpoint: Checkpoint
):
    """Normalize images and save them into a h5 file"""
    print("%sTransforming images%s" % (Colors.GREENBG, Colors.ENDC))

    df = open_csv(dataset_file(target_folder))

    current_index = checkpoint.index
    amount_of_rows_per_iteration = 500

    images: List[FilePath] = []
    while True:
        collected_rows = (
            df.slice(offset=current_index, length=amount_of_rows_per_iteration)
            .collect()
            .get_column("file")
            .to_list()
        )

        if len(collected_rows) <= 0:
            break

        images = [*images, *collected_rows]
        current_index += amount_of_rows_per_iteration

    max_width, max_height = new_dim

    image_i = checkpoint.index
    with h5py.File(images_h5_file(target_folder), "a") as file:
        for image_path in tqdm(images):
            with Image.open(image_path) as img:
                tensor = transform_image(img, max_width, max_height)
                file.create_dataset(f"{image_i}", data=tensor)

            image_i += 1
            checkpoint.index = image_i
            checkpoint.save()


def crate_dataset_folder(base_folder: FilePath):
    """Create a folder to store images for the dataset"""
    os.makedirs(dataset_path(base_folder), exist_ok=True)


def create_df() -> pl.LazyFrame:
    """returns a Polars LazyFrame based on schema"""
    return pl.LazyFrame(schema=df_schema)


def open_csv(path: FilePath) -> pl.LazyFrame:
    """opens the CSV file and import it as a LazyFrame"""
    csv = pl.scan_csv(path)
    return csv.cast(df_schema)


def save_df(df: pl.LazyFrame, file_path: FilePath):
    """
    Save dataset as csv.

    Even though it's not a good idea to open a csv file using
    scan_csv and then save using sink_csv, it works for some cases
    and I'll let it here for now.
    """
    df.sink_csv(file_path)


def append_rows_to_df(file_path: FilePath, rows: List[List[Any]]):
    """
    Use pythons built-in csv library to append rows into a file
    without loading it into memory directly.
    """

    with open(file_path, "a") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def start_df(base_file_path: FilePath):
    """
    generates an empty df and saves it on a csv file.

    It's not a good idea to use the scan_csv+sink_csv, but for
    an empty lazyFrame it works well.
    """
    df = create_df()
    save_df(df, dataset_file(base_file_path))

    del df
    gc.collect()


def main(args: Arguments):
    """generate, clean and save dataset and images"""

    crate_dataset_folder(args.target_folder)

    start_df(args.target_folder)

    checkpoint = Checkpoint(images_gen_checkpoint_file(args.target_folder))

    if checkpoint.stage == Stages.GEN_IMAGES:
        generate_images(
            args.target_folder,
            args.n_qubits,
            args.max_gates,
            args.shots,
            args.dataset_size,
            args.threads,
            checkpoint,
        )

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
