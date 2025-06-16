"""Generate dataset"""

from typing import Dict, List, TypedDict, Tuple, Any
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from itertools import product
import random

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager, StagedPassManager

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler

from tqdm import tqdm
import polars as pl
from PIL import Image
import h5py

from args.parser import parse_args, Arguments
from utils.constants import dataset_path, dataset_file, images_h5_file
from utils.datatypes import FilePath, df_schema, Dimensions
from utils.image import transform_image
from utils.colors import Colors
from generate.random_circuit import get_random_circuit
from utils.helpers import get_measurements

Schema = Dict[str, Any]
Dist = Dict[int, float]
States = List[int]
Measurements = List[int]


class CircuitResult(TypedDict):
    """Type for circuit results"""

    index: pl.Series  # int
    depth: pl.Series  # int
    file: pl.Series  # string
    measurements: pl.Series  # JSON string
    result: pl.Series  # JSON string
    hash: pl.Series  # string


def generate_circuit(
    circuit_image_path: FilePath, pm: StagedPassManager, n_qubits: int, total_gates: int
) -> Tuple[QuantumCircuit, int, Measurements]:
    """Generate circuit and return the isa version of the circuit, its depth and the qubits that were measured"""

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

    qc.draw("mpl", filename=circuit_image_path)

    depth = qc.depth()

    isa_qc = pm.run(qc)
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

    return {
        "index": pl.Series("index", [index], dtype=pl.UInt16),
        "depth": pl.Series("depth", [depth], dtype=pl.UInt8),
        "file": pl.Series("file", [image_path], dtype=pl.String),
        "result": pl.Series(
            "result", [json.dumps(list(result.values()))], dtype=pl.String
        ),
        "hash": pl.Series("hash", [file_hash], dtype=pl.String),
        "measurements": pl.Series(
            "measurements", [json.dumps(measurements)], dtype=pl.String
        ),
    }


def generate_images(
    target_folder: FilePath,
    n_qubits: int,
    total_gates: int,
    shots: int,
    dataset_size: int,
    total_threads: int,
):
    """
    Generate multiple images and saves a dataframe with information about them.
    It runs in multiple threads(processes in this case) to speed up.
    """

    dataset_file_path = dataset_file(target_folder)
    df = open_csv(dataset_file_path)

    bitstrings_to_int = [
        int("".join(comb), 2) for comb in product("01", repeat=n_qubits)
    ]

    base_dataset_path = dataset_path(target_folder)

    with tqdm(total=dataset_size) as progress:
        index = 0
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
                for future in as_completed(threads):  # type: ignore
                    tmp_df = create_df(future.result())
                    df.vstack(tmp_df, in_place=True)

            progress.update(total_threads)
            save_df(df, dataset_file_path)


def remove_duplicated_files(target_folder: FilePath):
    """Remove images that are duplicated based on its hash"""

    dataset_file_path = dataset_file(target_folder)

    df = open_csv(dataset_file_path)
    clean_df = df.unique(maintain_order=True, subset=["hash"])
    clean_df_indexes = clean_df.get_column("index")

    duplicated_values = df.filter(~pl.col("index").is_in(clean_df_indexes))

    print("%sDeleting duplicated files%s" % (Colors.GREENBG, Colors.ENDC))
    for row in tqdm(duplicated_values.iter_rows(named=True)):
        file = row["file"]
        os.remove(file)

    save_df(clean_df, dataset_file_path)


def transform_images(target_folder: FilePath, new_dim: Dimensions):
    """Normalize images and save them into a h5 file"""
    print("%sTransforming images%s" % (Colors.GREENBG, Colors.ENDC))

    df = open_csv(dataset_file(target_folder))

    max_width, max_height = new_dim

    image_i = 0
    with h5py.File(images_h5_file(target_folder), "w") as file:
        for row in tqdm(df.iter_rows(named=True)):
            image_path = row["file"]

            with Image.open(image_path) as img:
                tensor = transform_image(img, max_width, max_height)
                file.create_dataset(f"{image_i}", data=tensor)

            image_i += 1


def crate_dataset_folder(base_folder: FilePath):
    """Create a folder to store images for the dataset"""
    os.makedirs(dataset_path(base_folder), exist_ok=True)


default_df_data: CircuitResult = {
    "index": pl.Series(),
    "depth": pl.Series(),
    "file": pl.Series(),
    "measurements": pl.Series(),
    "result": pl.Series(),
    "hash": pl.Series(),
}


def create_df(data: CircuitResult = default_df_data) -> pl.DataFrame:
    """returns a Polars DataFrame schema"""
    return pl.DataFrame(data, schema=df_schema)


def open_csv(path: FilePath) -> pl.DataFrame:
    """opens the CSV file and import it as a DataFrame"""
    csv = pl.read_csv(path)
    return csv.cast(df_schema)


def save_df(df: pl.DataFrame, file_path: FilePath):
    """Save dataset as csv"""
    df.write_csv(file_path)


def start_df(base_file_path: FilePath):
    """generates an empty df and saves it on a csv file"""
    df = create_df()
    save_df(df, dataset_file(base_file_path))


def main(args: Arguments):
    """generate, clean and save dataset and images"""

    crate_dataset_folder(args.target_folder)

    start_df(args.target_folder)

    generate_images(
        args.target_folder,
        args.n_qubits,
        args.max_gates,
        args.shots,
        args.dataset_size,
        args.threads,
    )

    remove_duplicated_files(args.target_folder)

    transform_images(args.target_folder, args.new_image_dim)


if __name__ == "__main__":
    try:
        args = parse_args()
        main(args)
    except KeyboardInterrupt:
        sys.exit(0)
