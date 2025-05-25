"""Generate dataset"""
from typing import Dict, List, TypedDict, Tuple
import os 
from multiprocessing.pool import Pool
import hashlib
import json
from itertools import product
import random

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager, StagedPassManager

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import polars as pl
from PIL import Image
import torch 
import h5py

from constants import *
from image import transform_image
from colors import Colors
from random_circuit import get_random_circuit
from helpers import get_measurements

Dist = Dict[int,float]
States = List[int]
FilePath = str
Measurements = List[int]

class CircuitResult(TypedDict):
    """Type for circuit results"""
    index:pl.Series #int
    depth:pl.Series #int
    file:pl.Series #string
    measurements:pl.Series #JSON string
    result:pl.Series # JSON string
    hash:pl.Series #string

def generate_circuit(circuit_image_path:FilePath, pm:StagedPassManager) -> Tuple[QuantumCircuit, int, Measurements]:
    """Generate circuit and return the isa version of the circuit, its depth and the qubits that were measured"""

    qc = get_random_circuit(N_QUBITS, MAX_TOTAL_GATES)

    type_of_meas = random.randint(0,1)
    measurements = list(range(N_QUBITS))

    if type_of_meas == 0:
        measurements = get_measurements(N_QUBITS)
        total_measurements = len(measurements)

        classical_register = ClassicalRegister(total_measurements)
        qc.add_register(classical_register)
        qc.measure(measurements, classical_register)
    else:
        qc.measure_all()

    qc.draw('mpl', filename=circuit_image_path)

    depth = qc.depth() 

    isa_qc = pm.run(qc)
    return isa_qc, depth, measurements

def get_circuit_results(qc:QuantumCircuit, sampler:Sampler) -> Dist:
    """Execute cirucit on sampler. Returns its quasi dist"""

    return sampler.run([qc], shots=SHOTS).result().quasi_dists[0]

def fix_dist_gaps(dist:Dist, states:States):
    """Auxiliary function to fill the remaining bitstrings with 0"""

    for state in states:
        result_value = dist.get(state)
        if result_value is None:
            dist[state] = 0

def generate_image(index:int, states:States, image_path:FilePath) -> CircuitResult:
    """Run an experiment, save its image and return its results"""

    sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=sim, optimization_level=0)
    isa_qc,depth,measurements = generate_circuit(image_path, pm)

    with open(image_path, "rb") as file:
        file_hash = hashlib.md5(file.read()).hexdigest()

    sampler = Sampler()
    result = get_circuit_results(isa_qc, sampler)
    fix_dist_gaps(result, states)
   
    return {
        "index": pl.Series("index", [index], dtype=pl.UInt16),
        "depth":pl.Series("depth", [depth], dtype=pl.UInt8),
        "file": pl.Series("file", [image_path], dtype=pl.String),
        "result": pl.Series("result", [json.dumps(list(result.values()))], dtype=pl.String),
        "hash": pl.Series("hash", [file_hash], dtype=pl.String),
        "measurements": pl.Series("measurements", [json.dumps(measurements)], dtype=pl.String)
    }

def generate_images() -> pl.DataFrame:
    """
    Generate multiple images and return a dataframe with information about them.
    It runs in multiple threads(processes in this case) to speed up.
    """

    df = pl.DataFrame(schema={"index":pl.UInt16, "depth":pl.UInt8, "file":pl.String, "result":pl.String, "hash":pl.String, "measurements":pl.String})

    bitstrings_to_int = [ int(''.join(comb), 2) for comb in product('01', repeat=N_QUBITS) ]

    with tqdm(total=DATASET_SIZE)  as progress:

        index = 0 
        while index < DATASET_SIZE:

            args = []

            for i in range(TOTAL_THREADS):
                filename = '%d.jpeg'%(index)
                circuit_image_path = os.path.join(DATASET_PATH, filename)

                args.append((index, bitstrings_to_int, circuit_image_path))
                index += 1

            with Pool(processes=TOTAL_THREADS) as pool:
                results = pool.starmap(generate_image, args)
            
            for result in results:
                tmp_df = pl.DataFrame(result)
                df.vstack(tmp_df, in_place=True)

            progress.update(TOTAL_THREADS)

    return df

def remove_duplicated_files(df:pl.DataFrame) -> pl.DataFrame:
    """Remove images that are duplicated based on its hash"""

    clean_df = df.unique(maintain_order=True, subset=["hash"])
    clean_df_indexes = clean_df.get_column("index")

    duplicated_values = df.filter(~pl.col("index").is_in(clean_df_indexes))

    print("%sDeleting duplicated files%s"%(Colors.GREENBG, Colors.ENDC))
    for row in tqdm(duplicated_values.iter_rows(named=True)):
        file = row["file"]
        os.remove(file)

    return clean_df
    
    
def transform_images(df:pl.DataFrame):
    """Normalize images and save them into a h5 file"""
    print("%sTransforming images%s"%(Colors.GREENBG,Colors.ENDC))

    max_width,max_height = NEW_DIM

    image_i = 0
    with h5py.File(IMAGES_H5_FILE, "w") as file:
        for row in tqdm(df.iter_rows(named=True)):
            image_path = row["file"]

            with Image.open(image_path) as img:
                tensor = transform_image(img, max_width, max_height)
                file.create_dataset(f"{image_i}", data=tensor)

            image_i += 1

def main():
    """generate, clean and save dataset and images"""

    os.makedirs(DATASET_PATH, exist_ok=True)
    
    df = generate_images()
    df = remove_duplicated_files(df)
    
    df.write_csv(DATASET_FILE)    

    transform_images(df)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
