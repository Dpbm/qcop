from typing import Dict, List, TypedDict
import os 
from multiprocessing.pool import Pool
import hashlib
import json
from itertools import product

from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
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

Dist = Dict[int,float]
States = List[int]
class CircuitResult(TypedDict):
    index:pl.Series
    depth:pl.Series
    file:pl.Series
    result:pl.Series # JSON string
    hash:pl.Series


def generate_circuit(depth:int, circuit_image_path:str, pm:StagedPassManager) -> QuantumCircuit:
    qc = random_circuit(N_QUBITS, depth)
    qc.measure_all()
    qc.draw('mpl', filename=circuit_image_path)
    

    isa_qc = pm.run(qc)
    return isa_qc

def get_circuit_results(qc:QuantumCircuit, sampler:Sampler) -> Dist:
    return sampler.run([qc], shots=SHOTS).result().quasi_dists[0]

def fix_dist_gaps(dist:Dist, states:States):
    for state in states:
        result_value = dist.get(state)
        if result_value is None:
            dist[state] = 0

def generate_image(index:int, depth:int, states:States, image_path:str) -> CircuitResult:
    sim = AerSimulator()
    pm = generate_preset_pass_manager(backend=sim, optimization_level=0)
    isa_qc = generate_circuit(depth, image_path, pm)

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
        "hash": pl.Series("hash", [file_hash], dtype=pl.String)
    }

def generate_images() -> pl.DataFrame:
    df = pl.DataFrame(schema={"index":pl.UInt16, "depth":pl.UInt8, "file":pl.String, "result":pl.String, "hash":pl.String})

    bitstrings_to_int = [ int(''.join(comb), 2) for comb in product('01', repeat=N_QUBITS) ]

    with tqdm(total=DATASET_SIZE)  as progress:

        index = 0 
        while index < DATASET_SIZE:

            args = []

            for i in range(TOTAL_THREADS):
                depth = np.random.randint(MIN_DEPTH, MAX_DEPTH)

                filename = '%d-%d.jpeg'%(index,depth)
                circuit_image_path = os.path.join(DATASET_PATH, filename)

                args.append((index, depth, bitstrings_to_int, circuit_image_path))
                index += 1

            with Pool(processes=TOTAL_THREADS) as pool:
                results = pool.starmap(generate_image, args)
            
            for result in results:
                tmp_df = pl.DataFrame(result)
                df.vstack(tmp_df, in_place=True)

            progress.update(TOTAL_THREADS)

    return df

def remove_duplicated_files(df:pl.DataFrame) -> pl.DataFrame:
    clean_df = df.unique(maintain_order=True, subset=["hash"])
    clean_df_indexes = clean_df.get_column("index")

    duplicated_values = df.filter(~pl.col("index").is_in(clean_df_indexes))

    print("%sDeleting duplicated files%s"%(Colors.GREENBG, Colors.ENDC))
    for row in tqdm(pl.iter_rows(named=True)):
        file = row["file"]
        os.remove(file)

    return clean_df
    
    
def transform_images(df:pl.DataFrame, percentage_train:float=0.8):
    print("%sTransforming images%s"%(Colors.GREENBG,Colors.ENDC))
    print("%s%f For Training%s"%(Colors.YELLOWFG,percentage_train*100,Colors.ENDC))

    max_width,max_height = new_dim

    total_rows_training = int(len(df)*percentage_train)
    total_rows_test = len(df)-total_rows_training
    print("%s%d rows for training%s"%(Colors.YELLOWFG, total_rows_training, Colors.ENDC))
    print("%s%d rows for testing%s"%(Colors.YELLOWFG, total_rows_test, Colors.ENDC))

    images_train = h5py.File(IMAGES_TRAIN, "w")
    images_test = h5py.File(IMAGES_TEST, "w")

    train_rows = df.head(total_rows_training)
    test_rows = df.tail(total_rows_test)
    image_i = 0

    for rows, h5_file in zip((train_rows, test_rows), (images_train, images_test)):
        for row in tqdm(rows.iter_rows(named=True)):
            image_path = row["file"]
            with Image.open(image_path) as img:
                tensor = transform_image(img)
                h5_file.create_dataset(f"{image_i}", data=tensor)
            image_i += 1
    
    images_train.close()
    images_test.close()

def main():
    os.makedirs(DATASET_PATH, exist_ok=True)

    df = generate_images()
    df = remove_duplicated_files(df)

    df.write_csv(DATASET_PATH)    

    transform_images()
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
