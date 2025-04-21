from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler
from qiskit.transpiler import generate_preset_pass_manager
import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocessing.pool import Pool
from tqdm import tqdm
import pandas as pd
import hashlib
from PIL import Image
import torch 
import h5py
from math import floor
import json
from itertools import product

from constants import *
from image import transform_image

def generate_image(sim, index, depth):
    qc = random_circuit(N_QUBITS, depth)
    qc.measure_all()
    
    filename = f'{index}-{depth}.jpeg'
    circuit_image_path = os.path.join(DATASET_PATH, filename)
    qc.draw('mpl', filename=circuit_image_path)

    with open(circuit_image_path, "rb") as file:
        file_hash = hashlib.md5(file.read()).hexdigest()
    
    pm_aer = generate_preset_pass_manager(backend=sim, optimization_level=0)
    aer_isa = pm_aer.run([qc])[0]

    sampler = Sampler()
    result = sampler.run([aer_isa], shots=SHOTS).result().quasi_dists[0]

    for comb in product('01', repeat=N_QUBITS):
        bitstring = ''.join(comb)
        binary_to_dec = int(bitstring, 2)
        result_value = result.get(binary_to_dec)
        if result_value is None:
            result[binary_to_dec] = 0
   
    return {
        "depth":depth,
        "file": filename,
        "result": json.dumps(list(result.values())),
        "hash": file_hash,
    }

def generate_images():
    index = 0 
    df = pd.DataFrame(columns=("depth", "file", "result", "hash"))

    with tqdm(total=DATASET_SIZE)  as progress:
        while index < DATASET_SIZE:
            args = []
            for i in range(TOTAL_THREADS):
                depth = np.random.randint(MIN_DEPTH, MAX_DEPTH)
                sim = AerSimulator()
                args.append((sim, index, depth))
                index += 1

            with Pool(processes=TOTAL_THREADS) as pool:
                results = pool.starmap(generate_image, args)
            
            for result in results:
                df.loc[len(df)] = result

            progress.update(TOTAL_THREADS)

    df.to_csv(DATASET_FILE, index=False)

def remove_duplicated_files():
    csv_file = pd.read_csv(DATASET_FILE)
    duplicated_lines = csv_file.loc[csv_file.duplicated(subset="hash", keep="first")]

    indexes = duplicated_lines.index.to_list()
    files = [os.path.join(DATASET_PATH, file) for file in duplicated_lines["file"].to_list()]
    
    print("Dropping invalid rows")
    csv_file = csv_file.drop(indexes)
    print("Deleting duplicated files")
    for file in tqdm(files):
        os.remove(file)
    
    csv_file.to_csv(DATASET_FILE, index=False)
    
def transform_images(percentage_train=0.6):
    print("Transforming images")
    max_width,max_height = new_dim

    files = os.listdir(DATASET_PATH)
    total_files = len(files)
    total_train_files = floor(total_files * percentage_train)
    total_test_files = total_files - total_train_files

    images_train = h5py.File(IMAGES_TRAIN, "w")
    images_test = h5py.File(IMAGES_TEST, "w")

    for image_i in tqdm(range(total_train_files)):
        image_path = os.path.join(DATASET_PATH, files[image_i])
        
        with Image.open(image_path) as img:
            tensor = transform_image(img)
            images_train.create_dataset(f"{image_i}", data=tensor)
    
    for image_i in tqdm(range(total_test_files)):
        image_path = os.path.join(DATASET_PATH, files[total_train_files+image_i])
        
        with Image.open(image_path) as img:
            tensor = transform_image(img)
            images_test.create_dataset(f"{image_i}", data=tensor)


    images_train.close()
    images_test.close()

def main():
    os.makedirs(DATASET_PATH, exist_ok=True)
    generate_images()
    remove_duplicated_files()
    transform_images()
    

if __name__ == "__main__":
    main()
