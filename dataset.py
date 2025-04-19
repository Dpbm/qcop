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
import json
import hashlib
from PIL import Image
import torch 
from torchvision.transforms import v2
from constants import *

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

    sampler_aer = Sampler()
    result_aer = sampler_aer.run([aer_isa], shots=SHOTS).result().quasi_dists[0]
   
    with Image.open(circuit_image_path) as img:
        img_width, img_height = img.size

    return {
        "depth":depth,
        "file": filename,
        "result": json.dumps(result_aer),
        "hash": file_hash,
        "img_width":img_width,
        "img_height":img_height
    }

def generate_images():
    index = 0 
    df = pd.DataFrame(columns=("depth", "file", "result", "hash", "img_width", "img_height"))

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
    
def transform_images():
    print("Transforming images")
    max_width,max_height = get_new_dim()
    transform = v2.Compose([
        v2.Resize((max_width, max_height), interpolation=Image.LANCZOS),
        v2.PILToTensor(),
        v2.ToDtype(torch.float16),
    ])

    with open(NPY_IMAGES_FILE, "wb") as npy:
        for image in tqdm(os.listdir(DATASET_PATH)):
            image_path = os.path.join(DATASET_PATH, image)
            
            with Image.open(image_path) as img:
                tensor = transform
                np.save(npy, (np.asarray(img)/255.0))


def main():
    os.makedirs(DATASET_PATH, exist_ok=True)
    generate_images()
    remove_duplicated_files()
    transform_images()
    


if __name__ == "__main__":
    main()
