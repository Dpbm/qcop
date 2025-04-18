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

from constants import *


def generate(sim, index, initial_depth):

    qc = random_circuit(N_QUBITS, initial_depth)
    qc.measure_all()
    
    pm_aer = generate_preset_pass_manager(backend=sim, optimization_level=0)
    aer_isa = pm_aer.run([qc])[0]

    depth = aer_isa.depth()
   
    filename = f'{index}-{depth}.jpeg'
    circuit_image_path = os.path.join(DATASET_PATH, filename)
    qc.draw('mpl', filename=circuit_image_path)

    with open(circuit_image_path, "rb") as file:
        file_hash = hashlib.md5(file.read()).hexdigest()

    sampler_aer = Sampler()
    result_aer = sampler_aer.run([aer_isa], shots=SHOTS).result().quasi_dists[0]
    
    return {
        "depth":depth,
        "file": filename,
        "result": json.dumps(result_aer),
        "hash": file_hash
    }
    
def main():
    os.makedirs(DATASET_PATH, exist_ok=True)

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
                results = pool.starmap(generate, args)
            
            for result in results:
                df.loc[len(df)] = result

            progress.update(TOTAL_THREADS)

    df.to_csv(DATASET_FILE, index=False)

if __name__ == "__main__":
    main()
