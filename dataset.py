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

SHOTS = 1000

N_QUBITS = 5

MIN_DEPTH = 1
MAX_DEPTH = 100

TOTAL_THREADS = 10

DATASET_SIZE = 2000
DATASET_PATH = os.path.join('.', 'dataset')

def generate(sim, index, depth):

    qc = random_circuit(N_QUBITS, depth)
    qc.measure_all()
   
    filename = f'{index}-{depth}.jpeg'
    circuit_image_path = os.path.join(DATASET_PATH, filename)
    qc.draw('mpl', filename=circuit_image_path)

    pm_aer = generate_preset_pass_manager(backend=sim, optimization_level=0)
    aer_isa = pm_aer.run([qc])[0]


    sampler_aer = Sampler()
    result_aer = sampler_aer.run([aer_isa], shots=SHOTS).result().quasi_dists[0]
    
    return {
        "depth":depth,
        "file": filename,
        "result": json.dumps(result_aer),
    }
    
def main():
    os.makedirs(DATASET_PATH, exist_ok=True)

    index = 0 
    df = pd.DataFrame(columns=("depth", "file", "result"))

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

    df.to_csv("dataset.csv", index=False)

if __name__ == "__main__":
    main()
