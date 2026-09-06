"""Methods for handling circuit images."""

from itertools import product, combinations
from typing import List, Callable, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
from collections import defaultdict
import json
import hashlib

from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler

from utils.datatypes import  FilePath,DFRows,Measurements
from utils.constants import SCALE_CIRCUIT_SIZE
from generate.random_circuit import get_random_circuit
from .checkpoint import Checkpoint
from .dataframe import DF

STEP_FOR_SAVE_CHECKPOINT = 10

def update_index_callback(thread_index:int, img_index:int, checkpoint:Checkpoint):
    """Updated thread indexes and save checkpoint when necessary"""
    checkpoint.thread_indexes[thread_index] = img_index
    
    # this is not thread safe, but works in the general case
    if(img_index > 0 and img_index % STEP_FOR_SAVE_CHECKPOINT == 0):
        checkpoint.save()

class Images:
    """Class for handling circuit images."""

    def __init__(self, folder:FilePath):
        self._folder = folder

    def generate_images(
        self,
        n_qubits:int,
        amount_circuits:int,
        total_gates:int,
        shots:int,
        df:DF,
        total_threads: int,
        checkpoint:Checkpoint,
    ):
        """Generate the images split into threads."""

        circuit_format_counter = checkpoint.index

        measurement_combs = Images._get_combinations_of_measurements(n_qubits)
        total_measurement_combs = len(measurement_combs)

        with tqdm(total=amount_circuits, initial=circuit_format_counter) as progress:
            while circuit_format_counter < amount_circuits:
                args = []

                has_thread_indexes = len(checkpoint.thread_indexes) > 0
                total_iter =  (amount_circuits - circuit_format_counter) \
                                if total_threads > (amount_circuits - circuit_format_counter) \
                                else total_threads
                for i in range(total_iter):
                    args.append(
                        (
                            i,
                            (circuit_format_counter * total_measurement_combs) + (i * total_measurement_combs),
                            measurement_combs,
                            n_qubits,
                            total_gates,
                            shots,
                            lambda thread_index, img_index: update_index_callback(thread_index, img_index, checkpoint),
                            checkpoint.thread_indexes[i] if has_thread_indexes else 0
                        )
                    )

                    if not has_thread_indexes:
                        checkpoint.thread_indexes.append(0)


                with ThreadPoolExecutor(max_workers=total_measurement_combs) as pool:
                    threads = [pool.submit(self._generate_circuit_images, *arg) for arg in args]
                    
                    rows = []
                    for future in as_completed(threads):
                        try:
                            future_rows = [list(result.values())  for result in future.result()]
                            rows = [*rows, *future_rows]
                            circuit_format_counter += 1
                            progress.update(1)
                        except Exception as error:
                            print("Error: %s" % error)
                            sys.exit(1)

                    df.append_rows_to_file(rows)
                    checkpoint.index += total_iter
                    checkpoint.thread_indexes = [ 0 for _ in range(total_threads) ]
                    checkpoint.save()

    def _generate_circuit_images(
            self,
            thread_index:int,
            base_index:int, 
            measurements:Measurements, 
            n_qubits:int, 
            total_gates:int,
            shots: int,
            callback:Callable,
            current_index:int=0,
    ) -> DFRows:
        """ Run an experiment, save its images and return its results for different combinations 
        of measurements.
        """

        sim = AerSimulator()
        pm = generate_preset_pass_manager(backend=sim, optimization_level=0)
        sampler = Sampler()
        qc = get_random_circuit(n_qubits, total_gates)
        outputs = []

        total_meas_combs = len(measurements)

        for i in range(current_index, total_meas_combs):
            meas = measurements[i]
            qc_copy = qc.copy()
            
            img_index = base_index + i
            img_path = os.path.join(self._folder, "%d.png" % img_index)
            total_measurements = len(meas)

            # non-interactive backend
            matplotlib.use("Agg")

            classical_register = ClassicalRegister(total_measurements, name="c")
            qc_copy.add_register(classical_register)
            qc_copy.measure(meas, list(range(total_measurements)))

            gates_per_type_count = {
                    1:0, # single qubit gates
                    2:0, # two qubit gates
                }
            barriers_count = 0
            total_gates_count = defaultdict(int)

            for inst in qc_copy.data:
                if inst.name == "barrier":
                    barriers_count += 1
                elif inst.name != "measure":
                    qubits = len(inst.qubits)
                    gates_per_type_count[qubits] += 1
                    total_gates_count[inst.name] += 1

            drawing = qc_copy.draw(
                    "mpl", 
                    filename=img_path, 
                    fold=-1, 
                    scale=SCALE_CIRCUIT_SIZE)
            plt.close(drawing)

            depth = qc_copy.depth()
            isa_qc = pm.run(qc_copy)

            with open(img_path, "rb") as file:
                img = Image.open(file)
                width, height = img.size
                file_hash = hashlib.md5(file.read()).hexdigest()
                img.close()
                
            outcomes = sampler.run([isa_qc], shots=shots).result().quasi_dists[0]
            Images._fix_dist_gaps(outcomes, n_qubits)

            outputs.append({
                    "index": img_index,
                    "depth": depth,
                    "file": img_path,
                    "result": json.dumps(list(outcomes.values())),
                    "hash": file_hash,
                    "total_meas": total_measurements,
                    "measurements": json.dumps(meas),
                    "img_width": width,
                    "img_height": height,
                    "n_two_qubit_gates": gates_per_type_count.get(2, 0),
                    "n_one_qubit_gates": gates_per_type_count.get(1, 0),
                    "amount_gates": json.dumps(dict(total_gates_count)),
                    "file_size_bytes": os.path.getsize(img_path),
                    "n_barriers": barriers_count
                })
            callback(thread_index, img_index)
        return outputs

    @staticmethod
    def _fix_dist_gaps(dist: Dict[int,float], n_qubits:int):
        """ Set the remaining values of the distribution (remaining bitstrings) to zero """
        for i in range(2**n_qubits):
            dist[i] = dist.get(i, 0.0)

    @staticmethod
    def _get_combinations_of_measurements(n_qubits:int) -> Measurements:
        """
        Get all measurements combinations for n qubits.
        !!! It may be expensive with a many qubits  !!!
        """

        qubits_iter = list(range(n_qubits))
        measurement_combs = []
        for amount in range(1, n_qubits+1):
            measurement_combs = [
                *measurement_combs,
                *list(combinations(qubits_iter, amount)),  # type: ignore
            ]  # type: ignore

        return measurement_combs
