"""Methods for handling circuit images."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
from collections import defaultdict
import json
import hashlib
import random

from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, XGate, ZGate, HGate, IGate, CXGate, CZGate, SwapGate
from qiskit.quantum_info import Statevector

from utils.datatypes import  FilePath,DFRow,Measurements
from utils.constants import SCALE_CIRCUIT_SIZE, DEFAULT_RANDOM_SEED
from .checkpoint import Checkpoint
from .dataframe import DF

class RandomCircuit:
    def __init__(self, seed:int=DEFAULT_RANDOM_SEED):
        self._all_gates = {
            'x': lambda  x: XGate(), 
            'z': lambda x: ZGate(), 
            'h': lambda x: HGate(), 
            'id': lambda x: IGate(),  
            'cx': lambda x: CXGate(), 
            'cz': lambda x: CZGate(), 
            'swap': lambda x: SwapGate(), 
            'ry':lambda theta: RYGate(theta)
        }
        self._two_qubit = ['cx', 'cz', 'swap']
        self._one_qubit = ['x', 'z', 'id', 'h', 'ry']

        self._low_param = 0
        self._high_param = 2*np.pi
        
        self._rng = np.random.default_rng(seed)

    def _get_angle(self) -> float:
        return self._rng.uniform(low=self._low_param,high=self._high_param,size=None)

    
    def get_random_circuit(self, max_gates:int, num_qubits:int, max_barriers:int) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)

        # so the circuit can be empty, with 0.1% of chance
        if self._rng.random() <= 0.001:
            # it can have variants with barriers
            num_barriers = np.random.randint(0,max_barriers)
            if not num_barriers:
                return qc

            for _ in range(num_barriers):
                qc.barrier()
            return qc
            
        c = max_gates
        b = max_barriers
        while c > 0:

            # it can stop by random in the middle
            stop_now = self._rng.random() < 0.001
            if stop_now:
                break

            which_type = 2
            r = self._rng.random()
            if r <= 0.05:
                which_type = 1
            elif r <= 0.7:
                which_type = 2
            else:
                which_type = 3

            if which_type == 1:# barrier
                if b <= 0: # no more barriers in this case
                    continue

                if b == 1:
                    qc.barrier()
                    b = 0
                    continue
                
                quantity = np.random.randint(1,b)
                b -= quantity
                
                for _ in range(quantity):
                    qc.barrier()
    
            elif which_type == 2: # one qubit gate
                quantity = 1
                if c != 1:
                    quantity = np.random.randint(1,c)
                c -= quantity

                selected_gates = random.choices(self._one_qubit,k=quantity)
                selected_qubits = random.choices(range(num_qubits),k=quantity)

                for gate,qubit in zip(selected_gates, selected_qubits):
                    angle = self._get_angle()
                    qc.append(self._all_gates[gate](angle), [qubit])
                
                    
            else: # two qubit gate
                quantity = 1
                if c != 1:
                    quantity = np.random.randint(1,c)
                c -= quantity

                selected_gates = random.choices(self._two_qubit,k=quantity)
                
                controls = [np.random.randint(0,num_qubits-1) for _ in range(quantity)]
                targets = []
                for i in range(quantity):
                    #target and control must be different
                    available_targets = list(set(list(range(num_qubits))) - {controls[i]})
                    selected_target = random.choice(available_targets)
                    targets.append(selected_target)
                
                for gate,c,t in zip(selected_gates, controls, targets):
                    qc.append(self._all_gates[gate](None), [c,t])
            
        if self._rng.random() <= 0.01 and b > 0:
            qc.barrier()

        return qc
        

def get_random_circuit(n_qubits: int, total_gates: int) -> QuantumCircuit:
    """Setup parameters and generate a random circuit"""

    # we set a local seed, because if we set the same,
    # every time we restart it would generate the same circuit over-and-over
    local_seed = np.random.randint(0, np.iinfo(np.int32).max)
    rc = RandomCircuit(seed=local_seed)

    gates = int(np.random.randint(1, max(2, total_gates + 1)))
    barriers = int(np.random.randint(1,6))

    return rc.get_random_circuit(gates,n_qubits,barriers)


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

        with tqdm(total=amount_circuits, initial=circuit_format_counter) as progress:
            while circuit_format_counter < amount_circuits:
                args = []

                total_iter =  (amount_circuits - circuit_format_counter) \
                                if total_threads > (amount_circuits - circuit_format_counter) \
                                else total_threads

                for i in range(total_iter):
                    args.append(
                        (
                            circuit_format_counter+i,
                            n_qubits,
                            total_gates,
                            shots,
                        )
                    )



                with ThreadPoolExecutor() as pool:
                    threads = [pool.submit(self._generate_circuit_image, *arg) for arg in args]
                    
                    rows = []
                    for future in as_completed(threads):
                        try:
                            rows.append(future.result())
                            circuit_format_counter += 1
                            progress.update(1)
                        except Exception as error:
                            print("Error: %s" % error)
                            sys.exit(1)

                    df.append_rows_to_file(rows)
                    checkpoint.index += total_iter
                    checkpoint.save()
                    df.save_df()

    def _generate_circuit_image(
            self,
            index:int,
            n_qubits:int, 
            total_gates:int,
            shots: int,
    ) -> DFRow:
        """Get the statevector of a random circuit"""

        qc = get_random_circuit(n_qubits, total_gates)

        img_path = os.path.join(self._folder, "%d.png" % index)

        # non-interactive backend
        matplotlib.use("Agg")


        gates_per_type_count = {
                1:0, # single qubit gates
                2:0, # two qubit gates
            }
        barriers_count = 0
        total_gates_count = defaultdict(int)

        for inst in qc.data:
            if inst.name == "barrier":
                barriers_count += 1
                continue

            qubits = len(inst.qubits)
            gates_per_type_count[qubits] += 1
            total_gates_count[inst.name] += 1

        drawing = qc.draw(
                "mpl", 
                filename=img_path, 
                fold=-1, 
                scale=SCALE_CIRCUIT_SIZE)
        plt.close(drawing)

        depth = qc.depth()

        with open(img_path, "rb") as file:
            img = Image.open(file)
            width, height = img.size
            file_hash = hashlib.md5(file.read()).hexdigest()
            img.close()
                
        vector = Statevector(qc).probabilities().real.tolist()

        return {
                "index": index,
                "depth": depth,
                "file": img_path,
                "result": json.dumps(vector),
                "hash": file_hash,
                "img_width": width,
                "img_height": height,
                "n_two_qubit_gates": gates_per_type_count.get(2, 0),
                "n_one_qubit_gates": gates_per_type_count.get(1, 0),
                "amount_gates": json.dumps(dict(total_gates_count)),
                "file_size_bytes": os.path.getsize(img_path),
                "n_barriers": barriers_count
            }
