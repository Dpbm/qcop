"""Generate GHZ circuit"""
import argparse

from qiskit import QuantumCircuit
from PIL import Image
import torch
from transformers import pipeline
from accelerate import Accelerator
import numpy as np

from utils.image import transform_image
from utils.constants import SCALE_CIRCUIT_SIZE, DEFAULT_NUM_QUBITS, DEFAULT_TARGET_FOLDER, MODEL
from utils.datatypes import FilePath, Dimensions

from generate.dataset.files import Files
from generate.dataset.images import Images

def gen_circuit(n_qubits: int, target_folder: FilePath):
    files_handler = Files(target_folder)

    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for qubit in range(n_qubits - 1):
        qc.cx(qubit, qubit + 1)
    qc.measure_all()
    qc.draw(
            "mpl", 
            filename=files_handler.ghz_image_path, 
            fold=-1, 
            scale=SCALE_CIRCUIT_SIZE)

    
    device = Accelerator().device
    pipe = pipeline(
            task="image-feature-extraction", 
            model=MODEL, device=device)
    
    with Image.open(files_handler.ghz_image_path) as file:
        tensor = Image.fromarray(Images._transform_image(file).numpy())
        embedding = np.array(pipe(tensor), dtype=np.float16)

    print("[*] saving GHZ embedding")
    torch.save(embedding, files_handler.ghz_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-qubits", type=int, default=DEFAULT_NUM_QUBITS)
    parser.add_argument("--target-folder", type=str, required=True)
    args = parser.parse_args()
    gen_circuit(args.n_qubits, args.target_folder)
