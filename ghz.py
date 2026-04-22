"""Generate GHZ circuit"""
import os

from qiskit import QuantumCircuit
from PIL import Image
import torch

from utils.image import transform_image
from args.parser import parse_args
from utils.constants import SCALE_CIRCUIT_SIZE
from utils.datatypes import FilePath, Dimensions

from generate.dataset.files import Files

def gen_circuit(n_qubits: int, target_folder: FilePath):
    file_handler = Files(target_folder)
    if os.path.exists(file_handler.ghz_path) and os.path.exists(file_handler.ghz_image_path):
        return

    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for qubit in range(n_qubits - 1):
        qc.cx(qubit, qubit + 1)
    qc.measure_all()
    qc.draw("mpl", filename=file_handler.ghz_image_path, fold=-1, scale=SCALE_CIRCUIT_SIZE)

    with Image.open(file_handler.ghz_image_path) as file:
        tensor = Images._transform_image(file)
        torch.save(tensor, file_handler.ghz_path)

if __name__ == "__main__":
    args = parse_args()
    gen_circuit(args.n_qubits, args.target_folder)
