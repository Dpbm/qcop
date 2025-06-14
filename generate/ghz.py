"""Generate GHZ circuit"""

from qiskit import QuantumCircuit
from PIL import Image
import torch

from utils.image import transform_image
from args.parser import parse_args
from utils.constants import ghz_image_file, ghz_file
from utils.datatypes import FilePath, Dimensions


def gen_circuit(n_qubits: int, target_folder: FilePath, new_dim: Dimensions):
    qc = QuantumCircuit(n_qubits)
    qc.h(0)

    for qubit in range(n_qubits - 1):
        qc.cx(qubit, qubit + 1)

    qc.measure_all()

    ghz_image_path = ghz_image_file(target_folder)
    qc.draw("mpl", filename=ghz_image_path)

    with Image.open(ghz_image_path) as file:
        width, height = new_dim
        tensor = transform_image(file, width, height)
        torch.save(tensor, ghz_file(target_folder))


if __name__ == "__main__":
    args = parse_args()
    gen_circuit(args.n_qubits, args.target_folder, args.new_image_dim)
