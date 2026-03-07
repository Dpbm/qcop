import sys
import os
from typing import Optional, Tuple
from pathlib import Path
import re

from PIL import Image
    
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
    
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from utils.constants import DEFAULT_MAX_TOTAL_GATES, DEFAULT_NUM_QUBITS
from random_circuit import generate_circuit, generate_circuit_worst_case

CONSTANTS_FILE_PATH = os.path.join(base_path, "utils", "constants.py")
TESTS_FILE_PATH = os.path.join(base_path, "tests")
SCALE_SIZE = 0.5

def draw(qc:QuantumCircuit, ax:Optional[plt.Axes]=None) -> plt.Figure:
    """Draw circuit with tweaks."""
    return qc.draw("mpl", scale=SCALE_SIZE, fold=-1, ax=ax)

if __name__ == "__main__":
    print("Testing the min size of a circuit image...")
    qc1 = generate_circuit(DEFAULT_NUM_QUBITS, 0)
    small_fig_path = os.path.join(TESTS_FILE_PATH, "small.png")
    img1 = draw(qc1)
    img1.savefig(small_fig_path, bbox_inches="tight")
    plt.close(img1)
    min_width,min_height = Image.open(small_fig_path).size
    print(f"Size -> {min_width}x{min_height}px")

    print("Testing the max size of a circuit image...")
    qc2 = generate_circuit_worst_case(DEFAULT_NUM_QUBITS, DEFAULT_MAX_TOTAL_GATES)
    large_fig_path = os.path.join(TESTS_FILE_PATH, "large.png")
    img2 = draw(qc2)
    img2.savefig(large_fig_path, bbox_inches="tight")
    plt.close(img2)
    max_width,max_height = Image.open(large_fig_path).size
    print(f"Size -> {max_width}x{max_height}px")

    # fig,ax = plt.subplots(2)
    # draw(qc1,ax[0])
    # draw(qc2,ax[1])
    # plt.show()
    

    with open(CONSTANTS_FILE_PATH, "r+", encoding="utf-8") as file:
        read = file.read()

        pattern_dim = r'DEFAULT_NEW_DIM *= *\( *[0-9]{2,} *, *[0-9]{2,} *\)'
        pattern_scale = r'SCALE_CIRCUIT_SIZE *= *0\.[1-9]'

        new_dim_data = f"DEFAULT_NEW_DIM = ({max_width}, {max_height})"
        new_scale_factor = f"SCALE_CIRCUIT_SIZE = {SCALE_SIZE}"

        new_data = read

        if re.search(pattern_dim, new_data):
            new_data = re.sub(pattern_dim, new_dim_data, new_data)
        else:
            new_data += new_dim_data

        if re.search(pattern_scale, new_data):
            new_data = re.sub(pattern_scale, new_scale_factor, new_data)
        else:
            new_data += new_scale_factor
    
    with open(CONSTANTS_FILE_PATH, "w", encoding="utf-8") as file:
        file.write(new_data)
