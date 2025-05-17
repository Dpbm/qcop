"""Generate GHZ circuit"""

from qiskit import QuantumCircuit
import numpy as np
from PIL import Image
import torch

from image import transform_image
from constants import GHZ_FILE, GHZ_IMAGE_FILE, N_QUBITS
from helpers import should_measure

qc = QuantumCircuit(N_QUBITS)
qc.h(0)

for qubit in range(N_QUBITS-1):
   qc.cx(qubit, qubit+1)

if should_measure():
   qc.measure_all()

qc.draw('mpl', filename=GHZ_IMAGE_FILE)

with Image.open(GHZ_IMAGE_FILE) as file:
   tensor = transform_image(file)
   torch.save(tensor, GHZ_FILE)
