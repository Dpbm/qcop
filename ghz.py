from qiskit import QuantumCircuit
import numpy as np
from PIL import Image
import torch

from image import transform_image
from constants import GHZ_FILE, GHZ_IMAGE_FILE

qc = QuantumCircuit(5)
qc.h(0)
qc.cx(0,1)
qc.cx(1,2)
qc.cx(2,3)
qc.cx(3,4)
qc.measure_all()

qc.draw('mpl', filename=GHZ_IMAGE_FILE)


with Image.open(image_file) as file:
   tensor = transform_image(file)
   torch.save(tensor, GHZ_FILE)
