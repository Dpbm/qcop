from qiskit import QuantumCircuit
import numpy as np
from PIL import Image
import torch

from image import transform_image

image_file = "ghz.jpeg"

qc = QuantumCircuit(5)
qc.h(0)
qc.cx(0,1)
qc.cx(1,2)
qc.cx(2,3)
qc.cx(3,4)
qc.measure_all()

qc.draw('mpl', filename=image_file)


with Image.open(image_file) as file:
   tensor = transform_image(file)
   torch.save(tensor, "ghz.pt")
