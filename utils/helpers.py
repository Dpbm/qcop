"""Auxiliary functions and classes"""

from typing import List, Optional
import random
import os

import torch

from utils.constants import DEBUG, CHECKPOINT_FILE_PREFIX
from utils.datatypes import FilePath


class PlotImages:
    """Plot Result images from layer"""

    @staticmethod
    def plot_filters(images: torch.Tensor, title: Optional[str] = None):
        """Plot every image from current layer"""

        import numpy as np
        import matplotlib.pyplot as plt

        if not DEBUG:
            return

        images = images.cpu().detach().numpy()[0]

        cols = 4
        rows = int(np.ceil(images.shape[0] / cols))

        plt.figure(figsize=(40, 20))

        if title is not None:
            plt.title(title)

        plt.axis("off")

        for i, image in enumerate(images):
            ax = plt.subplot(rows, cols, i + 1)
            ax.axis("off")
            ax.imshow(image)
        plt.show()


def debug(*inp):
    """Print debug info"""
    if not DEBUG:
        return

    print("[!] ", *inp)


def get_measurements(n_qubits: int) -> List[int]:
    """Return a list of qubits to be measured"""
    total_measurements = random.randint(1, n_qubits)
    qubits = list(range(n_qubits))
    return random.sample(qubits, total_measurements)


def get_latest_model_checkpoint(target_folder: FilePath) -> Optional[FilePath]:
    """Returns the path of the lastest checkpoint"""

    if not os.path.exists(target_folder):
        return None

    files = [
        os.path.join(target_folder, file)
        for file in os.listdir(target_folder)
        if file.startswith(CHECKPOINT_FILE_PREFIX)
    ]

    if not files:
        return None

    modification_time = [os.path.getmtime(file) for file in files]

    latest_time = max(modification_time)

    file_index = modification_time.index(latest_time)

    return files[file_index]
