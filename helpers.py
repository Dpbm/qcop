"""Auxiliary functions and classes"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from constants import DEBUG

class PlotImages:
    """Plot Result images from layer"""

    @staticmethod
    def plot_filters(images:torch.Tensor, title:Optinal[str]=None):
        """Plot every image from current layer"""
        if not DEBUG:
            return

        images = images.cpu().detach().numpy()[0]

        cols = 4
        rows = np.ceil(images.shape[0]/cols)

        plt.figure(figsize=(40,20))

        if title is not None:
            plt.title(title)

        plt.axis('off')

        for i,image in enumerate(images):
            ax = plt.subplot(rows,cols,i+1)
            ax.imshow(image)
        plt.show()


def debug(*inp):
    """Print debug info"""
    if(not DEBUG):
        return

    print("[!] ", *inp)

def should_measure() -> bool:
    """returns a boolean with 50% probability to be True (measure the circuit)"""
    return bool(np.random.randint(0,2))