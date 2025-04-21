import numpy as np
import matplotlib.pyplot as plt
from constants import DEBUG
from math import ceil

class PlotImages:
    @staticmethod
    def plot_filters(images, title=None):
        if not DEBUG:
            return

        images = images.cpu().detach().numpy()[0]

        cols = 4
        rows = ceil(images.shape[0]/cols)

        plt.figure(figsize=(40,20))

        if title is not None:
            plt.title(title)

        plt.axis('off')

        for i,image in enumerate(images):
            ax = plt.subplot(rows,cols,i+1)
            ax.imshow(image)
        plt.show()


def debug(*inp):
    if(not DEBUG):
        return

    print(*inp)