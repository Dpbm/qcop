import torch
import numpy as np
import matplotlib.pyplot as plt

from constants import NPY_IMAGES_FILE

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.input_channels = 3
        self.output_neurons = 2**5 # 2**5 bitstrings combinations

        self.conv1 = torch.nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=3)
        self.pool1 = torch.nn.MaxPool2d(4,stride=3)

    def forward(self, image):
        image = self.conv1(image)
        return self.pool1(image)



def main():

    model = Model()
    with open(NPY_IMAGES_FILE, "rb") as npy:
        data = np.load(npy).astype(np.float32)
        data = np.moveaxis(data, -1, 0)
        
        plt.imshow(data[0])
        plt.show()
        plt.imshow(data[1])
        plt.show()
        plt.imshow(data[2])
        plt.show()

        torch_data = torch.from_numpy(data)
        print(model.forward(torch_data).shape)



if __name__ == "__main__":
    main()
