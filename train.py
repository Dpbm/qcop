import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import h5py

from constants import IMAGES_ARRAY_FILE, debug, DATASET_PATH

class ImagesDataset(Dataset):
    def __init__(self, device):
        self._obj = h5py.File(IMAGES_ARRAY_FILE, "r")
        self._total = len(self._obj)
        self._device = device


    def __len__(self):
        return self._total

    def _to_tensor(self,loaded_file):
        data = loaded_file.astype(np.float32)
        # data = np.moveaxis(data, -1, 0) # only if the image has 3 channels per pixel instead of 3 distinct channels
        return torch.from_numpy(data).to(self._device)

    def __getitem__(self, index):
        return self._to_tensor(self._obj[f"{index}"][()])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # image shape: 3, 1324, 2631
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=5, stride=5)
        self.pool1 = torch.nn.MaxPool2d(4,stride=4)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=3)
        self.pool2 = torch.nn.MaxPool2d(3,stride=3)

        self.conv3 = torch.nn.Conv2d(64, 16, kernel_size=2, stride=1)
        self.pool3 = torch.nn.MaxPool2d(2,stride=2)

        self.conv4 = torch.nn.Conv2d(16, 8, kernel_size=2, stride=1)
        self.pool4 = torch.nn.AvgPool2d(2, stride=1)

        self.fc = torch.nn.Linear(32, 32)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, image):
        debug(f"Input Data: {image.shape}")

        image = self.conv1(image)
        image = torch.nn.functional.relu(image)
        debug(image.shape)

        image = self.pool1(image)
        debug(image.shape)

        image = self.conv2(image)
        image = torch.nn.functional.relu(image)
        debug(image.shape)

        image = self.pool2(image)
        debug(image.shape)
        
        image = self.conv3(image)
        image = torch.nn.functional.relu(image)
        debug(image.shape)

        image = self.pool3(image)
        debug(image.shape)

        image = self.conv4(image)
        image = torch.nn.functional.relu(image)
        debug(image.shape)

        image = self.pool4(image)
        debug(image.shape)

        image = torch.flatten(image)
        debug(image.shape)

        image = self.fc(image)
        image = self.dropout(image)
        image = torch.nn.functional.softmax(image, dim=0)
        debug(image.shape)

        return image



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    debug(f"using: {device}")

    model = Model().to(device)
    data = ImagesDataset(device)

    result = model.forward(data[0])
    print(result)

if __name__ == "__main__":
    main()
