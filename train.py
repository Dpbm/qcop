import torch
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import h5py
import json
import time

from constants import IMAGES_TRAIN, IMAGES_TEST, debug, DATASET_PATH, DATASET_FILE, EPOCHS

class ImagesDataset(Dataset):
    def __init__(self, device, file):
        self._dataset = pd.read_csv(DATASET_FILE)
        self._obj = h5py.File(file, "r")
        self._total = len(self._obj)
        self._device = device

    def __len__(self):
        return self._total

    def _to_tensor(self,loaded_file):
        data = loaded_file.astype(np.float32)
        # data = np.moveaxis(data, -1, 0) # only if the image has 3 channels per pixel instead of 3 distinct channels
        return torch.from_numpy(data).to(self._device)

    def __getitem__(self, index):
        input_data = self._to_tensor(self._obj[f"{index}"][()])
        label = np.array(json.loads(self._dataset.loc[index]["result"]), dtype=np.float16)
        return input_data, label


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # image shape: 3, 1324, 2631
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1)
        self.pool1 = torch.nn.MaxPool2d(4,stride=4)

        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.pool2 = torch.nn.MaxPool2d(3,stride=3)

        self.conv3 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.pool3 = torch.nn.MaxPool2d(3,stride=3)

        self.conv4 = torch.nn.Conv2d(16, 8, kernel_size=2, stride=1)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2)
        
        self.conv5 = torch.nn.Conv2d(8, 4, kernel_size=2, stride=1)
        self.pool5 = torch.nn.MaxPool2d(2, stride=2)
        
        self.fc1 = torch.nn.Linear(544,120)
        self.fc2 = torch.nn.Linear(120,48)
        self.fc3 = torch.nn.Linear(48,32)
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
        
        image = self.conv5(image)
        image = torch.nn.functional.relu(image)
        debug(image.shape)

        image = self.pool5(image)
        debug(image.shape)
        
        image = image.view(image.shape[0], 544)
        debug(image.shape)

        image = self.fc1(image)
        image = torch.nn.functional.relu(image)
        image = self.dropout(image)
        debug(image.shape)

        
        image = self.fc2(image)
        image = torch.nn.functional.relu(image)
        image = self.dropout(image)
        debug(image.shape)

        image = self.fc3(image)
        debug(image.shape)

        image = torch.nn.functional.softmax(image, dim=1)
        debug(image.shape)

        return image

def loss_fn(output,label):
    out_cpu = output.cpu()

    abs_diff = torch.abs(label - out_cpu)
    return torch.sum(abs_diff)

def one_epoch(dataset, opt, model):
    total_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(dataset):
        image,label = data

        opt.zero_grad()

        output = model(image)
        loss = loss_fn(output, label)
        loss.backward()

        opt.step()

        total_loss += loss.item()
        if i % 10 == 0:
            last_loss = total_loss/10
            print(f"batch {i} loss {last_loss}")
            total_loss = 0.0

    return last_loss


def train(device):
    print("running training")

    model = Model().to(device)

    train_data = ImagesDataset(device, IMAGES_TRAIN)
    test_data = ImagesDataset(device, IMAGES_TEST)

    data_loader_train = DataLoader(train_data, batch_size=4, shuffle=False)
    data_loader_test = DataLoader(test_data, batch_size=4, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = 1_000_000

    for epoch in range(EPOCHS):
        print(f"epoch: {epoch}")
        model.train(True)

        loss = one_epoch(data_loader_train, opt, model)

        model.eval()
        
        running_loss = 0.0
        with torch.no_grad():
            for i, dataset in enumerate(data_loader_test):
               data, label = dataset
               output = model(data)
               loss = loss_fn(output,label)
               running_loss += loss
        avg_loss = running_loss/(i+1)
        print(f"AVG: {avg_loss}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_weights_path = f"model_{int(time.time())}"
            torch.save(model.state_dict(), best_model_weights_path)
    return model


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    debug(f"using: {device}")
    model = train(device)
    
    # model = Model().to(device)
    # test = ImagesDataset(device, IMAGES_TRAIN)
    # test_loader = DataLoader(test, batch_size=1, shuffle=False)
    # model.eval()
    #
    # loader_iter = iter(test_loader)
    # image,label = next(loader_iter)
    # print(f"correct: {label}")
    # print(f"eval: {model(image)}")

    ghz = torch.load("ghz.pt", map_location=device)
    ghz = ghz.to(torch.float32)
    result = model(torch.unsqueeze(ghz,0))

    torch.save(result, "ghz-prediction.pt")

if __name__ == "__main__":
    main()
