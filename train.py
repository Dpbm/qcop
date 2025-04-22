import torch
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import h5py
import json
import time

from constants import IMAGES_TRAIN, IMAGES_TEST, DATASET_PATH, DATASET_FILE, EPOCHS
from helpers import debug, PlotImages

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
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool1 = torch.nn.AvgPool2d(4,stride=4)

        self.conv3 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.pool2 = torch.nn.AvgPool2d(3,stride=3)

        self.conv4 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.pool3 = torch.nn.MaxPool2d(3,stride=3)

        self.conv5 = torch.nn.Conv2d(16, 8, kernel_size=2, stride=1)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2)

        self.conv6 = torch.nn.Conv2d(8, 4, kernel_size=2, stride=1)
        self.pool5 = torch.nn.MaxPool2d(2, stride=2)

        self.fc1 = torch.nn.Linear(84, 42)
        self.fc2 = torch.nn.Linear(42, 32)
        self.dropout = torch.nn.Dropout(p=0.3)


        self.norm64 = torch.nn.BatchNorm2d(64,affine=False)
        self.norm32 = torch.nn.BatchNorm2d(32,affine=False)
        self.norm16 = torch.nn.BatchNorm2d(16,affine=False)
        self.norm8 = torch.nn.BatchNorm2d(8,affine=False)
        self.norm4 = torch.nn.BatchNorm2d(4,affine=False)

    def forward(self, image):
        debug("Input Data: ", image.shape)

        image = self.conv1(image)
        image = torch.nn.functional.relu(image)
        PlotImages.plot_filters(image, title="Conv1")
        debug(image.shape)

        image = self.norm64(image)

        image = self.conv2(image)
        image = torch.nn.functional.relu(image)
        PlotImages.plot_filters(image, title="Conv2")
        debug(image.shape)
        
        image = self.norm64(image)

        image = self.pool1(image)
        PlotImages.plot_filters(image, title="Pool1")
        debug(image.shape)

        image = self.conv3(image)
        image = torch.nn.functional.relu(image)
        PlotImages.plot_filters(image, title="Conv3")
        debug(image.shape)
        
        image = self.norm32(image)
        
        image = self.pool2(image)
        PlotImages.plot_filters(image, title="Pool2")
        debug(image.shape)
        
        image = self.conv4(image)
        image = torch.nn.functional.relu(image)
        PlotImages.plot_filters(image, title="Conv4")
        debug(image.shape)

        image = self.norm16(image)

        image = self.pool3(image)
        PlotImages.plot_filters(image, title="Pool3")
        debug(image.shape)

        image = self.conv5(image)
        image = torch.nn.functional.relu(image)
        PlotImages.plot_filters(image, title="Conv5")
        debug(image.shape)
        
        image = self.norm8(image)

        image = self.pool4(image)
        PlotImages.plot_filters(image, title="Pool4")
        debug(image.shape)

        image = self.conv6(image)
        image = torch.nn.functional.relu(image)
        PlotImages.plot_filters(image, title="Conv6")
        debug(image.shape)
        
        image = self.norm4(image)

        image = self.pool5(image)
        PlotImages.plot_filters(image, title="Pool5")
        debug(image.shape)

        image = image.view(image.shape[0], 84)
        debug(image.shape)

        image = self.fc1(image)
        image = torch.nn.functional.relu(image)
        image = self.dropout(image)
        debug(image.shape)
        
        image = self.fc2(image)
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

    # ---- FOR MANUAL TESTS ----- 
    # model = Model().to(device)
    # test = ImagesDataset(device, IMAGES_TRAIN)
    # test_loader = DataLoader(test, batch_size=1, shuffle=False)
    # model.eval()
    #
    # loader_iter = iter(test_loader)
    # image,label = next(loader_iter)
    # print(f"correct: {label}")
    # print(f"eval: {model(image)}")

    # ---- FOR TRAINING THE REAL MODEL ----
    model = train(device)
    mode.eval()

    ghz = torch.load("ghz.pt", map_location=device)
    ghz = ghz.to(torch.float32)
    result = model(torch.unsqueeze(ghz,0))
    print("ghz prediction: ", result)
    torch.save(result, "ghz-prediction.pt")

if __name__ == "__main__":
    main()
