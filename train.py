import torch
import torch.nn as nn
import torch.nn.functional as F

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
        label = torch.from_numpy(np.array(json.loads(self._dataset.loc[index]["result"]), dtype=np.float16)).to(self._device)
        return input_data, label


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # image shape: 3, 1324, 2631


        self.image_layers = nn.ModuleList([
                nn.Conv2d(3,64,3, stride=1),
                nn.Conv2d(64,64,3, stride=1),
                nn.MaxPool2d(2, stride=2),


                nn.Conv2d(64,128,3, stride=1),
                nn.Conv2d(128,128,3, stride=1),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(128,256,3, stride=1),
                nn.Conv2d(256,256,3, stride=1),
                nn.Conv2d(256,256,3, stride=1),
                nn.Conv2d(256,256,3, stride=1),
                nn.MaxPool2d(2, stride=2),


                nn.Conv2d(256,512,3, stride=1),
                nn.Conv2d(512,512,3, stride=1),
                nn.Conv2d(512,512,3, stride=1),
                nn.Conv2d(512,512,3, stride=1),
                nn.MaxPool2d(2, stride=2),


                nn.Conv2d(512,256,3, stride=1),
                nn.Conv2d(256,256,3, stride=1),
                nn.Conv2d(256,256,3, stride=1),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(256,128,3, stride=1),
                nn.Conv2d(128,128,3, stride=1),
                nn.Conv2d(128,128,3, stride=1),
                nn.MaxPool2d(2, stride=2),

        ])

        #self.pool_after = {1,3,7,11,15,18}

        self.fc1 = nn.Linear(4608,4608)
        self.fc2 = nn.Linear(4608,2**5)
        self.dropout = nn.Dropout(p=0.7)



    def forward(self, image):
        debug("Input Data: %s"%(str(image.shape)))

        for i,layer in enumerate(self.image_layers):
            image = layer(image)

            if isinstance(layer, nn.Conv2d):
              image = F.relu(image)

            # PlotImages.plot_filters(image, title="Conv%d"%(i+1))
            debug(image.shape)

        image = image.view(image.shape[0], 4608)
        debug(image.shape)

        image = F.relu(self.fc1(image))
        debug(image.shape)

        image = self.dropout(image)

        image = F.softmax(F.relu(self.fc2(image)), dim=0)
        debug(image.shape)

        return image

def one_epoch(dataset, opt, model, loss_fn):
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
            print("batch %d loss %f"%(i, last_loss))
            total_loss = 0.0

    return last_loss


def train(device):
    print("running training")

    model = Model().to(device)

    train_data = ImagesDataset(device, IMAGES_TRAIN)
    test_data = ImagesDataset(device, IMAGES_TEST)

    data_loader_train = DataLoader(train_data, batch_size=4, shuffle=False)
    data_loader_test = DataLoader(test_data, batch_size=4, shuffle=False)

    loss_fn = nn.KLDivLoss(reduction="batchmean")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = 1_000_000

    for epoch in range(EPOCHS):
        print("epoch: %d"%(epoch))
        model.train(True)

        loss = one_epoch(data_loader_train, opt, model, loss_fn)

        model.eval()
        
        running_loss = 0.0
        with torch.no_grad():
            for i, dataset in enumerate(data_loader_test):
               data, label = dataset
               output = model(data)
               loss = loss_fn(output,label)
               running_loss += loss
        avg_loss = running_loss/(i+1)
        print("AVG: %f"%(avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_weights_path = "model_%d"%(int(time.time()))
            torch.save(model.state_dict(), best_model_weights_path)
    return model


def main():
    default_cuda_device = "cuda"
    device = default_cuda_device if torch.cuda.is_available() else 'cpu'
    
    gc.collect()
    if device == default_cuda_device:
        torch.cuda.empty_cache()

    debug("using: %s"%(device))

    #---- FOR MANUAL TESTS ----- 
    model = Model().to(device)
    test = ImagesDataset(device, IMAGES_TRAIN)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    model.train(False)
    model.eval()

    loader_iter = iter(test_loader)
    image,label = next(loader_iter)

    print(f"correct: {label}")
    model(image)
    # print(f"eval: {model(image)}")

    # ---- FOR TRAINING THE REAL MODEL ----
    # model = train(device)
    # model.eval()
    #
    # ghz = torch.load("ghz.pt", map_location=device)
    # ghz = ghz.to(torch.float32)
    # result = model(torch.unsqueeze(ghz,0))
    # print("ghz prediction: ", result)
    # torch.save(result, "ghz-prediction.pt")

if __name__ == "__main__":
    main()
