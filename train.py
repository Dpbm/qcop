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


from constants import IMAGES_TRAIN, IMAGES_TEST, DATASET_PATH, DATASET_FILE, EPOCHS, DEBUG
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

class Downsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=2)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, residual):
        residual = self.conv(residual)
        residual = self.norm(residual)
        return residual

class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, first_stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=first_stride, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.downsample = None if in_channels == out_channels else Downsample(in_channels, out_channels)

    def forward(self, image):
        residual = image if self.downsample is None else self.downsample(image)

        image = self.conv1(image)
        image = self.norm1(image)
        image = F.relu(image)
        
        image = self.conv2(image)
        image = self.norm2(image)
        image += residual
        image = F.relu(image)

        return image

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3,64,7,stride=2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.AvgPool2d(3, stride=2)

        self.out_neurons = 512*9*18
        self.fc1 = nn.Linear(self.out_neurons, 32)

        # image shape: 3, 1324, 2631
        self.blocks = nn.ModuleList([
            Block(64,64),
            Block(64,64),
            Block(64,64),

            Block(64,128,first_stride=2),
            Block(128,128),
            Block(128,128),

            Block(128,256, first_stride=2),
            Block(256,256),
            Block(256,256),

            Block(256, 512, first_stride=2),
            Block(512, 512),
            Block(512, 512),
        ])


    def forward(self, image):
        debug("Input Data: %s"%(str(image.shape)))

        image = F.relu(self.conv1(image))
        image = self.pool1(image)

        debug(image.shape)

        for i,layer in enumerate(self.blocks):
            image = layer(image)

            # PlotImages.plot_filters(image, title="Conv%d"%(i+1))
            debug(image.shape)

        image = self.pool2(image)
        debug(image.shape)
        
        image = image.view(image.shape[0], self.out_neurons)
        debug(image.shape)
        
        out = self.fc1(image)
        out = F.softmax(out, dim=1)
        debug(out.shape)
        return out

def one_epoch(dataset, opt, model, loss_fn, scheduler):
    total_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(dataset):
        image,label = data

        opt.zero_grad()

        output = torch.round(model(image), decimals=3)
        # print("model output: %s"%(str(output)))

        loss = loss_fn(output.log(), label)
        loss.backward()

        opt.step()

        scheduler.step()

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
    
    batch_size=4
    data_loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    data_loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    loss_fn = nn.KLDivLoss(reduction="batchmean")
    lr = 0.1
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(data_loader_train), epochs=EPOCHS)
    best_loss = 1_000_000

    for epoch in range(EPOCHS):
        print("epoch: %d"%(epoch))
        model.train(True)

        loss = one_epoch(data_loader_train, opt, model, loss_fn, scheduler)

        print("opt learning rate: %s; scheduler last learning rate: %s"%(str(opt.param_groups[0]['lr']), str(scheduler.get_last_lr())))

        model.eval()
        
        running_loss = 0.0
        with torch.no_grad():
            for i, dataset in enumerate(data_loader_test):
               data, label = dataset
               output = model(data)
               loss = loss_fn(output.log(),label)
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

    if DEBUG:
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
        print(f"eval: {model(image)}")

        exit()



    # ---- FOR TRAINING THE REAL MODEL ----
    model = train(device)
    model.eval()

    ghz = torch.load("ghz.pt", map_location=device)
    ghz = ghz.to(torch.float32)
    result = model(torch.unsqueeze(ghz,0))
    print("ghz prediction: ", result)
    torch.save(result, "ghz-prediction.pt")

if __name__ == "__main__":
    main()
