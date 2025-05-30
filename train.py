"""Train ResNet based model."""

from typing import Optional, Tuple
from collections import OrderedDict
import sys
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader

import polars as pl 
import numpy as np
import h5py

from constants import (
    IMAGES_H5_FILE, 
    DATASET_PATH, 
    DATASET_FILE, 
    EPOCHS, 
    DEBUG, 
    BATCH_SIZE,
    GHZ_FILE,
    GHZ_PRED_FILE
)
from helpers import debug, PlotImages
from colors import Colors

StateDict = OrderedDict
Device = str
FilePath = str
Channels = int

class ImagesDataset(Dataset):
    """Dataset class for handling batches and data itself"""

    def __init__(self, device:Device, file:FilePath, dataset_file:FilePath):
        self._dataset = pl.read_csv(dataset_file)
        self._obj = h5py.File(file, "r")
        self._total = len(self._obj)
        self._device = device

    def __len__(self) -> int:
        """return the amount of files"""
        return self._total

    def _to_tensor(self, loaded_file:np.array) -> torch.Tensor:
        """auxiliary method to map an np.array to tensor in the correct device and data type"""

        data = loaded_file.astype(np.float32)
        # data = np.moveaxis(data, -1, 0) # only if the image has 3 channels per pixel instead of 3 distinct channels
        return torch.from_numpy(data).to(self._device)

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an especific value inside the dataset with its label"""
        input_data = self._to_tensor(self._obj[f"{index}"][()])
        label = torch.from_numpy(np.array(json.loads(self._dataset.row(index, named=True)["result"]), dtype=np.float16)).to(self._device)
        return input_data, label

class Downsample(torch.nn.Module):
    """
    Downsample block. Used to normalize the output of a block to the input of the next block. 
    Useful when two blocks  have different input channels
    """

    def __init__(self, in_channels:Channels, out_channels:Channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=2)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, residual:torch.Tensor):
        """Apply the normalization method"""
        residual = self.conv(residual)
        residual = self.norm(residual)
        return residual

class Block(torch.nn.Module):
    """A ResNet block"""

    def __init__(self, in_channels:Channels, out_channels:Channels, first_stride:int=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=first_stride, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.downsample = None if in_channels == out_channels else Downsample(in_channels, out_channels)

    def forward(self, image:torch.Tensor) -> torch.Tensor:
        """Apply block transformations on the current image"""
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
    """The model architecture itself"""

    def __init__(self):
        super(Model, self).__init__()
        
        self.conv1 = nn.Conv2d(3,64,7,stride=2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.AvgPool2d(3, stride=2)

        self.out_neurons = 512*9*18
        self.fc1 = nn.Linear(self.out_neurons, 32)

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


    def forward(self, image:torch.Tensor) -> torch.Tensor:
        """Apply all transformations onto the input image"""

        debug("Input Data: %s"%(str(image.shape)))
        PlotImages.plot_filters(image, title="Input Image")

        image = F.relu(self.conv1(image))
        image = self.pool1(image)

        debug(image.shape)

        for i,layer in enumerate(self.blocks):
            image = layer(image)

            PlotImages.plot_filters(image, title="Conv%d"%(i+1))

            debug(image.shape)

        image = self.pool2(image)
        debug(image.shape)
        
        image = image.view(image.shape[0], self.out_neurons)
        debug(image.shape)
        
        out = self.fc1(image)
        out = F.softmax(out, dim=1)
        debug(out.shape)
        return out

    def save(self):
        """Save model weights."""
        path = "model_%s"%(time.ctime())
        torch.save(self.state_dict(), path)

class Checkpoint:
    """An auxiliary class to handle checkpoints"""

    def __init__(self, path:Optional[FilePath]):
        self._path = path
        self._data = {}

    def load(self):
        """Load check point if a path was provided"""
        if self._path is None:
            print("%sNo Checkpoint was provided!%s"%(Colors.YELLOWFG,Colors.ENDC))
            return

        print("%sLoading checkpoint from: %s...%s"%(Colors.MAGENTABG, self._path, Colors.ENDC))
        self._data = torch.load(self._path)

    @property
    def model(self) -> Optional[StateDict]:
        """return the model weights"""
        return self._data.get("model")

    @property
    def optimizer(self) -> Optional[StateDict]:
        """Get optimizer parameters"""
        return self._data.get("optimizer")

    @property
    def scheduler(self) -> Optional[StateDict]:
        """Get Scheduler parameters"""
        return self._data.get("scheduler")

    @property
    def epoch(self) -> int:
        """Get checkpoint epoch"""
        return self._data.get("epoch") or 0

    @staticmethod
    def save(epoch:int, model:StateDict, optimizer:StateDict, scheduler:StateDict):
        """Save checkpoint data"""
        path = "checkpoint_%s.pth"%(time.ctime())
        print("%sSaving checkpoint at: %s...%s"%(Colors.MAGENTABG,path,Colors.ENDC))
        checkpoint = {
            'epoch': epoch,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler
        }
        torch.save(checkpoint, path)



def one_epoch(
    dataset:DataLoader, 
    opt:torch.optim.Optimizer, 
    model:Model, 
    loss_fn:nn.modules.loss._Loss, 
    scheduler:torch.optim.lr_scheduler.LRScheduler):
    """Run one epoch on data"""

    total_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(dataset):
        image,label = data

        opt.zero_grad()

        output = torch.round(model(image), decimals=3)

        if DEBUG:
            print("%smodel output: %s%S"%(colors.YELLOWFG, str(output), Colors.ENDC))

        loss = loss_fn(output.log(), label) # the loss function requires our data to be in log format

        loss.backward()

        opt.step()
        scheduler.step()

        total_loss += loss.item()
        total_loss_steps = 50
        if i > 0 and i % total_loss_steps == 0:
            last_loss = total_loss/total_loss_steps
            print("%sbatch %d loss %f%s"%(Colors.MAGENTAFG,i, last_loss,Colors.ENDC))
            total_loss = 0.0

    return last_loss

def get_model(device:Device, state:StateDict=None):
    """
    Prepare model. 
    It creates a model, load into a given device and load weights if a state
    was provided.
    """
    model = Model().to(device)
    if state:
        model.load_state_dict(state)

    return model


def train(device:Device, checkpoint:Checkpoint):
    """Train model"""

    print("%sRunning Training%s"%(Colors.MAGENTABG, Colors.ENDC))

    model = get_model(device, checkpoint.model)

    train_data = ImagesDataset(device, IMAGES_TRAIN, DATASET_FILE)
    test_data = ImagesDataset(device, IMAGES_TEST, DATASET_FILE)
    
    data_loader_train = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    data_loader_test = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = nn.KLDivLoss(reduction="batchmean")
    lr = 0.1
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(data_loader_train), epochs=EPOCHS)
    best_loss = 1_000_000


    if checkpoint.optimizer:
        opt.load_state_dict(checkpoint.optimizer)
    if checkpoint.scheduler:
        scheduler.load_state_dict(checkpoint.scheduler)

    for epoch in range(checkpoint.epoch+1, EPOCHS):
        print("%sEpoch: %d%s"%(Colors.YELLOWFG, epoch, Colors.ENDC))
        model.train(True)

        loss = one_epoch(data_loader_train, opt, model, loss_fn, scheduler)

        print("%sopt learning rate: %s; scheduler last learning rate: %s%s"%(Colors.YELLOWFG, str(opt.param_groups[0]['lr']), str(scheduler.get_last_lr()), Colors.ENDC))

        model.eval()
        
        running_loss = 0.0
        with torch.no_grad():
            for i, dataset in enumerate(data_loader_test):
               data, label = dataset
               output = model(data)
               loss = loss_fn(output.log(),label)
               running_loss += loss

        avg_loss = running_loss/(i+1)
        print("%sAVG: %f%s"%(Colors.MAGENTAFG, avg_loss, Colors.ENDC))

        if avg_loss < best_loss:
            best_loss = avg_loss
            print("%sBest loss: %d%s"%(Colors.GREENBG, best_loss, Colors.ENDC))
        
        
        # save a checkpoint after every epoch
        Checkpoint.save(
            epoch,
            model.state_dict(),
            opt.state_dict(),
            scheduler.state_dict()
        )

    return model


def main():
    """
    Setup environment and start experiments.
    """

    default_cuda_device = "cuda"
    device = default_cuda_device if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args(sys.argv[1:])
    checkpoint_path = args.checkpoint if args.checkpoint else None

    checkpoint = Checkpoint(checkpoint_path)
    checkpoint.load()

    if device == default_cuda_device:
        torch.cuda.empty_cache()

    print("%susing: %s device %s"%(Colors.GREENBG, device, Colors.ENDC))

    if DEBUG:
        #---- FOR MANUAL TESTS ----- 
        model = get_model(device, checkpoint.model)
        test = ImagesDataset(device, IMAGES_TRAIN, DATASET_FILE)
        test_loader = DataLoader(test, batch_size=1, shuffle=True)
        model.train(False)
        model.eval()

        loader_iter = iter(test_loader)
        image,label = next(loader_iter)

        print("%scorrect: %s%s"%(Colors.GREENBG,str(label), Colors.ENDC))
        model(image)
        print("%seval: %s%s"%(Colors.GREENBG, str(model(image)), Colors.ENDC))

        sys.exit(0)

    # ---- FOR TRAINING THE REAL MODEL ----
    model = train(device,checkpoint)
    model.save()
    model.eval()

    ghz = torch.load(GHZ_FILE, map_location=device)
    ghz = ghz.to(torch.float32)
    result = model(torch.unsqueeze(ghz,0))
    print("%sghz prediction: %s%s"%(Colors.GREENBG, str(result), Colors.ENDC))
    torch.save(result, GHZ_PRED_FILE)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
