"""Train ResNet based model."""

from typing import Optional, Tuple, List, Any, TypedDict, Literal
from collections import OrderedDict
import sys
import os
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import polars as pl
import numpy as np
import h5py

from utils.constants import (
    DEBUG,
    MODEL_FILE_PREFIX,
    CHECKPOINT_FILE_PREFIX,
    dataset_file,
    images_h5_file,
    ghz_file,
    ghz_pred_file,
    output_plot_file,
    history_file,
)
from utils.helpers import debug, PlotImages
from utils.colors import Colors
from utils.datatypes import FilePath, df_schema
from args.parser import parse_args, Arguments

StateDict = OrderedDict
Device = str
Channels = int


class CheckpointData(TypedDict):
    """Data holded inside a Checkpoint"""

    epoch: int
    model: StateDict
    optimizer: StateDict
    scheduler: StateDict


class HistoryData(TypedDict):
    """Data holded inside a History object"""

    test: List[float]
    rmse: List[float]


class ImagesDataset(Dataset):
    """Dataset class for handling batches and data itself"""

    def __init__(
        self,
        device: Device,
        file: h5py.File,
        dataset: pl.LazyFrame,
        total_images: int,
        pivot: int,
    ):
        self._dataset = dataset
        self._obj = file
        self._pivot = pivot
        self._device = device
        self._total_images = total_images

    def __len__(self) -> int:
        """return the amount of files"""
        return self._total_images

    def _to_tensor(self, loaded_file: np.array) -> torch.Tensor:
        """auxiliary method to map an np.array to tensor in the correct device and data type"""

        data = loaded_file.astype(np.float32)
        # data = np.moveaxis(data, -1, 0) # only if the image has 3 channels per pixel instead of 3 distinct channels
        return torch.from_numpy(data).to(self._device)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an especific value inside the dataset with its label"""
        shifted_index = index + self._pivot
        input_data = self._to_tensor(self._obj[f"{shifted_index}"][()])

        dataset_data = self._dataset.slice(length=1, offset=index).collect()
        result = json.loads(dataset_data["result"].item(0))

        label = torch.from_numpy(np.array(result, dtype=np.float16)).to(self._device)
        return input_data, label


class Downsample(torch.nn.Module):
    """
    Downsample block. Used to normalize the output of a block to the input of the next block.
    Useful when two blocks  have different input channels
    """

    def __init__(self, in_channels: Channels, out_channels: Channels):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=2)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, residual: torch.Tensor):
        """Apply the normalization method"""
        residual = self.conv(residual)
        residual = self.norm(residual)
        return residual


class Block(torch.nn.Module):
    """A ResNet block"""

    def __init__(
        self, in_channels: Channels, out_channels: Channels, first_stride: int = 1
    ):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=first_stride, padding=1
        )
        self.norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.downsample = (
            None
            if in_channels == out_channels
            else Downsample(in_channels, out_channels)
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
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

        self.conv1 = nn.Conv2d(3, 64, 7, stride=2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.AvgPool2d(3, stride=2)

        self.out_neurons = 512 * 6 * 7
        self.fc1 = nn.Linear(self.out_neurons, 32)

        self.blocks = nn.ModuleList(
            [
                Block(64, 64),
                Block(64, 64),
                Block(64, 64),
                Block(64, 128, first_stride=2),
                Block(128, 128),
                Block(128, 128),
                Block(128, 256, first_stride=2),
                Block(256, 256),
                Block(256, 256),
                Block(256, 512, first_stride=2),
                Block(512, 512),
                Block(512, 512),
            ]
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply all transformations onto the input image"""

        debug("Input Data: %s" % (str(image.shape)))
        PlotImages.plot_filters(image, title="Input Image")

        image = F.relu(self.conv1(image))
        image = self.pool1(image)

        debug(image.shape)

        for i, layer in enumerate(self.blocks):
            image = layer(image)

            PlotImages.plot_filters(image, title="Conv%d" % (i + 1))

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
        path = "%s%s" % (MODEL_FILE_PREFIX, time.ctime())
        torch.save(self.state_dict(), path)


class Checkpoint:
    """An auxiliary class to handle checkpoints"""

    def __init__(self, path: Optional[FilePath]):
        self._path = path
        self._data: CheckpointData = {}  # type: ignore

    def load(self):
        """Load check point if a path was provided"""
        if self._path is None:
            print("%sNo Checkpoint was provided!%s" % (Colors.YELLOWFG, Colors.ENDC))
            return

        print(
            "%sLoading checkpoint from: %s...%s"
            % (Colors.MAGENTABG, self._path, Colors.ENDC)
        )
        self._data = torch.load(self._path)

    def was_provided(self) -> bool:
        """Returns true if user has provided a checkpoint to be loaded"""
        return self._path is not None

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
    def save(
        folder: FilePath,
        epoch: int,
        model: StateDict,
        optimizer: StateDict,
        scheduler: StateDict,
    ):
        """Save checkpoint data"""
        path = os.path.join(folder, "%s%s.pth" % (CHECKPOINT_FILE_PREFIX, time.ctime()))
        print("%sSaving checkpoint at: %s...%s" % (Colors.MAGENTABG, path, Colors.ENDC))
        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
        torch.save(checkpoint, path)


class History:
    """Class in charge for saving the progress of training the model"""

    def __init__(self, history_file: FilePath):
        self._data: HistoryData = {"test": [], "rmse": []}
        self._file_path = history_file

    def _add_to_key(self, key: Literal["test", "rmse"], value: Any):
        """add value to history"""
        self._data[key].append(value)

    def add_test_progress(self, value: float):
        """Updated test loss progress"""
        self._add_to_key("test", value)

    def add_rmse_progress(self, value: float):
        """Updated rmse progress"""
        self._add_to_key("rmse", value)

    def save(self):
        """Save dict to json file"""

        print(
            "%sSaving history file: %s...%s"
            % (Colors.GREENBG, self._file_path, Colors.ENDC)
        )

        with open(self._file_path, "w") as file:
            json.dump(self._data, file)

    def load(self):
        """Load json into dict"""

        if not os.path.exists(self._file_path):
            return

        print(
            "%sLoading history file: %s...%s"
            % (Colors.YELLOWFG, self._file_path, Colors.ENDC)
        )

        with open(self._file_path, "r") as file:
            self._data = json.load(file)

    def plot(self, output_file: FilePath):
        """Plot history"""

        import matplotlib.pyplot as plt

        x = len(self._data["test"])

        plt.plot(x, self._data["test"])
        plt.plot(x, self._data["rmse"])
        plt.grid()
        plt.title("Training progress")
        plt.xlabel("Epochs")
        plt.legend(["test", "rmse"])
        plt.savefig(output_file, bbox_inches="tight")
        plt.show()


def one_epoch(
    dataset: DataLoader,
    opt: torch.optim.Optimizer,
    model: Model,
    loss_fn: nn.modules.loss._Loss,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
):
    """Run one epoch on data"""

    total_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(dataset):
        image, label = data

        opt.zero_grad()

        output = torch.round(model(image), decimals=3)

        if DEBUG:
            print("%smodel output: %s%s" % (Colors.YELLOWFG, str(output), Colors.ENDC))

        loss = loss_fn(
            output.log(), label
        )  # the loss function requires our data to be in log format

        loss.backward()

        opt.step()
        scheduler.step()

        total_loss += loss.item()
        total_loss_steps = 50
        if i > 0 and i % total_loss_steps == 0:
            last_loss = total_loss / total_loss_steps
            print(
                "%sbatch %d loss %f%s" % (Colors.MAGENTAFG, i, last_loss, Colors.ENDC)
            )
            total_loss = 0.0

    return last_loss


def get_model(device: Device, state: Optional[StateDict] = None) -> torch.jit.ScriptModule:
    """
    Prepare script model (JIT).
    It creates a model, load into a given device and load weights if a state
    was provided.
    """
    model = Model().to(device)
    if state:
        model.load_state_dict(state)

    script_model = torch.jit.script(model)
    return script_model


def train(
    device: Device,
    checkpoint: Checkpoint,
    target_folder: FilePath,
    train_percentage: float,
    test_percentage: float,
    batch_size: int,
    epochs: int,
):
    """Train model"""

    print("%sRunning Training%s" % (Colors.MAGENTABG, Colors.ENDC))

    model = get_model(device, checkpoint.model)

    dataset = pl.scan_csv(dataset_file(target_folder), schema=df_schema)

    h5_file = h5py.File(images_h5_file(target_folder), "r")
    total_images = len(h5_file)

    # the amount of images for TRAINING, TESTING AND EVALUATING
    max_train = int(train_percentage * total_images)
    max_test = int(test_percentage * total_images)
    max_eval = int((1 - (train_percentage + test_percentage)) * total_images)

    # the starting index for images per dataset split
    pivot_training_index = 0
    pivot_testing_index = max_train - 1
    pivot_evaluating_index = total_images - max_eval - 1

    # split dataset (csv file) into sections
    train_dataset = dataset.slice(length=max_train, offset=pivot_training_index)
    test_dataset = dataset.slice(length=max_test, offset=pivot_testing_index)
    eval_dataset = dataset.slice(length=max_eval, offset=pivot_evaluating_index)

    train_data = ImagesDataset(
        device, h5_file, train_dataset, max_train, pivot_training_index
    )
    test_data = ImagesDataset(
        device, h5_file, test_dataset, max_test, pivot_testing_index
    )
    eval_data = ImagesDataset(
        device, h5_file, eval_dataset, max_eval, pivot_evaluating_index
    )

    data_loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    data_loader_test = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    data_loader_eval = DataLoader(eval_data, batch_size=batch_size, shuffle=True)

    history = History(history_file(target_folder))
    if checkpoint.was_provided():
        history.load()

    loss_fn = nn.KLDivLoss(reduction="batchmean")
    lr = 0.1
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=len(data_loader_train), epochs=epochs
    )
    best_loss = 1_000_000.0

    if checkpoint.optimizer:
        opt.load_state_dict(checkpoint.optimizer)
    if checkpoint.scheduler:
        scheduler.load_state_dict(checkpoint.scheduler)

    for epoch in range(checkpoint.epoch + 1, epochs):
        print("%sEpoch: %d%s" % (Colors.YELLOWFG, epoch, Colors.ENDC))
        model.train(True)

        loss = one_epoch(data_loader_train, opt, model, loss_fn, scheduler)

        print(
            "%sopt learning rate: %s; scheduler last learning rate: %s%s"
            % (
                Colors.YELLOWFG,
                str(opt.param_groups[0]["lr"]),
                str(scheduler.get_last_lr()),
                Colors.ENDC,
            )
        )

        model.eval()

        running_loss = 0.0
        targets = []
        outputs = []
        with torch.no_grad():
            for i, dataset in enumerate(data_loader_test):
                data, label = dataset
                output = model(data)

                outputs.append(output)
                targets.append(torch.Tensor(label))

                loss = loss_fn(output.log(), label)
                running_loss += loss

        avg_loss = running_loss / max_test
        rmse = get_RMSE(targets, outputs)

        history.add_test_progress(float(avg_loss))
        history.add_rmse_progress(float(rmse))

        history.save()

        print("%sAVG loss Test: %f%s" % (Colors.MAGENTAFG, avg_loss, Colors.ENDC))
        print("%sRMSE: %f%s" % (Colors.MAGENTAFG, rmse, Colors.ENDC))

        if avg_loss < best_loss:
            best_loss = avg_loss
            print("%sBest loss: %f%s" % (Colors.GREENBG, best_loss, Colors.ENDC))

        # save a checkpoint after every epoch
        Checkpoint.save(
            target_folder,
            epoch,
            model.state_dict(),
            opt.state_dict(),
            scheduler.state_dict(),
        )

    eval_loss = 0.0
    targets = []
    outputs = []
    with torch.no_grad():
        for i, dataset in enumerate(data_loader_eval):
            data, label = dataset
            output = model(data)

            outputs.append(output)
            targets.append(torch.Tensor(label))

            loss = loss_fn(output.log(), label)
            eval_loss += loss

    avg_loss = eval_loss / max_eval
    rmse = get_RMSE(targets, outputs)

    print("%sAVG loss Eval: %f%s" % (Colors.MAGENTAFG, avg_loss, Colors.ENDC))
    print("%sRMSE Eval: %f%s" % (Colors.MAGENTAFG, rmse, Colors.ENDC))

    h5_file.close()

    history.plot(output_plot_file(target_folder))

    return model


def get_device() -> Device:
    """
    Return the device to be used with pytorch
    """

    default_cuda_device = "cuda"
    device = default_cuda_device if torch.cuda.is_available() else "cpu"

    if device == default_cuda_device:
        torch.cuda.empty_cache()

    print("%susing: %s device %s" % (Colors.GREENBG, device, Colors.ENDC))

    return device


def get_RMSE(targets: List[torch.Tensor], outputs: List[torch.Tensor]):
    """Root Mean Squared Error."""
    diff_sum = sum(
        [(torch.sum(target - output)) ** 2 for target, output in zip(targets, outputs)]
    )
    n = len(targets)
    return torch.sqrt((1 / n) * diff_sum)


def run_debug_experiemnt(args: Arguments):
    """
    Run Manual tests.
    """
    device = get_device()

    checkpoint = Checkpoint(args.checkpoint)
    checkpoint.load()

    dataset = pl.scan_csv(dataset_file(args.target_folder), schema=df_schema)
    h5_file = h5py.File(images_h5_file(args.target_folder), "r")

    model = get_model(device, checkpoint.model)
    test = ImagesDataset(device, h5_file, dataset, len(h5_file), 0)
    test_loader = DataLoader(test, batch_size=1, shuffle=True)
    model.train(False)
    model.eval()

    loader_iter = iter(test_loader)
    image, label = next(loader_iter)

    print("%scorrect: %s%s" % (Colors.GREENBG, str(label), Colors.ENDC))
    model(image)
    print("%seval: %s%s" % (Colors.GREENBG, str(model(image)), Colors.ENDC))


def setup_and_run_training(args: Arguments):
    """
    Setup and run a training task.
    """

    device = get_device()

    checkpoint = Checkpoint(args.checkpoint)
    checkpoint.load()

    # TODO: USE PYTORCH SCRIPTING (JIT)
    model = train(
        device,
        checkpoint,
        args.target_folder,
        args.train_size,
        args.test_size,
        args.batch_size,
        args.epochs,
    )
    model.save()  # save best model
    model.eval()

    ghz = torch.load(ghz_file(args.target_folder), map_location=device)
    ghz = ghz.to(torch.float32)
    result = model(torch.unsqueeze(ghz, 0))
    print("%sghz prediction: %s%s" % (Colors.GREENBG, str(result), Colors.ENDC))
    torch.save(result, ghz_pred_file(args.target_folder))


def main():
    """
    Setup environment and start experiments.
    """

    args = parse_args()

    if DEBUG:
        run_debug_experiemnt(args)
        return

    setup_and_run_training(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
