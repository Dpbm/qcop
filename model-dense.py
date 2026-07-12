import argparse
from functools import reduce
import json

from tqdm import trange
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, random_split

from utils.constants import DEFAULT_RANDOM_SEED

from generate.dataset.files import Files
from dataset import Data

def main():
    parser = argparse.ArgumentParser(description="Train Dense Model")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--target-folder", type=str, required=True)
    parser.add_argument("--train-percentage", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    files_handler = Files(args.target_folder)
    
    with open(files_handler.embeddings_shape_path, "r") as shape_file:
        shape = list(json.load(shape_file))
        total_neurons_input = reduce(lambda x,y: x*y, shape)
    
    default_device = "cuda"
    device = default_device if torch.cuda.is_available() else "cpu"
    if device == default_device:
        torch.cuda.empty_cache()
    print("[*] Using device: ", device)

    first_hidden_size = 512
    second_hidden_size = 128
    output_layer_size = 2**5
    print(f"Neurons: {total_neurons_input} -> {first_hidden_size} -> {second_hidden_size} -> {output_layer_size}")

    model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(total_neurons_input, first_hidden_size),
                nn.BatchNorm1d(first_hidden_size),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(first_hidden_size, second_hidden_size),
                nn.GELU(),
                nn.Linear(second_hidden_size, output_layer_size),
                nn.Softmax(dim=1)
            ).to(device)
    

    data = Data(files_handler.csv_file_path, files_handler.embeddings_path)
    train_size = int(args.train_percentage * len(data))
    test_size = len(data) - train_size

    train_dataset, test_dataset = random_split(
            data, [train_size, test_size], 
            generator=torch.Generator().manual_seed(DEFAULT_RANDOM_SEED))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    loss_fn = nn.KLDivLoss(reduction="batchmean")

    print("-"*30)
    print("Training Model")
    opt = optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    progress = pd.DataFrame(columns=("epoch", "loss", "iter", "output"))
    progress_i = 0
    
    for epoch in trange(args.epochs):
        epoch_loss = 0

        model.train(True)

        for i, data in enumerate(train_loader):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            opt.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            opt.step()
            
            loss_step = loss.item()
            epoch_loss += loss_step
            progress.loc[progress_i] = {
                    "epoch": epoch,
                    "loss": loss_step,
                    "iter": i,
                    "output": str(outputs.tolist())
                }
            progress_i += 1

            if(i % 10 == 0):
                print("Current loss: ", epoch_loss/(i+1))

        

        print(epoch_loss, epoch_loss/len(train_loader))


    

if __name__ == "__main__":

    main()
