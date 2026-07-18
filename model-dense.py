import argparse
import shutil
from functools import reduce
import json
import asyncio
import os

from tqdm import trange
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.constants import DEFAULT_RANDOM_SEED, DEFAULT_MODEL_NAME
from generate.dataset.files import Files
from dataset import Data
from export import export_model_parallel

async def main():
    parser = argparse.ArgumentParser(description="Train Dense Model")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--target-folder", type=str, required=True)
    parser.add_argument("--train-percentage", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--es-patience", type=int, default=4) 
    parser.add_argument("--es-threshold", type=float, default=0.1) 
    parser.add_argument("--scheduler-patience", type=int, default=4)
    parser.add_argument("--scheduler-threshold", type=float, default=0.1)
    parser.add_argument("--load-checkpoint", type=bool, default=False)
    parser.add_argument("--model-name-kaggle", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-name-hf", type=str, default=DEFAULT_MODEL_NAME)
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
    opt = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(opt, patience=args.scheduler_patience, threshold=args.scheduler_threshold)

    progress = pd.DataFrame(columns=("epoch", "loss", "iter", "output", "step"))

    progress_i = 0
    best_loss = float('inf')
    last_loss = float('inf')
    model_weights = None
    early_stop_counter = 0
    starting_epoch = 0

    if args.load_checkpoint and os.path.exists(files_handler.model_checkpoint_path):
        print("[*] Loading Checkpoint")

        with open(files_handler.model_checkpoint_path, "r") as checkpoint:
            checkpoint_data = json.load(checkpoint)
            starting_epoch = checkpoint_data["epoch"]
            best_loss = checkpoint_data["best_loss"]
            last_loss = checkpoint_data["last_loss"]
            early_stop_counter = checkpoint_data["es_counter"]
            
            model_weights = checkpoint_data["weights"]
            state_dict = torch.load(model_weights, map_location=device)
            model.load_state_dict(state_dict)
            
            scheduler_data = torch.load(files_handler.scheduler_path)
            scheduler.load_state_dict(scheduler_data)
            
            opt_data = torch.load(files_handler.opt_path)
            opt.load_state_dict(opt_data)

            progress = pd.read_csv(files_handler.history_path)
            progress_i = len(progress)

    
    for epoch in trange(starting_epoch, args.epochs):
        epoch_loss = 0

        model.train(True)

        for i, data in enumerate(train_loader):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            opt.zero_grad()

            outputs = F.log_softmax(model(inputs), dim=1)

            loss = loss_fn(outputs, labels)
            loss.backward()

            opt.step()
            
            loss_step = loss.item()

            scheduler.step(loss_step)

            epoch_loss += loss_step
            progress.loc[progress_i] = {
                    "epoch": epoch,
                    "loss": loss_step,
                    "iter": i,
                    "output": str(outputs.tolist()),
                    "step": "train"
                }
            progress_i += 1

            if(i % 10 == 0):
                print("Current loss: ", epoch_loss/(i+1))

        print("Train Loss: ", epoch_loss, epoch_loss/len(train_loader))

        model.eval()
        
        test_loss = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs = data[0].to(device)
                labels = data[1].to(device)

                outputs = F.log_softmax(model(inputs), dim=1)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()

        print("Test Loss: ", test_loss, test_loss/len(test_loader))

        print("[*] Saving history")
        progress.to_csv(files_handler.history_path, index=False)

        if last_loss != float('inf') and last_loss-test_loss <= args.es_threshold:
            print("[*] model has not evolved")
            early_stop_counter += 1
        else:
            early_stop_counter = 0

        last_loss = test_loss

        if test_loss < best_loss:
            print("[*] Saving model weights")
            best_loss = test_loss
            model_path = files_handler.model_weights_path
            model_weights = model_path
            torch.save(model.state_dict(), model_path)


        with open(files_handler.model_checkpoint_path, "w") as checkpoint:
            print("[*] Saving Checkpoint")
            json.dump({
                    "epoch": epoch+1, 
                    "best_loss": best_loss, 
                    "last_loss": last_loss, 
                    "weights":model_weights,
                    "es_counter": early_stop_counter,
                }, checkpoint)
            torch.save(scheduler.state_dict(), files_handler.scheduler_path)
            torch.save(opt.state_dict(), files_handler.opt_path)
        
        if early_stop_counter >= args.es_patience:
            print("[*] Stopping earlier")
            break
        

    print("[*] Evaluating GHZ")
    shutil.copy(model_weights, files_handler.final_model_path)

    state_dict = torch.load(model_weights, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    ghz = torch.load(files_handler.ghz_path, map_location=device)
    with torch.no_grad():
        output = F.softmax(model(ghz), dim=1)
    print("GHZ prediction: ", output)

    print("[*] Exporting model...")
    await export_model_parallel(args.target_folder, args.model_name_kaggle, os.getenv("HUGGINGFACE_API_KEY"), args.model_name_hf)

if __name__ == "__main__":
    asyncio.run(main())
