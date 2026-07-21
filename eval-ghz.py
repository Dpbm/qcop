import argparse
import json
from functools import reduce

import torch.nn as nn
import torch.nn.functional as F
import torch

from generate.dataset.files import Files
from model_dense import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval GHZ circuit")
    parser.add_argument("--target-folder", type=str, required=True)
    args = parser.parse_args()

    files_handler = Files(args.target_folder)
    default_device = "cuda"
    device = default_device if torch.cuda.is_available() else "cpu"
    if device == default_device:
        torch.cuda.empty_cache()
    print("[*] Using device: ", device)
    
    with open(files_handler.embeddings_shape_path, "r") as shape_file:
        shape = list(json.load(shape_file))
        total_neurons_input = reduce(lambda x,y: x*y, shape)

    model = get_model(total_neurons_input, device)

    state_dict = torch.load(files_handler.final_model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)

    print("[*] Evaluating GHZ")
    ghz = torch.from_numpy(torch.load(files_handler.ghz_path, map_location=device, weights_only=False)).float().to(device)
    print(ghz)
    print(ghz.shape)
    print(type(ghz))

    model.eval()
    with torch.no_grad():
        output = F.softmax(model(ghz), dim=1)
    print("GHZ prediction: ", output)
