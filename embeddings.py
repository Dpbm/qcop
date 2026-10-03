import sys
import os
import asyncio
import argparse
import json

import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import pipeline
from accelerate import Accelerator
import h5py
import numpy as np

from utils.constants import DEFAULT_DATASET_NAME, MODEL, CSV_OUTPUTS_DATA

CHECKPOINT_FILE = os.path.join("data", "embeddings_checkpoint.json")
SHAPE_FILE = os.path.join("data", "shape.json")

def main():
    parser = argparse.ArgumentParser(description=f"Extract image embeddings from dataset with {MODEL}")
    parser.add_argument("--preload-amount", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--target-folder", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(os.path.join("data", CSV_OUTPUTS_DATA)):
        sys.exit(f"No file {CSV_OUTPUTS_DATA}")

    device = Accelerator().device
    print("[*] Using device: ", device)
    print("[*] preload amount of images: ", args.preload_amount)
    print("[*] batch size: ", args.batch_size)

    pipe = pipeline(task="image-feature-extraction", model=MODEL, device=device, batch_size=args.batch_size)

    checkpoint_exists = os.path.exists(CHECKPOINT_FILE)

    if not checkpoint_exists:
        print("[!] No Checkpoint...")
        df = pd.read_csv(os.path.join("data", CSV_OUTPUTS_DATA))
        h5_files = list(df.h5_file.unique())
        saved_shape = False
    else:
        with open(CHECKPOINT_FILE, "r") as checkpoint:
            print("[*] Loading Checkpoint...")
            data = json.load(checkpoint)
            h5_files = data["h5_files"]
            saved_shape = data["saved_shape"]
            keys = data["keys"]

    current_h5 = 0
    for h5 in h5_files:
        h5_file = h5.replace("../", '')
        print(f"[*] Using file: {h5_file}")

        with h5py.File(h5_file, 'r') as dataset_images:
            all_keys = list(dataset_images.keys()) if not checkpoint_exists else keys

            while True:
                selected_keys = all_keys[:args.preload_amount]
                
                preloaded_images = [
                    Image.fromarray(dataset_images[index][:])
                    for index in selected_keys
                ]

                embeddings = np.array(pipe(preloaded_images), dtype=np.float16)

                if not saved_shape:
                    with open(SHAPE_FILE, "w") as shape_file:
                        print("[*] Saving embeddings format")
                        json.dump(list(embeddings.shape[1:]), shape_file)
                    saved_shape = True

                with h5py.File(h5_file.replace("images_", "embeddings_"), "a") as embeddings_dataset:
                    for embedding,index in tqdm(zip(embeddings, selected_keys), desc="Saving embeddings: "):
                        embeddings_dataset.create_dataset(f"{index}", data=embedding)

                start_index += args.preload_amount
                all_keys = list(set(all_keys) - set(selected_keys))

                with open(CHECKPOINT_FILE, "w") as checkpoint:
                    print("[*] saving checkpoint...")
                    json.dump({
                        "h5_files":h5_files,
                        "saved_shape":saved_shape,
                        "keys":all_keys
                        },checkpoint)

                if len(all_keys) <= 0:
                    break


        current_h5 += 1

        with open(CHECKPOINT_FILE, "w") as checkpoint:
            print("[*] saving checkpoint for new h5...")
            json.dump({
                "h5_files":h5_files[current_h5:],
                "saved_shape":saved_shape,
                "keys":all_keys
                },checkpoint)
        

if __name__ == "__main__":
    main()
