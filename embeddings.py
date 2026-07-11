import os
import asyncio
import argparse
import json

from tqdm import tqdm
from PIL import Image
from transformers import pipeline
from accelerate import Accelerator
import h5py
import matplotlib.pyplot as plt
import numpy as np

from generate.dataset.files import Files
from utils.constants import DEFAULT_DATASET_NAME
from export import export_parallel

MODEL = "google/vit-base-patch16-384"

def main():
    parser = argparse.ArgumentParser(description=f"Extract image embeddings from dataset with {MODEL}")
    parser.add_argument("--preload-amount", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--target-folder", type=str, required=True)
    parser.add_argument("--initial-index", type=int, default=0)
    parser.add_argument("--only-export", type=bool, default=False)
    parser.add_argument("--dataset-name-kaggle", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-name-hf", type=str, default=DEFAULT_DATASET_NAME)
    args = parser.parse_args()

    if args.only_export:
        asyncio.run(export_parallel(args.target_folder, args.dataset_name_kaggle,os.getenv("HUGGINGFACE_API_KEY"), args.dataset_name_hf))
        exit()

    files_handler = Files(args.target_folder) 

    device = Accelerator().device
    print("[*] Using device: ", device)
    print("[*] preload amount of images: ", args.preload_amount)
    print("[*] batch size: ", args.batch_size)
    print("[*] initial index: (", args.initial_index, ")")

    pipe = pipeline(task="image-feature-extraction", model=MODEL, device=device, batch_size=args.batch_size)
    
    indexes = []
    finished = False

    if os.path.exists(files_handler.embeddings_checkpoint_path):
        print("[*] Reading chekpoint")
        with open(files_handler.embeddings_checkpoint_path,'r') as checkpoint:
            data = json.load(checkpoint)
            indexes = list(data["indexes"])
            finished = data["finished"]
    
    if not finished:

        with h5py.File(files_handler.h5_file_path, "r") as dataset:

            if not indexes:
                print("[*] adding indexes")
                indexes = list(dataset.keys())
                with open(files_handler.embeddings_checkpoint_path, 'w') as checkpoint:
                    json.dump({"indexes":indexes, finished:False},checkpoint)

            print("[*] total images in the dataset: ", len(indexes))

            current_index = 0
            while indexes:

                get_index = lambda i : (current_index*args.preload_amount)+i
                selected_indexes = [indexes[get_index(i)] 
                                        for i in range(args.preload_amount)
                                        if get_index(i) <= len(indexes)-1]
                preloaded_images = [Image.fromarray(dataset[selected_indexes[i]][:])
                                             for i in range(args.preload_amount)]
                
                print("Processing (", current_index, ")")

                embeddings = np.array(pipe(preloaded_images))
                embeddings_shape = embeddings.shape
                embeddings = np.reshape(embeddings, (args.preload_amount, embeddings_shape[1]*embeddings_shape[2]*embeddings_shape[3]))

                with h5py.File(files_handler.embeddings_path, "a") as embeddings_dataset:
                    for embedding,index in tqdm(zip(embeddings, selected_indexes), desc="Saving embeddings: "):
                        embeddings_dataset.create_dataset(index, data=embedding)

                indexes = list(set(indexes) - set(selected_indexes))
                with open(files_handler.embeddings_checkpoint_path, 'w') as checkpoint:
                    print("[*] Updating checkpoint")
                    json.dump({"indexes":indexes, "finished":False}, checkpoint)
                current_index += 1

    with open(files_handler.embeddings_checkpoint_path, 'w') as checkpoint:
        print("[*] Updating checkpoint (Finished)")
        json.dump({"indexes":[], "finished":True}, checkpoint)

    print("[*] Uploading dataset")
    asyncio.run(export_parallel(args.target_folder, args.dataset_name_kaggle,os.getenv("HUGGINGFACE_API_KEY"), args.dataset_name_hf))

if __name__ == "__main__":
    main()
