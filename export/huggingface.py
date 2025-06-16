"""Export dataset and model to huggingface"""

import os
import argparse
import sys

from huggingface_hub import HfApi

from utils.datatypes import FilePath
from export.helpers import get_latest_model
from utils.constants import MODEL_FILE_PREFIX, CHECKPOINT_FILE_PREFIX

def upload_dataset(folder:FilePath):
    """
    Upload dataset to huggingface
    """

    api = HfApi(token=os.getenv("HF_TOKEN"))
    dataset_name = str(os.getenv("HF_DATASET"))

    api.upload_folder(
        folder_path=folder,
        repo_id=dataset_name,
        repo_type="dataset",
        ignore_patterns=[
            "dataset/", 
            "*.json", 
            f"{MODEL_FILE_PREFIX}*", 
            f"{CHECKPOINT_FILE_PREFIX}*", 
            "*.png",
        ]
    )

def upload_model(folder:str):
    """
    Get model file and upload it to huggingface
    """

    latest_model = get_latest_model(folder)
    api = HfApi(token=os.getenv("HF_TOKEN"))
    model_name = str(os.getenv("HF_MODEL_REPO"))
    api.upload_file(
        path_or_fileobj=os.path.join(folder,latest_model),
        path_in_repo=latest_model,
        repo_id=model_name,
        repo_type="model",
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args(sys.argv[1:])

    upload_dataset(args.path)
    upload_model(args.path)

