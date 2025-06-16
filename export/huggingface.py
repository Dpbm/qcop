"""Export dataset and model to huggingface"""

import os

from huggingface_hub import HfApi

def upload_dataset(dataset_name:str, folder:str):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_folder(
        folder_path=folder,
        repo_id=dataset_name,
        repo_type="dataset",
        ignore_patterns=["dataset/"]
    )