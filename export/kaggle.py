"""Export dataset and model to kaggle"""

import os
import sys
from time import ctime
import argparse

import kagglehub as kh

from utils.datatypes import FilePath

def upload_dataset(folder:FilePath):
    """
    Upload dataset files to kaggle
    """
    version = ctime()
    dataset_name = str(os.getenv("KAGGLE_DATASET"))

    kh.dataset_upload(
        dataset_name, 
        folder, 
        version_notes=version,
        ignore_patterns=["dataset/"]
    )

def upload_model(folder:str):
    """
    Get model file and upload it to kaggle
    """

    version = ctime()
    model_name = str(os.getenv("KAGGLE_MODEL"))

    kh.model_upload(
        handle=model_name,
        local_model_dir=folder,
        version_notes=version,
        ignore_patterns=["dataset/","ghz*","*.zip","*.h5","*.csv"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args(sys.argv[1:])

    upload_dataset(args.path)
    upload_model(args.path)