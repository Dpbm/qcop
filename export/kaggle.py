"""Export dataset and model to kaggle"""

from time import ctime

import kagglehub as kh


def upload_dataset(dataset_name:str, folder:str):
    """
    Upload dataset files to kaggle
    """

    version = ctime()
    kh.dataset_upload(
        dataset_name, 
        folder, 
        version_notes=version,
        ignore_patterns=["dataset/"])