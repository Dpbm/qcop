"""Helpers for exporting data"""

import os

from utils.constants import MODEL_FILE_PREFIX
from utils.datatypes import FilePath

def get_latest_model(folder:FilePath) -> FilePath:
    """
    Check between model files which was the latest modified (the latest model)
    """

    model_files = []
    for file in os.listdir(folder):
        if not file.startswith(MODEL_FILE_PREFIX):
            continue
        model_files.append(file)

    get_file_mod_time = lambda file: os.path.getmtime(os.path.join(folder, file)) # noqa: E731
    model_files.sort(key=get_file_mod_time, reverse=True)
    return model_files[0]

    


