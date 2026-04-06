"""Methods for handling dataset files."""
import os
from typing import List

from tqdm import tqdm

from utils.datatypes import FilePath

class Files:
    """Class for handling dataset files."""

    def __init__(self, base_folder:FilePath):
        self._base_folder = base_folder

    def create_dataset_folder(self):
        """Create the dataset folder."""
        os.makedirs(self._base_folder, exists_ok=True)

    @staticmethod
    def remove_duplicated_files(files:List[FilePath]):
        """Remove images that are duplicated (same hash)."""

        # We should create a tmp df with the clean data
        # and save it to CSV first, before proceeding.

        for file in tqdm(files, desc="Removing file: "):
            os.remove(file)