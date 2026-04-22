"""Methods for handling dataset files."""
import os
from typing import List

from tqdm import tqdm

from utils.datatypes import FilePath

class Files:
    """Class for handling dataset files."""

    def __init__(self, base_folder:FilePath):
        self._base_folder = base_folder
        self._dataset_images_path = os.path.join(base_folder, "images")
        self._dataset_file_path = os.path.join(base_folder, "dataset.csv")
        self._dataset_h5_file_path = os.path.join(base_folder, "images.h5")
        self._checkpoint_path = os.path.join(base_folder, "checkpoint-gen.json")
        self._ghz_file_path = os.path.join(base_folder, "ghz.pth")
        self._ghz_image_file_path = os.path.join(base_folder, "ghz.png")
        self._final_model_path = os.path.join(base_folder, "final_model.safetensors")

    @property
    def images_path(self) -> FilePath:
        """Dataset images path"""
        return self._dataset_images_path

    @property
    def csv_file_path(self) -> FilePath:
        """Dataset csv file path"""
        return self._dataset_file_path

    @property
    def h5_file_path(self) -> FilePath:
        """Dataset h5 file path"""
        return self._dataset_h5_file_path

    @property
    def checkpoint_path(self) -> FilePath:
        """Dataset generation checkpoint file path"""
        return self._checkpoint_path
    
    @property
    def ghz_image_path(self) -> FilePath:
        """GHZ image file path"""
        return self._ghz_image_file_path

    @property
    def ghz_path(self) -> FilePath:
        """GHZ file path"""
        return self._ghz_file_path

    @property
    def final_model_path(self) -> FilePath:
        """Final model weights path"""
        return self._final_model_path

    def create_dataset_folder(self):
        """Create the dataset folder."""
        os.makedirs(self._dataset_images_path, exists_ok=True)

    @staticmethod
    def remove_duplicated_files(files:List[FilePath]):
        """Remove images that are duplicated (same hash)."""

        # We should create a tmp df with the clean data
        # and save it to CSV first, before proceeding.

        for file in tqdm(files, desc="Removing file: "):
            os.remove(file)
