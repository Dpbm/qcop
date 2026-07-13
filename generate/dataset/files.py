"""Methods for handling dataset files."""
import os
import shutil
from typing import List
from time import ctime

from tqdm import tqdm

from utils.datatypes import FilePath

class Files:
    """Class for handling dataset files."""

    def __init__(self, base_folder:FilePath):
        self._base_folder = base_folder
        self._dataset_images_path = os.path.join(base_folder, "images")
        self._dataset_file_path = os.path.join(base_folder, "dataset.csv")
        self._df_tmp_path = os.path.join(base_folder, "tmp-df.csv")
        self._dataset_h5_file_path = os.path.join(base_folder, "images.h5")
        self._checkpoint_path = os.path.join(base_folder, "checkpoint-gen.json")
        self._ghz_file_path = os.path.join(base_folder, "ghz.pth")
        self._ghz_image_file_path = os.path.join(base_folder, "ghz.png")
        self._final_model_path = os.path.join(base_folder, "final_model.safetensors")
        self._pre_analysis_path = os.path.join(base_folder, "pre-analysis.html")
        self._post_analysis_path = os.path.join(base_folder, "post-analysis.html")

        self._embeddings_checkpoint_path = os.path.join(base_folder, "embeddings_checkpoint.json")
        self._embeddings_path = os.path.join(base_folder, "embeddings.h5")
        self._embeddings_shape_path = os.path.join(base_folder, "embeddings_shape.json")
    
        self._model_checkpoint_path = os.path.join(base_folder, "checkpoint.json")
        self._history_path = os.path.join(base_folder, "history.csv")

    @property
    def df_tmp_path(self) -> FilePath:
        """Returns Temp Dataframe path."""
        return self._df_tmp_path

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

    @property
    def pre_analysis_path(self) -> FilePath:
        """Pre analysis HTML file path"""
        return self._pre_analysis_path

    @property
    def post_analysis_path(self) -> FilePath:
        """Post analysis HTML file path"""
        return self._post_analysis_path
    
    @property
    def embeddings_checkpoint_path(self) -> FilePath:
        """Embeddings checkpoint file path"""
        return self._embeddings_checkpoint_path

    @property
    def embeddings_path(self) -> FilePath:
        """Embeddings h5 file path"""
        return self._embeddings_path
    
    @property
    def embeddings_shape_path(self) -> FilePath:
        """Embeddings dimensions"""
        return self._embeddings_shape_path

    @property
    def model_weights_path(self) -> FilePath:
        """Create a path for the model weights"""
        return os.path.join(self._base_folder, f"model_{ctime()}")

    @property
    def model_checkpoint_path(self) -> FilePath:
        """Model checkpoint"""
        return self._model_checkpoint_path

    @property
    def history_path(self) -> FilePath:
        """Model evolution history checkpoint"""
        return self._history_path
    
    def create_dataset_folder(self):
        """Create the dataset folder."""
        os.makedirs(self._dataset_images_path, exist_ok=True)


    @staticmethod
    def remove_duplicated_files(files:List[FilePath]) -> List[FilePath]:
        """
        Remove images that are duplicated (same hash).

        Returns:
            Files that do not exist. -> List[FilePath]
        """

        # We should create a tmp df with the clean data
        # and save it to CSV first, before proceeding.

        dont_exist_files = []

        for file in tqdm(files, desc="Removing file: "):
            try:
                os.remove(file)
            except FileNotFoundError:
                print("[!] File %s doesn't exist"%file)
                dont_exist_files.append(file)
            
        return dont_exist_files

    def move_tmp_to_definitive(self):
        """change tmp file name to the definitive dataset"""
        os.remove(self._dataset_file_path)
        shutil.move(self._df_tmp_path, self._dataset_file_path)
