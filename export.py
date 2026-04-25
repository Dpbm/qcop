"""export dataset and model to external repositories"""

from abc import ABC, abstractmethod
from typing import Optional, List
from time import ctime
import asyncio
from pathlib import Path

import kagglehub as kh
from huggingface_hub import HfApi

from utils.datatypes import FilePath

class Exporter(ABC):
    ignore_patterns_dataset = [
            "dataset/",
            "*.json",
            "checkpoint_*",
            "model_*",
            "*.png",
            "*tmp*",
            "*.dat",
            "*.safetensors"
        ]

    @abstractmethod
    def upload_dataset(self):
        pass

    @abstractmethod
    def upload_model(self):
        pass


class KaggleExporter(Exporter):
    def __init__(
            self, 
            target_folder:FilePath,
            dataset_name:Optional[str]=None, 
            model_name:Optional[str]=None
    ):
        self._target_folder = target_folder
        self._model_name = model_name
        self._dataset_name = dataset_name    
        
        self._ignore_patterns_model = [
            "dataset/",
            "*.json",
            "checkpoint_*",
            "*.csv",
            "*.png",
            "*tmp*",
            "*.dat",
            "*.h5",
            "*.pth"
        ]

    async def upload_dataset(self):
        """Upload dataset to Kaggle"""
        assert self._dataset_name is not None, "You must set a dataset name"
        await asyncio.create_task(
                kh.dataset_upload(
                    self._dataset_name,
                    self._target_folder,
                    version_note=ctime(),
                    ignore_patterns=Exporter.ignore_patterns_dataset
            ))
    
    async def upload_model(self):
        """Upload model to Kaggle"""
        assert self._model_name is not None, "You must set a model name"
        await asyncio.create_task(
                kh.model_upload(
                    handle=self._model_name,
                    local_model_dir=self._target_folder,
                    version_notes=ctime(),
                    ignore_patterns=self._ignore_patterns_model
            ))
                
    


class HuggingFaceExporter(Exporter):
    def __init__(
            self, 
            api_key:str,
            target_folder:FilePath,
            dataset_name:Optional[str]=None, 
            model_name:Optional[str]=None
    ):
        self._api = HfApi(token=api_key)
        self._target_folder = target_folder
        self._model_name = model_name
        self._dataset_name = dataset_name

        files_handler = Files(target_folder)

        self._model_file_path = files_handler.final_model_path
        self._model_file = Path(files_handler.final_model_path).name

        self._ignore_patterns_model = [
            "dataset/",
            "*.json",
            "checkpoint_*",
            "*.csv",
            "*.png",
            "*tmp*",
            "*.dat",
            "*.h5",
            "*.pth"
        ]

    async def upload_dataset(self):
        """Upload dataset to HuggingFace"""
        assert self._dataset_name is not None, "You must set a dataset name"

        await asyncio.create_task(
                self._api.upload_folder(
                    folder_path=self._target_folder,
                    repo_id=self._dataset_name,
                    repo_type="dataset",
                    ignore_patterns=Exporter.ignore_patterns_dataset
                ))
    
    async def upload_model(self):
        """Upload model to Kaggle"""
        assert self._model_name is not None, "You must set a model name"
        await asyncio.create_task(
                self._api.upload_file(
                    path_or_fileobj=self._model_file_path,
                    path_in_repo=self._model_file,
                    repo_id=self._model_name,
                    repo_type="model",
                ))
