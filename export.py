"""export dataset and model to external repositories"""

from abc import ABC, abstractmethod
from typing import Optional, List
from time import ctime 
from pathlib import Path
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

import kagglehub as kh
from huggingface_hub import HfApi

from utils.datatypes import FilePath
from generate.dataset.files import Files

class Exporter(ABC):
    ignore_patterns_dataset = [
            "dataset/",
            "*.json",
            "checkpoint_*",
            "model_*",
            "*.png",
            "*tmp*",
            "*.dat",
            "*.safetensors",
            "embeddings_checkpoint.json"
        ]
    ignore_patterns_model = [
            "dataset/",
            "*.json",
            "checkpoint_*",
            "model_*",
            "*.csv",
            "*.png",
            "*tmp*",
            "*.dat",
            "*.h5",
            "*.pth",
            "*.html"
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
        

    def upload_dataset(self):
        """Upload dataset to Kaggle"""
        assert self._dataset_name is not None, "You must set a dataset name"
        print("[*] Uploading to Kaggle")
        kh.dataset_upload(
            self._dataset_name,
            self._target_folder,
            version_notes=ctime(),
            ignore_patterns=super().ignore_patterns_dataset
        )
    
    def upload_model(self):
        """Upload model to Kaggle"""
        assert self._model_name is not None, "You must set a model name"
        print("[*] Uploading to Kaggle")
        kh.model_upload(
            handle=self._model_name,
            local_model_dir=self._target_folder,
            version_notes=ctime(),
            ignore_patterns=super().ignore_patterns_model
        )
                
    


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

    def upload_dataset(self):
        """Upload dataset to HuggingFace"""
        assert self._dataset_name is not None, "You must set a dataset name"
        print("[*] Uploading to Huggingface")
        self._api.upload_folder(
            folder_path=self._target_folder,
            repo_id=self._dataset_name,
            repo_type="dataset",
            ignore_patterns=super().ignore_patterns_dataset
        )
    
    def upload_model(self):
        """Upload model to Kaggle"""
        assert self._model_name is not None, "You must set a model name"
        print("[*] Uploading to Huggingface")
        self._api.upload_file(
            path_or_fileobj=self._model_file_path,
            path_in_repo=self._model_file,
            repo_id=self._model_name,
            repo_type="model",
        )

async def export_parallel(target_folder:FilePath, kaggle_df_name:str, hf_token:str, hf_df_name:str):
    """Export datasets in parallel using asyncio"""

    kaggle = KaggleExporter(target_folder, dataset_name=kaggle_df_name)
    hf = HuggingFaceExporter(hf_token, target_folder, dataset_name=hf_df_name)

    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=2) as pool:
        kaggle_upload = loop.run_in_executor(pool,kaggle.upload_dataset)
        hf_upload = loop.run_in_executor(pool,hf.upload_dataset)
        await asyncio.gather(kaggle_upload, hf_upload)

async def export_model_parallel(target_folder:FilePath, kaggle_model_name:str, hf_token:str, hf_model_name:str):
    """Export model in parallel using asyncio"""

    kaggle = KaggleExporter(target_folder, model_name=kaggle_model_name)
    hf = HuggingFaceExporter(hf_token, target_folder, model_name=hf_model_name)

    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=2) as pool:
        kaggle_upload = loop.run_in_executor(pool,kaggle.upload_model)
        hf_upload = loop.run_in_executor(pool,hf.upload_model)
        await asyncio.gather(kaggle_upload, hf_upload)

