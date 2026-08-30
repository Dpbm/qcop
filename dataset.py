from typing import Tuple
import json

from torch.utils.data import Dataset
import torch
import h5py
import polars as pl
import numpy as np

from generate.dataset.dataframe import DF
from utils.datatypes import FilePath

class Data(Dataset):
    def __init__(self, df_path:FilePath, images_path:FilePath):
        _df_handler = DF(df_path)

        self._df = _df_handler.load_lazy_frame()
        self._dataset = h5py.File(images_path, "r")
        self._size = len(self._dataset)

    def __len__(self):
        return self._size

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if(index >= self._size):
            raise IndexError("Index %d does not exist in the dataset"%(index))

        df_row = self._df.filter(pl.col("index") == index).collect()

        if df_row.is_empty():
            raise IndexError("Value not found for index: %d"%(index))

        embedding = np.array(self._dataset[str(index)])
        return torch.Tensor(embedding), torch.Tensor(json.loads(df_row["result"][0]))




