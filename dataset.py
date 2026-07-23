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
        self._max_index = self._df.select(pl.col("index").max()).collect().item()
        print(self._max_index, max(list(map(int, list(self._dataset.keys())))))

    def __len__(self):
        return self._size

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        current_index = index
        while True:
            if(current_index > self._max_index):
                raise IndexError("Index %d does not exist in the dataset"%(current_index))

            df_row = self._df.filter(pl.col("index") == current_index).collect()

            if not df_row.is_empty():
                embedding = np.array(self._dataset[str(current_index)])
                return torch.Tensor(embedding), torch.Tensor(json.loads(df_row["result"][0]))

            current_index += 1



