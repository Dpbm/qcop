"""Checkpoint for dataset generation"""
from typing import List
import json
import os

from generate.dataset.files import Files
from utils.datatypes import FilePath

class Checkpoint:
    """Class to handle generate data checkpoints"""

    def __init__(self, path: FilePath):
        self._path = path

        with open(path, "r") as file:
            data = json.load(file)
            self._thread_indexes = data.get("thread_indexes", [])
            self._index = data.get("index", 0)

    @classmethod
    def get_checkpoint(cls, path:FilePath):
        if not os.path.exists(path):
            Checkpoint.create_empty(path)
        return cls(path)

    @property
    def index(self) -> int:
        """get checkpoint generation index"""
        return self._index

    @index.setter
    def index(self, value: int):
        """update index"""
        self._index = value
    
    @property
    def thread_indexes(self) -> List[int]:
        """get thread indexes"""
        return self._thread_indexes

    @thread_indexes.setter
    def thread_indexes(self, value: List[int]):
        """update thread indexes"""
        self._thread_indexes = value

    def save(self):
        """Saves checkpoint to a json file"""
        with open(self._path, "w") as file:
            data = {
                "index": self._index,
                "thread_indexes": self._thread_indexes
            }
            json.dump(data, file)
    
    @staticmethod
    def create_empty(path:FilePath):
        """Create an empty checkpoint"""
        with open(path, "w") as file:
            data = {
                "index": 0,
                "thread_indexes": []
            }
            json.dump(data, file)


