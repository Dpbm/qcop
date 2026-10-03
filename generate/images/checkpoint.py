"""Checkpoint for dataset generation"""
import json
import os

from generate.dataset.files import Files
from utils.datatypes import FilePath

class Checkpoint:
    """Class to handle generate data checkpoints"""

    def __init__(self, path: FilePath):
        self._path = path
        
        if not os.path.exists(path):
            Checkpoint.create_empty(path)

        with open(path, "r") as file:
            data = json.load(file)
            self._index = data.get("index", 0)

    @property
    def index(self) -> int:
        """get checkpoint generation index"""
        return self._index

    @index.setter
    def index(self, value: int):
        """update index"""
        self._index = value
    
    def save(self):
        """Saves checkpoint to a json file"""
        with open(self._path, "w") as file:
            data = {
                "index": self._index,
            }
            json.dump(data, file)
    
    @staticmethod
    def create_empty(path:FilePath):
        """Create an empty checkpoint"""
        with open(path, "w") as file:
            data = {
                "index": 0,
            }
            json.dump(data, file)


