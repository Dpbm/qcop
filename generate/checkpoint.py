"""Checkpoint for dataset generation"""

from typing import Optional, List
from enum import Enum
import json
import os

from generate.dataset.files import Files
from utils.datatypes import FilePath

class Stages(Enum):
    """Enum for dataset generation stages"""
    GEN_IMAGES = "gen"
    SHUFFLE = "shuffle"
    DUPLICATES = "duplicates"
    TRANSFORM = "transform"
    EXPORT = "export"

class Checkpoint:
    """Class to handle generate data checkpoints"""

    __slots__ = ["_path", "_stage", "_index", "_thread_indexes"]

    def __init__(self, path: Optional[FilePath]):
        self._path = path

        with open(path, "r") as file:
            data = json.load(file)
            self._stage = Stages(data.get("stage", Stages.GEN_IMAGES.value))
            self._thread_indexes = data.get("thread_indexes", [])
            self._index = data.get("index", 0)

    @classmethod
    def get_checkpoint(cls, path:FilePath):
        if not os.path.exists(path):
            Checkpoint.create_empty(path)
        return cls(path)

    @property
    def stage(self) -> Stages:
        """get checkpoint generation stage"""
        return self._stage

    @stage.setter
    def stage(self, value: Stages):
        """Update stage"""
        self._stage = value

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

    def next_stage(self):
        """Move state to the new stage"""
        if self._stage == Stages.GEN_IMAGES:
            self._stage = Stages.SHUFFLE

        elif self._stage == Stages.SHUFFLE:
            self._stage = Stages.DUPLICATES

        elif self._stage == Stages.DUPLICATES:
            self._stage = Stages.TRANSFORM
        
        elif self._stage == Stages.TRANSFORM:
            self._stage = Stages.EXPORT

        else:
            self._stage = Stages.GEN_IMAGES

        self._index = 0
        self._thread_indexes = []


    def save(self):
        """Saves checkpoint to a json file"""
        with open(self._path, "w") as file:
            data = {
                "stage": self._stage.value,
                "index": self._index,
                "thread_indexes": self._thread_indexes
            }
            json.dump(data, file)
    
    @staticmethod
    def create_empty(path:FilePath):
        """Create an empty checkpoint"""
        with open(path, "w") as file:
            data = {
                "stage": Stages.GEN_IMAGES.value,
                "index": 0,
                "thread_indexes": []
            }
            json.dump(data, file)


