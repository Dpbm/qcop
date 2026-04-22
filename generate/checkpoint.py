"""Checkpoint for dataset generation"""

from enum import Enum
import json

from dataset.files import Files

class Stages(Enum):
    """Enum for dataset generation stages"""
    GEN_IMAGES = "gen"
    SHUFFLE = "shuffle"
    DUPLICATES = "duplicates"
    TRANSFORM = "transform"

class Checkpoint:
    """Class to handle generate data checkpoints"""

    __slots__ = ["_path", "_stage", "_index"]

    def __init__(self, path: Optional[FilePath]):
        self._path = path

        with open(path, "r") as file:
            data = json.load(file)
            stage = data.get("stage")
            self._stage = Stages.GEN_IMAGES if stage is None else Stages(stage)
            self._index = data.get("index", 0)

    @classmethod
    def get_checkpoint(cls, path:FilePath):
        if not os.path.exists(path):
            Checkpoint.create_empty(checkpoint_path)

        return cls(checkpoint_path)

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

    def next_stage(self):
        """Move state to the new stage"""
        if self._stage == Stages.GEN_IMAGES:
            self._stage = Stages.SHUFFLE

        elif self._stage == Stages.SHUFFLE:
            self._stage = Stages.DUPLICATES

        elif self._stage == Stages.DUPLICATES:
            self._stage = Stages.TRANSFORM

        else:
            self._stage = Stages.GEN_IMAGES

        self._index = 0


    def save(self):
        """Saves checkpoint to a json file"""
        with open(self._path, "w") as file:
            data = {
                "stage": self._stage.value,
                "index": self._index,
            }
            json.dump(data, file)
    
    @staticmethod
    def create_empty(path:FilePath):
        """Create an empty checkpoint"""
        with open(path, "w") as file:
            data = {
                "stage": Stages.GEN_IMAGES,
                "index": 0,
            }
            json.dump(data, file)


