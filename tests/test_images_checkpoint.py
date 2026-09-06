import os
import json

import pytest

from generate.images import Checkpoint

class TestImagesCheckpoint:
    def test_get_checkpoint_create_an_empty_file(self,checkpoint_path):
        assert not os.path.exists(checkpoint_path)

        checkpoint = Checkpoint(checkpoint_path)

        assert os.path.exists(checkpoint_path)

        with open(checkpoint_path,"r") as file:
            assert json.load(file) == {"index":0,"thread_indexes":[]}

    def test_checkpoint_already_exists(self,checkpoint_path):
        with open(checkpoint_path,"w") as file:
            json.dump({"index":140, "thread_indexes":[1,23,45]},file)
        
        checkpoint = Checkpoint(checkpoint_path)
        assert checkpoint.index == 140
        assert checkpoint.thread_indexes == [1,23,45]

    def test_setters_and_save(self,checkpoint_path):
        checkpoint = Checkpoint(checkpoint_path)
        
        assert checkpoint.index == 0
        assert checkpoint.thread_indexes == []

        checkpoint.index = 10
        checkpoint.thread_indexes = [1,2,3,9191,3]

        assert checkpoint.index == 10
        assert checkpoint.thread_indexes == [1,2,3,9191,3]

        checkpoint.save()

        with open(checkpoint_path,"r") as file:
            assert json.load(file) == {"index":10,"thread_indexes":[1,2,3,9191,3]}




