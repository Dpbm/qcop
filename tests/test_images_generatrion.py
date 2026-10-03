import os
import json

import pytest
import pandas as pd

from generate.images import Checkpoint, Images, DF


class TestImageGeneration:
    def test_generate_circuit_images(self, images_path):
        os.makedirs(images_path, exist_ok=True)
        im = Images(images_path)

        r = im._generate_circuit_image(
            10,
            2,
            4,
            1000,
            )
        
        assert os.listdir(images_path) == [ "10.png" ]
        assert r["index"] == 10
        assert r["file"] == os.path.join(images_path,"10.png")
        assert not r["file_size_bytes"] <= 0

    def test_generate_multiple_circuits(self, images_path,checkpoint_path, df_path): 
        os.makedirs(images_path, exist_ok=True)
        im = Images(images_path)

        checkpoint = Checkpoint(checkpoint_path)
        df = DF(df_path)

        im.generate_images(2,5,4,1000,df,2,checkpoint)

        assert checkpoint.index == 5
        assert df._current_index == 5
        assert sorted(os.listdir(images_path)) == sorted([ f"{i}.png" for i in range(5)])

        read_df = pd.read_csv(df_path)
        assert len(read_df) == 5


    def test_generate_multiple_circuits_start_from_checkpoint(self, images_path,checkpoint_path,df_path): 
        os.makedirs(images_path, exist_ok=True)
        im = Images(images_path)

        df = DF(df_path)

        checkpoint = Checkpoint(checkpoint_path)
        checkpoint.index = 2              # already created 2 circuits

        im.generate_images(
            n_qubits=2,
            amount_circuits=5,
            total_gates=4,
            shots=1000,
            df=df,
            total_threads=2,
            checkpoint=checkpoint
            )

        assert checkpoint.index == 5 # must create 5 circuits, so in total 5 elements

        assert df._current_index == 3 # must be 3, since we are aiming to create only 3 circuits
                                      # the code implies that, since the index is 2, we have already
                                      # created 2 circuits previously

        assert [f"{i+2}.png" for i in range(3)] == sorted(os.listdir(images_path))

