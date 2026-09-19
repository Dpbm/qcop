from argparse import Namespace
import os

import pytest
import pandas as pd

from generate_images_dataset import main
from utils.constants import IMAGES_PATH,DATASET_FILE


class TestScript:

    def test_main_saved_files(self,target_folder):
        main(Namespace(
            target_folder=target_folder, 
            n_qubits=2, 
            amount_circuits=1,
            shots=1000,
            threads=1,
            max_gates=10))

        assert len(os.listdir(target_folder)) == 3
        assert len(os.listdir(os.path.join(target_folder,IMAGES_PATH))) == 3

        df = pd.read_csv(os.path.join(target_folder,DATASET_FILE))
        assert len(df) == 3
        




