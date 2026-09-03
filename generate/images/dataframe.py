
"""Holds Dataframe based utilities."""
import os

import pandas as pd

from utils.datatypes import FilePath,DFRows

class DF:
    """A class to hold a dataframe."""
    def __init__(self, filepath:FilePath):
        self._df_path = filepath
        self._columns = (
            "index",
            "depth",
            "file",
            "result",
            "hash",
            "total_meas",
            "measurements",
            "img_width",
            "img_height",
            "n_two_qubit_gates",
            "n_one_qubit_gates",
            "amount_gates",
            "file_size_bytes",
            "n_barriers",
        )
        self._current_index = 0
        self._df = None

        if os.path.exists(self._df_path):
            self._load_dataframe()
        else:
            self._create_df_file()

    def save_df(self):
        """Saves dataframe into csv file."""
        self._df.to_csv(self._df_path, index=False)

    def _create_df_file(self):
        """Create the dataframe file"""
        obj =  pd.DataFrame(columns=self._columns)
        self._df = obj
        self.save_df()

    def _load_dataframe(self) -> None:
        """Opens csv file into a dataframe object."""
        csv = pd.read_csv(self._df_path)
        self._df = csv
        self._current_index = len(csv)

    def append_rows_to_file(self, rows: DFRows):
        """
        append rows in a dataframe
        """
        for row in rows:
            self._df.loc[self._current_index] = row
            self._current_index += 1


