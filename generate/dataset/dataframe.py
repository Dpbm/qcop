"""Holds Dataframe based utilities."""

import os
from typing import Any, Dict, List
import csv as csvLib
from pathlib import Path

import polars as pl

from utils.datatypes import FilePath
from utils.constants import DEFAULT_RANDOM_SEED
from utils.classproperty import  ClassProperty

Schema = Dict[str, Any]

class DF:
    """A class to hold a dataframe."""

    def __init__(self, filepath:FilePath, seed:int=DEFAULT_RANDOM_SEED):
        self._df_path = filepath
        self._df_tmp_path = os.path.join(Path(filepath).parent, "tmp-df.csv")
        self._seed = seed

    @property
    def df_tmp_path(self) -> FilePath:
        """Returns Temp Dataframe path."""
        return self._df_tmp_path

    @df_tmp_path.setter
    def df_tmp_path(self, filepath:FilePath):
        """Set Temp Dataframe path."""
        self._df_tmp_path = filepath

    @property
    def df_file_exists(self) -> bool:
        """Returns True if the file exists."""
        return os.path.exists(self._df_path)

    @ClassProperty
    def df_schema(self) -> Schema:
        """Return the Polars Dataframe schema."""
        return {
            "index": pl.UInt32,
            "depth": pl.UInt8,
            "file": pl.String,
            "result": pl.String,
            "hash": pl.String,
            "measurements": pl.String,
            "img_width": pl.UInt16,
            "img_height": pl.UInt16,
            "n_two_qubit_gates": pl.UInt8,
            "n_one_qubit_gates": pl.UInt8,
            "file_size_bytes": pl.Uint32,
            "n_barriers": pl.UInt8,
        }

    @staticmethod
    def crete_df_obj() -> pl.LazyFrame:
        """Returns a Dataframe object based on schema."""
        return pl.LazyFrame(schema=DF.df_schema)

    def lazy_save(self, obj: pl.LazyFrame):
        """Saves the lazy frame object to a csv file."""
        obj.sink_csv(obj, self._df_path)

    def create_df_file(self):
        """Create the dataframe file"""
        obj = DF.crete_df_obj()
        self.lazy_save(obj)

    def load_lazy_frame(self) -> pl.LazyFrame:
       """Opens csv file inside a lazy frame object."""
       csv = pl.scan_csv(self._df_path)
       return csv.cast(DF.df_schema)

    def load_dataframe(self) -> pl.DataFrame:
        """Opens csv file into a dataframe object."""
        csv = pl.read_csv(self._df_path)
        return csv.cast(DF.df_schema)

    def shuffle_df(self, df:pl.DataFrame) -> pl.DataFrame:
        """Shuffles the dataframe."""
        return df.sample(fraction=1.0, shuffle=True, seed=self._seed)

    def append_rows_to_file(self, rows: Rows):
        """
        Use python's built-in csv library to append rows into a file
        without loading it into memory directly.
        """

        with open(self._df_path, "a") as file:
            writer = csvLib.writer(file)
            writer.writerows(rows)

    @staticmethod
    def clean_duplicated_rows(df:pl.LazyFrame) -> pl.LazyFrame:
        """Clean Df duplicated rows based on filepath and hash."""
        clean_df = df.filter(pl.col("file").is_first_distinct())
        clean_df = clean_df.filter(pl.col("hash").is_first_distinct())
        return clean_df

    @staticmethod
    def get_duplicated_files(df:pl.LazyFrame, clean:pl.LazyFrame) -> List[str]:
        """Get the files that are duplicated by applying a df diff."""
        duplicated_files = (
            df.join(clean_df, on=df.collect_schema().names(), how="anti")
            .collect()
            .get_column("file")
        )

        return duplicated_files.to_list()  # type: ignore








