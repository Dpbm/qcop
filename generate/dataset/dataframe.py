"""Holds Dataframe based utilities."""
import os
from typing import Any, Dict, List
import csv as csvLib
from pathlib import Path

import polars as pl
import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

from utils.datatypes import FilePath
from utils.constants import DEFAULT_RANDOM_SEED
from utils.classproperty import  ClassProperty
from generate.datatypes import *

class DF:
    """A class to hold a dataframe."""

    def __init__(self, filepath:FilePath, seed:int=DEFAULT_RANDOM_SEED):
        self._df_path = filepath
        self._seed = seed

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
            "total_meas": pl.UInt8,
            "measurements": pl.String,
            "img_width": pl.UInt16,
            "img_height": pl.UInt16,
            "n_two_qubit_gates": pl.UInt8,
            "n_one_qubit_gates": pl.UInt8,
            "amount_gates": pl.String,
            "file_size_bytes": pl.UInt32,
            "n_barriers": pl.UInt8,
        }

    @staticmethod
    def create_df_obj() -> pl.LazyFrame:
        """Returns a Dataframe object based on schema."""
        return pl.LazyFrame(schema=DF.df_schema)

    def lazy_save(self, obj: pl.LazyFrame):
        """Saves the lazy frame object to a csv file."""
        obj.sink_csv(self._df_path)

    def lazy_save_to_tmp(self, obj:pl.LazyFrame, tmp_path:FilePath):
        """Saves the df into a tmp file"""
        obj.sink_csv(tmp_path)

    def save_df(self, obj: pl.DataFrame):
        """Saves dataframe into file."""
        obj.write_csv(self._df_path)

    def create_df_file(self):
        """Create the dataframe file"""
        if self.df_file_exists:
            return

        obj = DF.create_df_obj()
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
    def get_files_via_left_join(left:pl.LazyFrame, right:pl.LazyFrame) -> List[str]:
        """Get the files that are on the left LazyFrame by applying a df diff."""
        duplicated_files = (
            left.join(right, on=left.collect_schema().names(), how="anti")
            .collect()
            .get_column("file")
        )

        return duplicated_files.to_list()  # type: ignore

    @staticmethod
    def remove_rows_with_non_existant_files(df:pl.LazyFrame, files:List[FilePath]):
        """Remove the rows with invalid files."""
        print("[*] Removing %d files from dataset" % len(files))
        return df.filter(~pl.col("file").is_in(files))

    @staticmethod
    def keep_25_to_75_quantiles_by_depth(df:pl.LazyFrame) -> pl.LazyFrame:
        """Create a lazy frame with only rows that the depth are in the middle two quantiles"""
        return df.filter(
                    pl.col("depth").is_between(
                        pl.col("depth").quantile(0.25).over(pl.lit(1)),
                        pl.col("depth").quantile(0.75).over(pl.lit(1)),
                        closed="both"
                    )
                )

    @staticmethod
    def run_statistics_notebook(target_path:FilePath):
        """Run the jupyter notebook to get the statistics"""
        notebook_path = os.path.join(".", "notebooks", "data-analysis.ipynb")

        with open(notebook_path, "r") as file:
            nb = nbformat.read(file, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": os.path.join(".","notebooks")}})

        html_exporter = HTMLExporter()
        body, _ = html_exporter.from_notebook_node(nb)

        with open(target_path, "w", encoding="utf-8") as file:
            print("[*] Saving pre-analysis...")
            file.write(body)


