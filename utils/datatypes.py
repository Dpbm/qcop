"""Common datatypes"""

from typing import Tuple
import polars as pl

FilePath = str
Dimensions = Tuple[int, int]

# Polars dataframe schema
df_schema = {
    "index": pl.UInt16,
    "depth": pl.UInt8,
    "file": pl.String,
    "result": pl.String,
    "hash": pl.String,
    "measurements": pl.String,
}
