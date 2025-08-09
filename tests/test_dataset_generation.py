from typing import List
import os
import gc

import pytest
import polars as pl

from dataset import (
    clean_duplicated_rows_df, 
    open_csv, 
    save_df, 
    start_df,
    get_duplicated_files_list_by_diff
)
from utils.datatypes import df_schema


@pytest.fixture()
def base_df() -> str:
    """
    The dataset that cotains the correct data.
    """
    return os.path.join(".", "tests", "dataset.csv")


@pytest.fixture()
def tmp_df() -> str:
    """
    A csv file meant to be temporary
    for tests.
    """
    return os.path.join(".", "tests", "dataset-test-output.csv") 

@pytest.fixture()
def tmp_df2() -> str:
    """
    A csv file meant to be temporary
    for tests (used as a alternative for the other one, 
    when necessary).
    """
    return os.path.join(".", "tests", "dataset-test-output2.csv") 

@pytest.fixture()
def total_runs_sample() -> int:
    """
    Sample size for unstable functions
    """
    return 1000

@pytest.fixture()
def duplicated_files() -> List[str]:
    """
    Returns a list with duplicated files in
    from csv.
    """

    return [
        "/home/airflow/data/dataset/298.png",
        "/home/airflow/data/dataset/301.png",
        "/home/airflow/data/dataset/305.png",
    ]


@pytest.fixture(autouse=True)
def clear_file(tmp_df, tmp_df2):
    """
    Clear tmp csv file
    """
    if(os.path.exists(tmp_df)): 
        os.remove(tmp_df)
    if(os.path.exists(tmp_df2)): 
        os.remove(tmp_df2)

class TestCSVFile:
    def test_open_csv(self,base_df):
        """
        Should open the df with no problems and cast it 
        to correct data types.
        """
        df = open_csv(base_df).collect()

        assert len(df) == 11
        assert df.schema == df_schema

    def test_gen_df_no_previous_file(self,tmp_df):
        """
        should create a new csv file.
        """

        assert not os.path.exists(tmp_df)
        start_df(tmp_df)
        assert os.path.exists(tmp_df)

        df_data = pl.read_csv(tmp_df)
        assert len(df_data) == 0

    def test_gen_df_file_already_exists(self,base_df,tmp_df):
        """
        Should not overwrite the existent file.
        """

        df = pl.read_csv(base_df)
        df.write_csv(tmp_df)

        assert os.path.exists(tmp_df)
        assert len(pl.read_csv(tmp_df)) == 11
        start_df(tmp_df)
        assert os.path.exists(tmp_df)
        assert len(pl.read_csv(tmp_df)) == 11






class TestDatasetClean:


    """Test dataset generation parts"""

    def test_clean_duplicated_rows_return_the_correct_of_rows(self, base_df):
        """Test if when we clear a dataframe, the unique rows are correct"""

        df = open_csv(base_df)

        assert len(df.collect()) == 11

        clean_df = clean_duplicated_rows_df(df)
        clean_df_index_rows = clean_df.collect().get_column('index').to_list()

        correct_indexes = [295, 296, 297, 298, 299, 300, 301, 302]

        assert len(clean_df.collect()) == 8
        assert clean_df_index_rows == correct_indexes

    def test_save_df_without_any_modifications_different_files(self, base_df, tmp_df):
        """
        Test if csv file is saved without problems doing no modifications and 
        saving in different files.
        """

        df = open_csv(base_df)
        save_df(df, tmp_df)
        target_csv = pl.read_csv(tmp_df)

        assert len(target_csv) == 11

    def test_save_df_with_modifications_different_files(self, base_df, tmp_df):
        """
        Test if csv file is saved without problems doing modifications and 
        saving in different files.
        """

        df = open_csv(base_df)
        df2 = clean_duplicated_rows_df(df)
        save_df(df2, tmp_df)
        target_csv = pl.read_csv(tmp_df)

        assert len(target_csv) == 8

    def test_save_df_with_modifications_different_files_and_rename(self, base_df, tmp_df, tmp_df2):
        """
        Test if csv file is saved without problems doing modifications and 
        saving in different files and renaming tmp_df2 to tmp_df.
        """

        df = pl.read_csv(base_df)
        df.write_csv(tmp_df)

        df3 = open_csv(tmp_df)
        df3 = clean_duplicated_rows_df(df3)
        save_df(df3, tmp_df2)
        
        os.remove(tmp_df)
        os.rename(tmp_df2, tmp_df)

        target_csv = pl.read_csv(tmp_df)

        assert len(target_csv) == 8

    def test_get_duplicated_files_list_by_diff(self, base_df, duplicated_files):
        """
        Must take the diff between the raw csv and the cleaned one
        and return a list of files that are duplicated and must be
        removed.
        """

        df = open_csv(base_df)
        clean_df = clean_duplicated_rows_df(df)
        files_list = get_duplicated_files_list_by_diff(df, clean_df)

        assert files_list == duplicated_files

    def test_remove_duplicates_sequence(self, base_df, tmp_df, tmp_df2):
        """
        We must be able to run the entire clean up sequence without losing
        any data.
        """

        df = pl.read_csv(base_df)
        df.write_csv(tmp_df)

        del df
        gc.collect()


        df = open_csv(tmp_df)
        assert len(df.collect()) == 11
        clean_df = clean_duplicated_rows_df(df)
        assert len(clean_df.collect()) == 8
        duplicated_files = get_duplicated_files_list_by_diff(df, clean_df)
        assert len(duplicated_files) == 3

        save_df(clean_df, tmp_df2)

        assert os.path.exists(tmp_df2)
        assert len(pl.read_csv(tmp_df2)) == 8

        os.remove(tmp_df)
        os.rename(tmp_df2, tmp_df)
        
        assert os.path.exists(tmp_df)
        assert len(pl.read_csv(tmp_df)) == 8
        assert not os.path.exists(tmp_df2)

        del df
        del clean_df
        gc.collect()

        assert os.path.exists(tmp_df)
        assert len(pl.read_csv(tmp_df)) == 8






    # SINCE SAVING A LAZY FRAME AS CSV IN THE SAME FILE IS NOT STABLE, 
    # WE GONNA IGNORE THE TESTS BELLOW. FOR THE PRODUCTION CODE, WE GONNA 
    # SAVE THE UPDATED LAZY FRAME IN A DIFFRERENT FILE AND THEN RENAME IT.
    
    # def test_save_df_with_modifications_same_file(self, base_df, tmp_df):
    #     """
    #     Test if saving in the same file is done incorrectly by polars, 
    #     this time, some modifications are being done before saving.
    #     """

    #     df = pl.read_csv(base_df)
    #     df.write_csv(tmp_df)

    #     df2 = open_csv(tmp_df)
    #     df2 = clean_duplicated_rows_df(df2)
    #     save_df(df2, tmp_df)

    #     target_csv = pl.read_csv(tmp_df)

    #     assert len(target_csv) == 0
     
    # def test_save_df_without_any_modifications_same_file(self, base_df, tmp_df):
    #     """
    #     Test if saving in the same file is done incorrectly by polars, 
    #     even though no modifications were done. It's run some arbitrary amount of
    #     times to check if at least once it fails.
    #     """

    #     df = pl.read_csv(base_df)
    #     df.write_csv(tmp_df)


    #     df2 = open_csv(tmp_df)

    #     # I don't know why we need to do some operation 
    #     # before doing scan + sink csv
    #     df2 = df2.filter() 
    #     save_df(df2, tmp_df)

    #     target_csv = pl.read_csv(tmp_df)

    #     # since it's not stable, it can't do this correctly sometimes
    #     assert len(target_csv) == 0