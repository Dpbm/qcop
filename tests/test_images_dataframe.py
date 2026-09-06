import os
import pytest

import pandas as pd

from generate.images import DF

class TestDF:
    def test_df_doesnt_exist(self,df_path):
        df = DF(df_path)

        assert df._df_path == df_path
        assert df._df is not None
        assert tuple(df._df.columns) == df._columns
    
    def test_df_already_exist(self,df_path):
        df_tmp = pd.DataFrame({"test":[1,2,3]})
        df_tmp.to_csv(df_path,index=False)

        df = DF(df_path)
        assert df._df_path == df_path
        assert list(df._df.columns) == ["test"]
        assert df._current_index == 3

    def test_append_rows(self,df_path):
        df = DF(df_path)
        rows = [
            {   
                key:i
                for key in df._columns
            }
            for i in range(10)
        ]
        df.append_rows_to_file(rows)

        assert len(df._df) == 10
        assert df._current_index == 10

    def test_save_df(self,df_path):
        df = DF(df_path)
        rows = [
            {   
                key:i
                for key in df._columns
            }
            for i in range(10)
        ]
        df.append_rows_to_file(rows)
        df.save_df()

        df_tmp = pd.read_csv(df_path)
        assert len(df_tmp) == 10
