"""File to hold any fixtures that would be used in different tests"""

import os
import shutil

import pytest

@pytest.fixture
def checkpoint_path() -> str:
    """Fake checkpoint file"""
    return os.path.join("tests", "checkpoint.json")

@pytest.fixture
def df_path() -> str:
    """A dummy csv file for testing"""
    return os.path.join("tests", "test_df.csv")

@pytest.fixture
def images_path() -> str:
    """A dummy folder for storing generated images"""
    return os.path.join("tests", "images")

def clean(*args) -> None:

    for path in args:
        if not os.path.exists(path):
            continue

        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

@pytest.fixture(autouse=True)
def clean_up(checkpoint_path, df_path, images_path):
    """Clean checkpoint files"""

    #before tests
    clean(checkpoint_path, df_path, images_path)

    yield
    
    #after tests
    clean(checkpoint_path, df_path, images_path)