"""File to hold any fixtures that would be used in different tests"""

import os
import shutil

import pytest

@pytest.fixture
def target_folder() -> str:
    """Target data folder"""
    return os.path.join("tests", "data")

@pytest.fixture
def checkpoint_path(target_folder) -> str:
    """Fake checkpoint file"""
    return os.path.join(target_folder, "checkpoint.json")

@pytest.fixture
def df_path(target_folder) -> str:
    """A dummy csv file for testing"""
    return os.path.join(target_folder, "test_df.csv")

@pytest.fixture
def images_path(target_folder) -> str:
    """A dummy folder for storing generated images"""
    return os.path.join(target_folder, "images")

def clean(folder) -> None:
    if not os.path.exists(folder):
        return
    
    shutil.rmtree(folder)


@pytest.fixture(autouse=True)
def clean_up(target_folder):
    """Clean checkpoint files"""

    #before tests
    clean(target_folder)

    os.makedirs(target_folder, exist_ok=True)

    yield
    
    #after tests
    clean(target_folder)