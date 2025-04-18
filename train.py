import pandas as pd 
from PIL import Image
import json

from constants import DATASET_FILE

if __name__ == "__main__":
    df = pd.read_csv(DATASET_FILE)
    print(df.head())

    print(json.loads(df.loc[0][-2])["0"])
