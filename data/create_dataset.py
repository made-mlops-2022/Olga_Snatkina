import typing as tp
import pandas as pd


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
