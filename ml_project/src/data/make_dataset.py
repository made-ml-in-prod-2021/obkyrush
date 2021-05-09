import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

from src.entities import SplittingParams, FeatureParams


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def extract_target(data: pd.DataFrame,
                   params: FeatureParams) -> Tuple[pd.DataFrame,
                                                pd.Series]:
    target = data[params.target_col]
    return data.drop(columns=[params.target_col]), target


def split_train_val_data(
        data: pd.DataFrame,
        target: pd.Series,
        params: SplittingParams) -> Tuple[pd.DataFrame, pd.Series,
                                          pd.DataFrame, pd.Series]:
    """
    :rtype: object
    """
    X_train, X_test, y_train, y_test,  = train_test_split(
        data, target, test_size=params.test_size, random_state=params.random_state
    )
    return X_train, y_train, X_test, y_test
