"""Custom feature transformers implementation"""
import pandas as pd
from typing import List, Optional


class CustomTransformer:
    def __init__(self, df_train: pd.DataFrame,
                 cat_features: Optional[List[str]] = None,
                 num_features: Optional[List[str]] = None,
                 normalize: Optional[bool] = True):
        if cat_features is None:
            cat_features = []
        if num_features is None:
            num_features = []
        self.cat_features = list(cat_features)
        self.num_features = list(num_features)
        self.input_dim = df_train.shape[1]
        self.normalize = normalize
        self.norm_values = {}
        # pandas.get_dummies can't transform categories which are not included in test df
        # so let's do a custom get_dummies
        self.cat_unique = {}

    def fit(self, df_train: pd.DataFrame):
        if df_train.shape[1] != self.input_dim:
            raise Exception(f"Wrong dataframe dimension! {self.input_dim} expected, got {df_train.shape[1]}!")

        for cat_column in self.cat_features:
            self.cat_unique[cat_column] = sorted(df_train[cat_column].unique())
        if self.normalize:
            for num_column in self.num_features:
                mean = df_train[num_column].mean()
                std = df_train[num_column].std()
                if std == 0:
                    std = 1
                self.norm_values[num_column] = (mean, std)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[1] != self.input_dim:
            raise Exception(f"Wrong dataframe dimension! {self.input_dim} expected, got {df.shape[1]}!")

        result = pd.DataFrame(df)
        for cat_column in self.cat_features:
            for unique_val in self.cat_unique[cat_column]:
                dummy_col = (result[cat_column] == unique_val).astype('int')
                result[f"{cat_column}_{unique_val}"] = dummy_col
        result = result.drop(columns=self.cat_features)
        if self.normalize:
            for num_column in self.num_features:
                mean, std = self.norm_values[num_column]
                result[num_column] = (result[num_column] - mean) / std
        return result


class NoopTransformer:
    def __init__(self, df_train: pd.DataFrame):
        self.input_dim = df_train.shape[1]

    def fit(self, df_train: pd.DataFrame):
        if df_train.shape[1] != self.input_dim:
            raise Exception(f"Wrong dataframe dimension! {self.input_dim} expected, got {df_train.shape[1]}!")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[1] != self.input_dim:
            raise Exception(f"Wrong dataframe dimension! {self.input_dim} expected, got {df.shape[1]}!")
        return pd.DataFrame(df)
