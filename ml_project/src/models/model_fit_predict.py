import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline

from src.entities.model_params import TrainingParams

SklearnRegressionModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnRegressionModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=100, random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(model: Pipeline, features: pd.DataFrame, proba=False) -> np.ndarray:
    if proba:
        preds = model.predict_proba(features)[:, 1]
    else:
        preds = model.predict(features)
    return preds


def evaluate_model(probs: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "roc_auc": roc_auc_score(target, probs),
        "accuracy": accuracy_score(target, probs),
    }


def create_inference_pipeline(
    model: SklearnRegressionModel, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
