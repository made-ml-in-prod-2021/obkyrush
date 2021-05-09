import json
import logging
import os
import sys
from pathlib import Path

import click
import pandas as pd


from data import read_data, split_train_val_data, extract_target
from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from src.features import build_features
from src.features.build_features import make_features, build_transformer
from src.models import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)

from src.models.model_fit_predict import create_inference_pipeline

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)
    return run_train_pipeline(training_pipeline_params)


def run_train_pipeline(training_pipeline_params):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    data, target = extract_target(data, params=training_pipeline_params.feature_params)
    X_train, y_train, X_val, y_val = split_train_val_data(
        data, target, training_pipeline_params.splitting_params
    )

    logger.info(f"X_train.shape is {X_train.shape}")
    logger.info(f"X_test.shape is {X_val.shape}")
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(X_train)
    X_train_processed = make_features(transformer, X_train)
    logger.info(f"train_features.shape is {X_train_processed.shape}")
    model = train_model(
        X_train_processed, y_train, training_pipeline_params.train_params
    )

    inference_pipeline = create_inference_pipeline(model, transformer)
    predicts = predict_model(
        inference_pipeline,
        X_val
    )
    metrics = evaluate_model(
        predicts,
        y_val
    )
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"metrics is {metrics}")

    path_to_model = serialize_model(
        inference_pipeline, training_pipeline_params.output_model_path
    )
    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
