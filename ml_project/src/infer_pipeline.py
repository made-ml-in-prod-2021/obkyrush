import pickle
import logging
import sys

import click
import pandas as pd


from data import read_data
from src.entities.infer_pipeline_params import (
    InferPipelineParams,
    read_infer_pipeline_params,
)
from src.features import build_features
from src.features.build_features import make_features
from src.models import predict_model


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def infer_pipeline(config_path: str):
    infer_pipeline_params = read_infer_pipeline_params(config_path)
    return run_infer_pipeline(infer_pipeline_params)


def run_infer_pipeline(infer_pipeline_params):
    logger.info(f"start infer pipeline with params {infer_pipeline_params}")
    data = read_data(infer_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    with open(infer_pipeline_params.transformer_path, 'rb') as f_transformer:
        transformer = pickle.load(f_transformer)
    data_processed = make_features(transformer, data)

    with open(infer_pipeline_params.model_path, 'rb') as f_model:
        model = pickle.load(f_model)

    predicts = predict_model(model, data_processed, proba=infer_pipeline_params.proba)
    with open(infer_pipeline_params.output_path, "w") as f_result:
        f_result.write(predicts)


@click.command(name="infer_pipeline")
@click.argument("config_path")
def infer_pipeline_command(config_path: str):
    infer_pipeline(config_path)


if __name__ == "__main__":
    infer_pipeline_command()
