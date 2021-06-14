from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class InferPipelineParams:
    input_data_path: str
    transformer_path: str
    model_path: str
    proba: bool = field(default=False)


InferPipelineParamsSchema = class_schema(InferPipelineParams)


def read_infer_pipeline_params(path: str) -> InferPipelineParams:
    with open(path, "r") as input_stream:
        schema = InferPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
