from dataclasses import dataclass, field
import typing as tp
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class DataParams:
    val_size: float = field(default=0.1)
    random_state: int = field(default=0)


@dataclass()
class TrainParams:
    random_state: int = field(default=0)
    model_type: str = field(default="LogisticRegression")
    max_iter: int = field(default=500)


@dataclass()
class FeatureParams:
    numerical_features: tp.List[str]
    target: tp.Optional[str]


@dataclass()
class PipelineParams:
    input_data_path: str
    output_model_path: str
    input_test_data_path: str
    path_for_predicts: str
    metric_path: str
    splitting_params: DataParams
    train_params: TrainParams
    feature_params: FeatureParams


PipelineParams_schema = class_schema(PipelineParams)


def read_config(path: str) -> PipelineParams:
    with open(path, 'r') as cn:
        schema = PipelineParams_schema()
        return schema.load(yaml.safe_load(cn))
