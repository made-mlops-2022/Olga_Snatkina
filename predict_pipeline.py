import pandas as pd

from data import read_data
from models.predict_evaluate_test import predict
from config.create_config_params import read_config, PipelineParams
import click
import logging
import pickle


logger = logging.getLogger(__name__)
handler = logging.FileHandler('log_file.log')
logger.addHandler(handler)


def run_pipeline(config_path: str, level_log):
    if level_log == "INFO":
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    config_params = read_config(config_path)

    data = read_data(config_params.input_test_data_path)
    logger.info("Data for predict uploaded")
    data = data[config_params.feature_params.numerical_features]
    logger.debug("Data shape {}".format(data.shape))
    y_col = config_params.feature_params.target
    if y_col in data.columns:
        X = data.drop(config_params.feature_params.target, axis=1)
        y = data[config_params.feature_params.target]
    else:
        X = data
        y = None

    with open(config_params.output_model_path, "rb") as f:
        model = pickle.load(f)
    logger.info("model is uploaded")
    predicts = predict(model, X)
    logger.debug("predicts shape {}".format(predicts.shape))

    pd.DataFrame(predicts).to_csv(config_params.path_for_predicts, index=False)
    logger.info("predictions written to file")

    return config_params.path_for_predicts


@click.command(name="predict_pipeline")
@click.argument("config_path")
@click.argument("level_log", default="DEBUG")
def pipeline_command(config_path: str, level_log: str):
    run_pipeline(config_path, level_log)


if __name__ == "__main__":
    pipeline_command()
