from data import read_data
from models import split_train_test_samples, train
from models.predict_evaluate_test import predict, evaluate, save_model
from config.create_config_params import read_config, PipelineParams
import click
import json
import logging

logger = logging.getLogger(__name__)
handler = logging.FileHandler('log_file.log')
logger.addHandler(handler)


def run_pipeline(config_path: str, level_log):
    if level_log == "INFO":
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    config_params = read_config(config_path)

    data = read_data(config_params.input_data_path)
    logger.info("Data uploaded")
    data = data[config_params.feature_params.numerical_features]
    logger.debug("Data shape {}".format(data.shape))

    X = data.drop(config_params.feature_params.target, axis=1)
    y = data[config_params.feature_params.target]
    X_train, X_test, y_train, y_test = split_train_test_samples(X, y, config_params.splitting_params)
    logger.debug("train shape {}".format(X_train.shape))
    logger.debug("test shape {}".format(X_test.shape))

    model = train(config_params.train_params, X_train, y_train)
    logger.info("model is ready")
    predicts = predict(model, X_test)
    logger.debug("predicts shape {}".format(predicts.shape))
    score = evaluate(predicts, y_test)

    with open(config_params.metric_path, "w") as metric_file:
        json.dump(score, metric_file)
    logger.info("metrics written to file")
    path_model = save_model(model, config_params.output_model_path)
    logger.info("model saved")

    return path_model, score


@click.command(name="main_pipeline")
@click.argument("config_path")
@click.argument("level_log", default="DEBUG")
def pipeline_command(config_path: str, level_log: str):
    run_pipeline(config_path, level_log)


if __name__ == "__main__":
    pipeline_command()
