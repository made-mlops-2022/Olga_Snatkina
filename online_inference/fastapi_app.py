from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import pickle
import pandas as pd
import logging
import os


ml_project_app = FastAPI()

logger = logging.getLogger()
logging.basicConfig(
    filename="./app_log.log",
    level=logging.INFO
)


class DatasetType(BaseModel):
    age: float
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: float
    chol: float
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]


@ml_project_app.on_event("startup")
def load_model() -> None:
    global MODEL, MODEL_PATH
    MODEL_PATH = os.getenv("MODEL_PATH")
    MODEL = None
    try:
        with open(MODEL_PATH, 'rb') as file:
            MODEL = pickle.load(file)
            logger.info('Model loaded')
    except FileNotFoundError:
        logger.info('Model path not exist')


@ml_project_app.post('/predict')
def predict(data: DatasetType):
    logger.info('Got request')
    data_df = pd.DataFrame.from_records([data.dict()])
    preds = MODEL.predict(data_df)
    logger.info('predictions ready')
    logger.info('Send prediction')
    response = "yes" if preds[0] else "no"
    return {"prediction": response}


@ml_project_app.get('/')
def root() -> dict:
    return {'message': 'Hello world!'}


@ml_project_app.get('/health')
def health() -> int:
    if MODEL is not None:
        return 200
