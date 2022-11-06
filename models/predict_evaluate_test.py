import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

SKLEARN_MODEL = [LogisticRegression, RandomForestClassifier]


def predict(model: SKLEARN_MODEL, X_test: pd.DataFrame) -> np.ndarray:
    y_pred = model.predict(X_test)
    return y_pred


def evaluate(y_preds: np.ndarray, y_test: pd.DataFrame) -> float:
    score = accuracy_score(y_test, y_preds)
    return score


def save_model(model: object, path: str) -> str:
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path
