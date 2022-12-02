from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from config.create_config_params import DataParams, TrainParams

SKLEARN_MODEL = [LogisticRegression, RandomForestClassifier]


def split_train_test_samples(X: pd.DataFrame, y: pd.DataFrame, params: DataParams) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.val_size,
                                                        random_state=params.random_state)
    return X_train, X_test, y_train, y_test


def train(params: TrainParams, X_train: pd.DataFrame, y_train: pd.DataFrame) -> SKLEARN_MODEL:
    if params.model_type == "LogisticRegression":
        model = LogisticRegression(
            random_state=params.random_state,
            max_iter=params.max_iter
        )
    elif params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=params.random_state,
        )
    else:
        raise NotImplementedError()
    model.fit(X_train, y_train)
    return model


