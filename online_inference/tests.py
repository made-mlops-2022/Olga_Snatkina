import json
import os
import pytest
from fastapi.testclient import TestClient
from fastapi_app import predict, load_model, ml_project_app

client = TestClient(ml_project_app)


@pytest.fixture(scope="session", autouse=True)
def initialize_model() -> None:
    assert os.environ["MODEL_PATH"] == "/models/model.pkl"
    load()


def test_missing_features() -> None:
    request = {
        "age": 48, "sex": 0,"cp": 2,"trestbps": 20, "chol": 100, "fbs": 0, "restecg": 2, "thalach": 150,"exang": 0, "oldpeak": 8, "slope": 1, "ca": 1, "thal": 1
    }
    response = client.post(url="/predict", content=json.dumps(request))
    assert response.status_code == 200
    assert response.json() == {"preds":[0]}


def test_numerical_features() -> None:
    request = {
        "mintemp": 13.1,
        "maxtemp": 30.1,
        "rainfall": 15.4,
        "evaporation": 0.0,
        "sunshine": 0.0,
        "windgustdir": "W",
        "windgustspeed": 28.0,
        "winddir9am": "S",
        "winddir3pm": "SSE",
        "windspeed9am": 15.0,
        "windspeed3pm": 11.0,
        "humidity9am": 101.0,
        "humidity3pm": 45.0,
        "pressure9am": 1007.0,
        "pressure3pm": 1005.7,
        "cloud9am": 0.0,
        "cloud3pm": 0.0,
        "temp9am": 20.1,
        "temp3pm": 28.2,
        "raintoday": "No",
    }
    response = client.post(url="/predict", content=json.dumps(request))
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Wrong humidity value"


def test_categorical_features() -> None:
    request = {
        "mintemp": 13.1,
        "maxtemp": 30.1,
        "rainfall": 15.4,
        "evaporation": 0.0,
        "sunshine": 0.0,
        "windgustdir": "West",
        "windgustspeed": 28.0,
        "winddir9am": "South",
        "winddir3pm": "South-south-west",
        "windspeed9am": 15.0,
        "windspeed3pm": 11.0,
        "humidity9am": 51.0,
        "humidity3pm": 75.0,
        "pressure9am": 1007.0,
        "pressure3pm": 1005.7,
        "cloud9am": 0.0,
        "cloud3pm": 0.0,
        "temp9am": 20.1,
        "temp3pm": 28.2,
        "raintoday": "Maybe...",
    }
    response = client.post(url="/predict", content=json.dumps(request))
    assert response.status_code == 422
    assert (
        response.json()["detail"][0]["msg"]
        == "unexpected value; permitted: 'W', 'NNW', "
        "'ENE', 'SSE', 'S', 'NE', 'SSW', "
        "'N', 'WSW', 'SE', 'ESE', 'E', 'NW', 'NNE', 'SW', 'WNW'"
    )


def test_predict_is_rain() -> None:
    request = {
        "mintemp": 13.1,
        "maxtemp": 30.1,
        "rainfall": 15.4,
        "evaporation": 0.0,
        "sunshine": 0.0,
        "windgustdir": "W",
        "windgustspeed": 28.0,
        "winddir9am": "S",
        "winddir3pm": "SSE",
        "windspeed9am": 15.0,
        "windspeed3pm": 11.0,
        "humidity9am": 51.0,
        "humidity3pm": 75.0,
        "pressure9am": 1007.0,
        "pressure3pm": 1005.7,
        "cloud9am": 0.0,
        "cloud3pm": 0.0,
        "temp9am": 20.1,
        "temp3pm": 28.2,
        "raintoday": "Yes",
    }
    response = client.post(url="/predict", content=json.dumps(request))
    assert response.status_code == 200
    assert response.json() == {"Rain": "YES"}


def test_predict_no_rain() -> None:
    request = {
        "mintemp": 13.1,
        "maxtemp": 30.1,
        "rainfall": 15.4,
        "evaporation": 0.0,
        "sunshine": 0.0,
        "windgustdir": "W",
        "windgustspeed": 28.0,
        "winddir9am": "S",
        "winddir3pm": "SSE",
        "windspeed9am": 15.0,
        "windspeed3pm": 11.0,
        "humidity9am": 51.0,
        "humidity3pm": 45.0,
        "pressure9am": 1007.0,
        "pressure3pm": 1005.7,
        "cloud9am": 0.0,
        "cloud3pm": 0.0,
        "temp9am": 20.1,
        "temp3pm": 28.2,
        "raintoday": "No",
    }
    response = client.post(url="/predict", content=json.dumps(request))
    assert response.status_code == 200
    assert response.json() == {"Rain": "NO"}


if __name__ == "__main__":
    test_numerical_features()
    test_categorical_features()
    test_predict_is_rain()
    test_predict_no_rain()
    test_missing_features()