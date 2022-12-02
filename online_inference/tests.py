import json
import os
import pytest
from fastapi.testclient import TestClient
from fastapi_app import ml_project_app, load_model
import httpx

client = TestClient(ml_project_app)


@pytest.fixture(scope="session", autouse=True)
def initialize_model():
    os.environ["MODEL_PATH"] = "models/model.pkl"
    load_model()


def test_predict() -> None:
    request = {
        "age": 48,
        "sex": 0, "cp": 2,
        "trestbps": 20,
        "chol": 100,
        "fbs": 0,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 8,
        "slope": 1,
        "ca": 1,
        "thal": 1
    }
    response = client.post(url="/predict", content=json.dumps(request))
    assert response.status_code == 200
    assert response.json() == {"prediction": "no"}


def test_predict_two() -> None:
    request = {
        "age": 49,
        "sex": 1,
        "cp": 2,
        "trestbps": 20,
        "chol": 110,
        "fbs": 0,
        "restecg": 2,
        "thalach": 155,
        "exang": 0,
        "oldpeak": 7,
        "slope": 1,
        "ca": 2,
        "thal": 1
    }
    response = client.post(url="/predict", content=json.dumps(request))
    assert response.status_code == 200
    assert response.json() == {"prediction": "yes"}


def test_invalid_value() -> None:
    request = {
        "age": 49,
        "sex": 2,
        "cp": 2,
        "trestbps": 20,
        "chol": 110,
        "fbs": 0, "restecg": 2,
        "thalach": 155,
        "exang": 0,
        "oldpeak": 7,
        "slope": 1,
        "ca": 2,
        "thal": 1
    }
    response = client.post(url="/predict", content=json.dumps(request))
    assert response.status_code == 422
    assert (
        response.json()["detail"][0]["msg"]
        == "unexpected value; permitted: 0, 1"
    )


if __name__ == "__main__":
    initialize_model()
    test_predict()
    test_predict_two()
    test_invalid_value()
