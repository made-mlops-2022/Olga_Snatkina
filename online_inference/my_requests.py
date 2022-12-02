import requests
import pandas as pd
import json
import os


SERVER = 'http://0.0.0.0:8000/predict'
TARGET = 'condition'


def predict(port=8000):
    DATA_PATH = os.getenv('DATA_PATH')
    df = pd.read_csv(DATA_PATH)
    X = df.drop([TARGET], axis=1).tail(30)
    data_json = X.to_dict(orient="records")
    for request in data_json:
        response = requests.post(SERVER, data=json.dumps(request))
    if response.status_code == 200:
        print(response.json())
        return True
    else:
        print('none response')


if __name__ == '__main__':
    predict()
