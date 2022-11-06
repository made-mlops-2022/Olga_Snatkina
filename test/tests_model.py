from unittest import TestCase
from numpy import ndarray
from data.create_dataset import read_data
from models.fit_models import train
from models.predict_evaluate_test import save_model, predict, evaluate
from config.create_config_params import TrainParams
from sklearn.linear_model import LogisticRegression
import os


class TestDataPrep(TestCase):
    def test_train_data(self):
        path_f = "test/fake.csv"
        print('test1')
        df = read_data(path_f)
        X = df.drop(["condition"], axis=1)
        y = df.condition
        model = train(TrainParams, X, y)
        self.assertIsInstance(model, LogisticRegression)

        path_t = "model_test.pkl"
        save_model(model, path_t)
        self.assertTrue(os.path.exists(path_t))

    def test_predict_eval(self):
        path_f = "test/fake.csv"
        print('tyt')
        df = read_data(path_f)
        X = df.drop(["condition"], axis=1)
        y = df.condition
        model = train(TrainParams, X, y)

        y_preds = predict(model, X)
        score = evaluate(y_preds, y)
        self.assertGreater(score, 0)
        self.assertEqual(y_preds.shape, y.shape)
        self.assertIsInstance(y_preds, ndarray)
