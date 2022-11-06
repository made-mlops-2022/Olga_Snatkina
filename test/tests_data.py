from unittest import TestCase
from pandas import DataFrame
from data.create_dataset import read_data
from models.fit_models import split_train_test_samples
from config.create_config_params import DataParams


class TestDataPrep(TestCase):
    def test_read_data(self):
        path_f = "test/fake.csv"
        print('test1')
        df = read_data(path_f)
        self.assertIsInstance(df, DataFrame)
        self.assertEqual(500, df.shape[0])
        self.assertIn("condition", df.columns)

    def test_splitting_data(self):
        path_f = "test/fake.csv"
        print('tyt')
        df = read_data(path_f)
        X = df.drop(["condition"], axis=1)
        y = df.condition
        x_train, x_test, y_train, y_test = split_train_test_samples(X, y, DataParams)
        self.assertEqual(x_train.shape, (450, 13))
        self.assertEqual(x_test.shape, (50, 13))
        self.assertEqual(y_train.shape, (450,))
        self.assertEqual(y_test.shape, (50,))
