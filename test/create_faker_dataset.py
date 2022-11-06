import csv
from faker import Faker


def datagenerate(records, headers):
    fake = Faker()
    with open("fake.csv", 'wt') as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=headers)
        writer.writeheader()
        for i in range(records):
            writer.writerow({
                "age": fake.pyint(min_value=0, max_value=100),
                "sex": fake.pyint(min_value=0, max_value=1),
                "cp": fake.pyfloat(min_value=0, max_value=3),
                "trestbps": fake.pyfloat(min_value=90, max_value=250),
                "chol": fake.pyfloat(min_value=120, max_value=600),
                "fbs": fake.pyint(min_value=0, max_value=1),
                "restecg": fake.pyfloat(min_value=0, max_value=2),
                "thalach": fake.pyfloat(min_value=70, max_value=200),
                "exang": fake.pyint(min_value=0, max_value=1),
                "oldpeak": fake.pyfloat(min_value=0, max_value=7),
                "slope": fake.pyfloat(min_value=0, max_value=2),
                "ca": fake.pyfloat(min_value=0, max_value=3),
                "thal": fake.pyfloat(min_value=0, max_value=2),
                "condition": fake.pyint(min_value=0, max_value=1)
            })


if __name__ == '__main__':
    records = 500
    headers = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'condition']
    datagenerate(records, headers)
    print("CSV generation complete!")