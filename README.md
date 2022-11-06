# Olga_Snatkina
MADE MLOPS

**Installation:**
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
**Usage:**

should specify the path to the config and the level of logging
```
 python train_pipeline.py config/config.yaml DEBUG 
 python predict_pipeline.py config/config.yaml INFO
 ```

**Test**:
```
 python -m unittest test/tests_*.py
 ```

```
├── README.md          <- The top-level README for developers using this project.
├── data               <- Directory with dataset for training and saving predictions.
│   └── create_dataset <- Upload csv.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
|   ├── __init__.   <- Makes src a Python module
│   ├── fit_models  <- splitting to train and test, train model
│   │
│   ├── predict_evaluate_test      <- predict, evaluate, save model
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── config             <- Create config params using dataclasses and yaml files.
│   ├── create_config_params        <- Read config, create dataclasses schema.
│   |  
│   ├── config.yaml    <- config file for model LogisticRegression 
│   |  
│   ├── config2.yaml   <- config file for model RandomForestClassifier
|
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── tests              <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── create_faker_dataset          <- script for generating synthetic data. 
│   │
│   ├── tests_data     <- test for data modules
│   │
│   ├── tests_data     <- test for model modules
│   │
├── predict_pipeline   <- pipeline module for predict data
└── train_pipeline     <- pipeline module for train models
```
