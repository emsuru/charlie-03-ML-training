# Immo Charlie Phase 03: Price Predictor

[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## 📖 Description

This Price Predictor is built to predict residential property prices. It is phase 3 out of a larger project. See phase 1 (data collection) [here](https://github.com/emsuru/charlie-01-data-collection) and phase 2 (data analysis) [here](https://github.com/emsuru/charlie-02-data-analysis). The Predictor takes a dataset of properties and their features as input, runs it through a preprocessing pipeline, uses the preprocessed data to train a machine learning model and finally runs the model on new data to output predictions. 

The model is currently live on the web [at this API endpoint](https://github.com/emsuru/charlie-04-ML-deployment).

## 🧬 Project structure

```

charlie-03-ML/
│
├── input_data/
│   └── properties.csv                --- data: training dataset (.csv)
│   └── newdata.csv                   --- data: new dataset(s) for predictions (.csv)
│
├── preprocessing/                    --- definition code: data preprocessing Class to clean, impute & encode data
│   └── __init__.py
│   └── data_preprocessor.py
│   └── num_imputer.pkl
│   └── *_.pkl (other objects)        --- preprocessing Objects saved in .pkl format to be used in predict.py
│
├── training/                         --- definition code: the base training Class, that all models inherit from
│   └── __init__.py
│   └── model_trainer.py
│
├── models/                           --- execution code: runs base training code with various ML algorithms
│   └── __init__.py
│   └── train_random_forest.py
│   └── train_xgboost.py
│   └── train_*.py (other models)
│
├── saved_models/                     --- trained models: saved in .pkl format to use for predictions (large files, not pushed to git yet)
│   └── model_*.pkl
│
├── predict.py                        --- execution code: runs predictions on new dataset(s) and saves output in .csv format
│
├── output_data/                      --- data: predictions (.csv)
│   └── predictions.csv
│
├── .gitignore
├── requirements.txt
├── MODELCARD.md
└── README.md
```


## Model training

I have iteratively tested 1. various preprocessing pipelines and 2. various algorithms to train and evaluate multiple ML models. I found that the RandomForest Regressor and XGBoost Regressor models performed the best.

Here is an example of the evaluation results for the test set, with three different algorithms:

```

| Model           | MAE         |   RMSE         | R2    |
|                 |             |                |       |
| LinearRegressor | 159,733.50  | 341,468.96     | 0.32  |
| XGBoost         | 101,317.91  | 222,860.80     | 0.71  |
| Random Forest   |  90,923.73  | 210,820.60     | 0.74  |


```
No finetuning has been done, these are the scores at the first attempt.

## 🛠️ Features

- modular OOP-based projects structure, supports training with multiple machine learning algorithmss, for easy experimentation & model comparison
- data preprocessing pipeline (`preprocessing/data_preprocessor.py`) that cleans, imputes, and encodes to prepare for training
- separate `predict.py` script to load trained models, make predictions on new datasets and save predictions in output .csv file


### 🚀 Upcoming Features

- **simplifying preprocessing**: further simplifying the preprocessing pipeline to make it more efficient, using scikit's Pipeline class
- **hyperparameter tuning**: improving the performance of the baseline models


## 👩‍💻 Usage

To use this project for training a model and for making predictions, follow these steps:

### Training a Model

1. Place your training dataset in the `input_data/` directory.
2. Choose the model training script from the `models/` directory (e.g., `train_linear_regression.py`).
3. Run the script to train the model. This will print model evaluation scores to terminal & save the trained model in the `saved_models/` directory.

Please note: do _**not**_ include ".py" in the file name when executing.

Example:

```

python -m models/train_linear_regression

```


### Making Predictions

1. Ensure you have a trained model saved in the `saved_models/` directory.
2. Place the new dataset for prediction in the `input_data/` directory. This dataset should have the same schema as the training dataset.
3. Update the `predict.py` script with the correct paths for the model and the new dataset.
4. Run `predict.py` to preprocess the new dataset and make predictions. Your predictions are saved to a .csv in the `output_data/` directory.

```

python predict.py

```


## 📂 Project background & timeline

This is the third phase of a four-phase project to create a complete ML pipeline for predicting residential property prices. This project phase took one week to complete in March 2024.

The project was completed as part of my 7-month AI training bootcamp at BeCode in Ghent, Belgium.


## ⚠️ Warning

All my code is currently *heavily*:

- docstringed
- commented
- .. and sometimes typed

This is to help me learn and for my feedback sessions with our coach.

---

Thank you for visiting my project page!

Connect with me on [LinkedIn](https://www.linkedin.com/in/mirunasuru/)
