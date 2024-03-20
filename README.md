# Immo Charlie Phase 03: Price Predictor

[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## 📖 Description

This Price Predictor is built to predict residential property prices.

It takes a dataset of properties and their features as input, runs it through an ML pipeline, and outputs its predictions.


## 🧬 Project structure

```

charlie-03-ML/
│
├── data/
│   └── properties.csv                --- data: training dataset
│   └── newdata.csv                   --- data: new dataset(s) for predictions
│
├── preprocessing/                    --- definition code for data preprocessing classes
│   └── __init__.py
│   └── data_preprocessor.py
│   └── newdata_preprocessor.py
│
├── training/                         --- definition code for base training class
│   └── __init__.py
│   └── model_trainer.py
│   └── training_columns.txt
│
├── models/                           --- execution code: runs training code on several ML algorithms
│   └── __init__.py
│   └── train_catboost.py
│   └── train_gradient_boosting.py
│   └── train_linear_regression.py
│   └── train_random_forest.py
│   └── train_xgboost.py
│   └── train_*.py (other models)
│
├── saved_models/                     --- saved models in .pkl format (not pushed to repo yet)
│   └── model_*.pkl
│
├── predict.py                        --- execution code: runs predictions on new dataset(s)
├── .gitignore
├── requirements.txt
├── MODELCARD.md
└── README.md
```


## Project Status:

Currently training and evaluating various ML models. Will update soon to leave only the best performing model and clean the repo, before moving on to the 4th and last phase of the project.

- the saved_models folder with the pickle files is not pushed to this repo yet from local machine (need to handle large file uploads to git)
- the MODELCARD is work-in-progress
- predict.py is work-in-progress



## 🛠️ Features

- supports training with various machine learning algorithmss, for easy experimentation & model comparison
- data preprocessing pipeline (`preprocessing/data_preprocessor.py`) that cleans, imputes, and encodes to rpepare for training
- new data validation & preprocessing, to prepare new data sets for prediction (`preprocessing/newdata_preprocessor.py`)
- separate `predict.py` script to load trained models, make predictions on new datasets and save predictions in output .csv file
- modular design 

### 🚀 Upcoming Features

- **hyperparameter tuning**: improving the performance of the baseline models 
- **automated new data validation**: futher automating newdata validation checks against the model's expected input schema.
- **API deployment**: this will the final development stage for this four-stage learning project 


## 👩‍💻 Usage

To use this project for training a model or making predictions, follow these steps:

### Training a Model

1. Place your training dataset in the `data/` directory.
2. Choose the model training script from the `models/` directory (e.g., `train_linear_regression.py`).
3. Run the script to train the model. This will also save the trained model in the `saved_models/` directory.

Example:

```

python3 models/train_xgboost (# note: do not include .py when running)

```


### Making Predictions

1. Ensure you have a trained model saved in the `saved_models/` directory.
2. Place the new dataset for prediction in the `data/` directory.
3. Update the `predict.py` script with the correct paths for the model and the new dataset.
4. Run `predict.py` to preprocess the new dataset and make predictions. This will print predictions to the console or save to a file (--_to be updated once I finalise writing predict.py_--))


```

python3 predict.py

```


## 📂 Project background & timeline

This is the third phase of a four-phase project to create a complete ML pipeline for predicting residential property prices. This project phase took one week to complete in March 2024.

The project was completed as part of my 7-month AI training bootcamp at BeCode in Ghent, Belgium.


## ⚠️ Warning

All my code is currently *heavily*:

- docstringed
- commented

.. and sometimes typed.

This is to help me learn and for my feedback sessions with our coach.

---

Thank you for visiting my project page!

Connect with me on [LinkedIn](https://www.linkedin.com/in/mirunasuru/)
