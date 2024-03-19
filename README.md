# Immo Charlie Phase 03: Price Predictor

[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## 📖 Description

This Price Predictor is designed to predict residential property prices based on their features.

Takes a dataset of properties and their features as input, runs an ML algorithm on the data and outputs its predictions.

Current project stage: training and evaluating various ML models on the dataset. Will update soon to leave only the best performing model and clean the repo, before moving on to the 4th and last phase of the project.


## 🧬 Project structure

```

charlie-03-ML/
│
├── data/
│   └── properties.csv                --- training dataset
│   └── new_dataset_*.csv             --- new data to predict on
│
├── preprocessing/                    --- data preprocessing code
│   ├── __init__.py
│   └── data_preprocessor.py
│
├── training/                         --- base training code
│   ├── __init__.py
│   └── model_trainer.py
│
├── models/                           --- running base training code with various ML algorithms
│   ├── __init__.py
│   └── train_catboost.py
│   └── train_gradient_boosting.py
│   └── train_linear_regression.py
│   └── train_random_forest.py
│   └── train_xgboost.py
│   └── train_*.py (other models)
│
├── saved_models/                     --- saved models in .pkl format
│   └── model_*.pkl
│
├── predict.py                        --- script for making predictions on new data
└── .gitignore
└── requirements.txt
└── MODELCARD.md
└── README.md
```


Notes: the saved_models folder with the pickle files is not pushed to this repo yet from local machine (need to handle large file uploads to git), the MODELCARD is not yet started, predict.py is work in progress.


## 🛠️ Features

.. to be filled in soon ..


## 👩‍💻 Usage

.. to be filled in soon ..


## 📂 Project background & timeline

.. to be filled in soon ..


## ⚠️ Warning

All my code is currently *heavily*:

- docstringed
- commented

.. and sometimes typed.

This is to help me learn and for my feedback sessions with our coach.

---

Thank you for visiting my project page!

Connect with me on [LinkedIn](https://www.linkedin.com/in/mirunasuru/)
