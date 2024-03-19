## --- !!! STILL WORK IN PROGRESS !!! ---

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib  # If using sklearn version < 0.24, else use 'from joblib import load'
from preprocessing.data_preprocessor import DataPreprocessor

def load_model(model_path):
    return joblib.load(model_path)

def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    preprocessor = DataPreprocessor(df)
    preprocessor.clean_drop().clean_impute().clean_encode()
    # Assuming preprocess_split returns just the features (X) for prediction (???)
    X, _ = preprocessor.preprocess_split(predict_mode=True)
    return X

def predict(model, X):
    return model.predict(X)

if __name__ == "__main__":
    model_path = "linear_regression_model.pkl" # this is customizab;e so I change it with different models
    new_data_path = "path/to/new/dataset.csv" # I need to update this

    model = load_model(model_path)
    X_new = preprocess_data(new_data_path)
    predictions = predict(model, X_new)

    # Optionally, save or print predictions
    print(predictions)


## ----

# 1. Preparing for a New Dataset
# When I receive a new dataset:
# -- ensure it's in CSV format and has the same structure (same columns) as the training data, except for the target variable.
# -- place it in an accessible location and update the new_data_path variable in predict.py accordingly.

# 2. Running Predictions
# To run predictions on the new dataset:
# -- Open a terminal or command prompt.
# -- Navigate to the directory containing predict.py.
# -- Run the script using Python. ex: python3 predict.py

# This will load the model, preprocess the new dataset, and print the predictions to the console.
# Can update this to save the predictions to a file if required.
# Note
# haev to make sure that all custom modules (DataPreprocessor in this case) and their methods used in preprocessing are accessible and work in the same way as during training.
# When switching models (e.g., from LinearRegression to another), replace the model loading in predict.py accordingly.
# The preprocessing steps should remain consistent across models, assuming they require the data in the same format.
