## --- !!! STILL VERY MUCH WORK IN PROGRESS !!! AAAAARGH ---

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib  # If using sklearn version < 0.24, else use 'from joblib import load'
from preprocessing.data_preprocessor import DataPreprocessor
from preprocessing.newdata_preprocessor import NewDataPreprocessor

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
    model_path = "xgboost_model.pkl" # change with any other model from saved_models/
    new_data_path = "data/newdata.csv" # update this if different path

    model = load_model(model_path)
    X_new = preprocess_data(new_data_path)
    predictions = predict(model, X_new)

   # Save or print predictions
    predictions.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")
    #print(predictions)


## ----
