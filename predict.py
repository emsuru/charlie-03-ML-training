## --- !!! STILL VERY MUCH WORK IN PROGRESS !!! AAAAARGH ---

import pandas as pd
import joblib
from preprocessing.newdata_preprocessor import NewDataPreprocessor


def load_model(model_path):
    return joblib.load(model_path)

def load_training_columns(filepath):
    with open(filepath, 'r') as file:
        columns = [line.strip() for line in file.readlines()]
    return columns

def preprocess_data(data_path):
    new_df = pd.read_csv(data_path)
    # # Initialize the NewDataPreprocessor with the new dataset and training columns
    # new_data_preprocessor = NewDataPreprocessor(new_df, training_columns)
    # Prepare the new dataset for prediction
    # X_new = new_data_preprocessor.prepare_for_prediction()
    return X_new

def predict(model, X):
    return model.predict(X)

if __name__ == "__main__":
    model_path = "saved_models/xgboost_model.pkl"  # change with any other model from saved_models/
    new_data_path = "data/properties_2.csv"  # update this if different path

    training_columns = load_training_columns('training/training_columns.txt')

    model = load_model(model_path)
    X_new = preprocess_data(new_data_path)
    predictions = predict(model, X_new)

    # Save or print predictions
    # Ensure predictions is a DataFrame or convert it to one before saving
    pd.DataFrame(predictions).to_csv('data/predictions.csv', index=False)
    print("Predictions saved to predictions.csv")
    #print(predictions)
