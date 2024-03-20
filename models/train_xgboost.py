import pandas as pd
import xgboost as xgb
from preprocessing.data_preprocessor import DataPreprocessor
from training.model_trainer import ModelTrainer

def train_xgboost():
    df = pd.read_csv('data/properties.csv')
    preprocessor = DataPreprocessor(df)
    preprocessor.clean_drop().clean_impute().clean_encode()
    preprocessor.save_training_columns('training/training_columns.txt')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_split()
    X_train, X_test = preprocessor.preprocess_encode(X_train, X_test)
    X_train, X_test = preprocessor.preprocess_feat_select(X_train, X_test, y_train)
    X_train, X_test = preprocessor.preprocess_impute(X_train, X_test)

    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    trainer.train_model(xgb.XGBRegressor(objective ='reg:squarederror'))
    trainer.evaluate_model()
    trainer.save_model("xgboost_model.pkl")

if __name__ == "__main__":
    train_xgboost()
