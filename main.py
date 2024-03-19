import pandas as pd
from preprocessing.data_preprocessor import DataPreprocessor
from training.model_trainer import ModelTrainer
from sklearn.linear_model import LinearRegression

def main():
    df = pd.read_csv('data/properties.csv')
    preprocessor = DataPreprocessor(df)
    # Assume preprocess_split is a method that returns the split data
    X_train, X_test, y_train, y_test = preprocessor.preprocess_split()

    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    trainer.train_model(LinearRegression())
    # Assume evaluate_model prints or returns evaluation metrics
    trainer.evaluate_model()
    trainer.save_model("linear_regression_model.pkl")

if __name__ == "__main__":
    main()
