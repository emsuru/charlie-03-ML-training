import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None

    def train_model(self, model):
        self.model = model
        self.model.fit(self.X_train, self.y_train)
        return self

    def evaluate_model(self):
        #Make predicions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # training set evaluation
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_rmse = mean_squared_error(self.y_train, y_train_pred, squared=False)
        train_r2 = r2_score(self.y_train, y_train_pred)

        print("Training Set Evaluation:")
        print(f"Mean Absolute Error: {train_mae}")
        print(f"Mean Squared Error: {train_mse}")
        print(f"Root Mean Squared Error: {train_rmse}")
        print(f"R-squared: {train_r2}")

        # testing set evaluation
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_rmse = mean_squared_error(self.y_test, y_test_pred, squared=False)
        test_r2 = r2_score(self.y_test, y_test_pred)

        print("\nTest Set Evaluation:")
        print(f"Mean Absolute Error: {test_mae}")
        print(f"Mean Squared Error: {test_mse}")
        print(f"Root Mean Squared Error: {test_rmse}")
        print(f"R-squared: {test_r2}")

    def save_model(self, filename):
        joblib_file = f"saved_models/{filename}"
        joblib.dump(self.model, joblib_file)
        print(f"Model saved to {joblib_file}")
