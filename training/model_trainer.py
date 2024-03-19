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
        # Evaluation logic here
        return evaluation_metrics
        pass

    def save_model(self, filename):
        joblib_file = f"saved_models/{filename}"
        joblib.dump(self.model, joblib_file)
        print(f"Model saved to {joblib_file}")
