import pandas as pd
from preprocessing.data_preprocessor import DataPreprocessor
from training.model_trainer import ModelTrainer
from sklearn.linear_model import LinearRegression

def main():
    df = pd.read_csv('data/properties.csv')
    preprocessor = DataPreprocessor(df) # Initialize my preprocessor with the dataset

    # Apply cleanup steps, safe to perform on the entire dataset (no data leakage)
    preprocessor.clean_drop()  # Drop duplicates and **unequivocally** unnecessary columns
    preprocessor.clean_impute()  # Impute missing values with "missing" for categorical columns
    preprocessor.clean_encode()  # Encode categorical columns with binning

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = preprocessor.preprocess_split()

    # Apply preprocessing steps to the training and testing sets separately, to avoid data leakage
    X_train, X_test = preprocessor.preprocess_encode(X_train, X_test) # Encode categorical columns
    X_train, X_test = preprocessor.preprocess_feat_select(X_train, X_test, y_train)   # Feature selection based on correlation of numerical feat in trainingset w/ target variable
    X_train, X_test = preprocessor.preprocess_impute(X_train, X_test) # Imputing missing values with median of trainingset for numerical columns

    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    trainer.train_model(LinearRegression())
    trainer.evaluate_model()
    trainer.save_model("linear_regression_model.pkl")



if __name__ == "__main__":
    main()
