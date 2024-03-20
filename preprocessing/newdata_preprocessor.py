import pandas as pd
from preprocessing.data_preprocessor import DataPreprocessor

class NewDataPreprocessor(DataPreprocessor):
    def __init__(self, df, training_columns):
        """
        Initializes the NewDataPreprocessor with the new dataset and the columns of the training dataset.
        :param df: The new dataset as a pandas DataFrame.
        :param training_columns: The columns of the training dataset.
        """
        super().__init__(df)
        self.training_columns = training_columns
        self.verify_dataset_structure()

    def verify_dataset_structure(self):
        """
        Verifies that the new dataset has the same structure as the training dataset.
        Adjusts the new dataset to match the training dataset structure if necessary.
        """
        # Ensure the new dataset has the same columns (excluding 'price') as the training dataset
        expected_columns = [col for col in self.training_columns if col != 'price']
        missing_columns = set(expected_columns) - set(self.df.columns)
        extra_columns = set(self.df.columns) - set(expected_columns)

        if missing_columns:
            raise ValueError(f"New dataset is missing columns: {missing_columns}")
        if extra_columns:
            print(f"New dataset has extra columns, which will be ignored: {extra_columns}")
            self.df = self.df[expected_columns]

    def prepare_for_prediction(self):
        """
        Prepares the new dataset for prediction by applying the same preprocessing steps as the training dataset.
        :return: The preprocessed features (X) ready for prediction.
        """
        # Apply the inherited preprocessing methods
        self.clean_drop().clean_impute().clean_encode()

        # If the 'price' column exists, it's ignored since it's not needed for prediction
        if 'price' in self.df.columns:
            self.df.drop(columns=['price'], inplace=True)

        # Ensure the dataset now matches the structure expected by the model
        # This includes any encoding and feature selection applied during training
        # Note: This step assumes that methods like `clean_encode` and `preprocess_feat_select`
        # have been properly implemented in the parent class to not require the target variable.

        # Return the preprocessed dataset
        return self.df
