import pandas as pd
import joblib
from preprocessing.data_preprocessor import DataPreprocessor

def load_preprocessor(preprocessor_path):
    """Load the saved preprocessor object."""
    return joblib.load(preprocessor_path)

def load_model(model_path):
    """Load the saved model."""
    return joblib.load(model_path)

# commented out: an earlier version of the preprocess_data function,
# where we would only have a general preprocessor object with a .transform method that would encapsulate all preprocessing steps

# def preprocess_data(data_path, preprocessor):
#     """Preprocess the new data using the loaded preprocessor."""
#     new_df = pd.read_csv(data_path)
#     # Apply preprocessing transformations
#     preprocessed_data = preprocessor.transform(new_df)
#     return preprocessed_data

def preprocess_data(data_path, preprocessor_paths):
    """Preprocess the new data using the loaded preprocessor objects."""
    new_df = pd.read_csv(data_path)

    # Load preprocessing objects
    ohe = joblib.load(preprocessor_paths['onehotencoder'])
    num_imputer = joblib.load(preprocessor_paths['num_imputer'])
    state_mapping = joblib.load(preprocessor_paths['state_mapping'])
    epc_mapping = joblib.load(preprocessor_paths['epc_mapping'])
    columns_to_keep = joblib.load(preprocessor_paths['columns_to_keep'])

    #Apply cleanup methods first
    preprocessor = DataPreprocessor(new_df)
    preprocessor.clean_drop().clean_impute()
    new_df = preprocessor.df

    # clean_encode() --> Apply mappings for state and epc
    new_df['state_building_encoded'] = new_df['state_building'].map(state_mapping)
    new_df['epc'] = new_df['epc'].map(epc_mapping)

    # Check if the target variable 'price' is in the dataframe and drop it if found
    if 'price' in new_df.columns:
        new_df.drop('price', axis=1, inplace=True)
        print("'price' column found in the dataset. It has been dropped for prediction.")

    # Apply preprocessing transformations
    # categorical encoding
    cat_cols = new_df.select_dtypes(include=['object', 'category']).columns
    new_df_encoded = ohe.transform(new_df[cat_cols])
    new_df_encoded_df = pd.DataFrame(new_df_encoded.toarray(), columns=ohe.get_feature_names_out(cat_cols), index=new_df.index)
    new_df = new_df.drop(columns=cat_cols).join(new_df_encoded_df)

    # feature selection / Ensure the new data has the same columns as the model was trained on, filling missing columns with zeros
    new_df = new_df.reindex(columns=columns_to_keep, fill_value=0)

    #numerical imputation
    numeric_cols = new_df.select_dtypes(include=['int64', 'float64']).columns
    new_df[numeric_cols] = num_imputer.transform(new_df[numeric_cols])

    return new_df

def predict(model, X):
    """Make predictions using the preprocessed data and the loaded model."""
    return model.predict(X)

def save_predictions(predictions, output_path='data/predictions.csv'):
    """Save the predictions to a CSV file."""
    pd.DataFrame(predictions, columns=['PredictedPrice']).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    # Paths to the saved preprocessor and model
    # preprocessor_path = 'saved_preprocessors/data_preprocessor.pkl' # to be updated (if we only have a single pkl encapsulating all preprocessing steps)
    preprocessor_paths = {
        'onehotencoder': 'preprocessing/onehotencoder.pkl',
        'num_imputer': 'preprocessing/num_imputer.pkl',
        'state_mapping': 'preprocessing/state_mapping.pkl',
        'epc_mapping': 'preprocessing/epc_mapping.pkl',
        'columns_to_keep': 'preprocessing/columns_to_keep.pkl'
    }
    model_path = 'saved_models/random_forest_model.pkl' # change this to whatever model we want to use
    new_data_path = 'data/newdata_25.csv'  # Path to the new data

    # Load the model
    model = load_model(model_path)

    # Preprocess the new data and make predictions
    preprocessed_new_data = preprocess_data(new_data_path, preprocessor_paths)
    predictions = predict(model, preprocessed_new_data)

    # Save predictions
    save_predictions(predictions)
