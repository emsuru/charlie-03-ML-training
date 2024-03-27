import pandas as pd
import joblib
from preprocessing.data_preprocessor import DataPreprocessor

def load_preprocessor(preprocessor_path):
    """Load the saved preprocessor object."""
    return joblib.load(preprocessor_path)

def load_model(model_path):
    """Load the saved model."""
    return joblib.load(model_path)

def clean_newdata(data_path):
    """Clean the new data using the DataPreprocessor class."""
    df = pd.read_csv(data_path)
    preprocessor = DataPreprocessor(df)
    preprocessor.clean_drop().clean_impute().encode_state_building().encode_epc()
    cleaned_df = preprocessor.df
    return cleaned_df

def preprocess_newdata(cleaned_df, preprocessor_paths):
    """Preprocess the new data using the loaded preprocessor objects."""
    new_df = cleaned_df.copy()

    # Load preprocessing objects
    ohe = joblib.load(preprocessor_paths['onehotencoder'])
    num_imputer = joblib.load(preprocessor_paths['num_imputer'])
    columns_to_keep = joblib.load(preprocessor_paths['columns_to_keep'])

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

    # feature selection based on correlation (this comes from preprocess_feat_select
    # and ensures the new data has the same columns as the model was trained on, filling missing columns with zeros
    new_df = new_df.reindex(columns=columns_to_keep, fill_value=0)

    #numerical imputation
    numeric_cols = new_df.select_dtypes(include=['int64', 'float64']).columns
    new_df[numeric_cols] = num_imputer.transform(new_df[numeric_cols])

    return new_df

def predict(model, X):
    """Make predictions using the preprocessed data and the loaded model."""
    return model.predict(X)

# DEAR USER 1/3: update the 'output_path' to save your predictions with your desired file name

def save_predictions(predictions, output_path='output_data/predictions_250.csv'):
    """Save the predictions to a CSV file."""
    pd.DataFrame(predictions, columns=['PredictedPrice']).to_csv(output_path, index=False)
    print(f"Predictions for Random Forest saved to {output_path}")

# DEAR USER 2/3: update the 'new_data_path' to match the name of your new data

if __name__ == "__main__":
    # 1. Load and clean the new data (using clean & encode methods from the DataPreprocessor class)
    new_data_path = 'input_data/newdata_250.csv'
    cleaned_new_data = clean_newdata(new_data_path)

    # 2. Load the preprocessing objects (that were saved during training, when running Preprocessor)
    preprocessor_paths = {
        'onehotencoder': 'preprocessing/onehotencoder.pkl',
        'num_imputer': 'preprocessing/num_imputer.pkl',
        'columns_to_keep': 'preprocessing/columns_to_keep.pkl'
    }
    preprocessed_new_data = preprocess_newdata(cleaned_new_data, preprocessor_paths)

# DEAR USER 3/3: update the 'model_path' with the saved model you want to use for your predictions

    # 3. Load the model (that was saved during training, when running any model in models/ directory, e.g. RandomForest)
    model_path = 'saved_models/random_forest_model.pkl'
    model = load_model(model_path)

    # 4. Make & save predictions
    predictions = predict(model, preprocessed_new_data)
    save_predictions(predictions)
