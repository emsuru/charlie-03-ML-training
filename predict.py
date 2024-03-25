import pandas as pd
import joblib

def load_preprocessor(preprocessor_path):
    """Load the saved preprocessor object."""
    return joblib.load(preprocessor_path)

def load_model(model_path):
    """Load the saved model."""
    return joblib.load(model_path)

def preprocess_data(data_path, preprocessor):
    """Preprocess the new data using the loaded preprocessor."""
    new_df = pd.read_csv(data_path)
    # Apply preprocessing transformations
    preprocessed_data = preprocessor.transform(new_df)
    return preprocessed_data

def predict(model, X):
    """Make predictions using the preprocessed data and the loaded model."""
    return model.predict(X)

def save_predictions(predictions, output_path='data/predictions.csv'):
    """Save the predictions to a CSV file."""
    pd.DataFrame(predictions, columns=['PredictedPrice']).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    # Paths to the saved preprocessor and model
    preprocessor_path = 'saved_preprocessors/data_preprocessor.pkl'
    model_path = 'saved_models/your_model_name.pkl'
    new_data_path = 'data/new_dataset.csv'  # Path to the new data

    # Load the preprocessor and model
    preprocessor = load_preprocessor(preprocessor_path)
    model = load_model(model_path)

    # Preprocess the new data and make predictions
    preprocessed_new_data = preprocess_data(new_data_path, preprocessor)
    predictions = predict(model, preprocessed_new_data)

    # Save predictions
    save_predictions(predictions)
