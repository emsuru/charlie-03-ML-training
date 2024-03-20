#

# Model Card: Immo Charlie Price Predictor

## Model Details
- **Model Type**: Linear Regression, with additional models including CatBoost, XGBoost, Random Forest, etc.
- **Training Data**: `data/properties.csv`
- **Preprocessing**: Data cleaning, imputation, and encoding as per `preprocessing/data_preprocessor.py`.
- **Evaluation Metrics**: Model performance evaluated using standard metrics (MAE, MSE, RMSE, R²).

## Intended Use
- **Primary Use**: Predicting residential property prices in Belgium, based on various features.
- **Potential Users**: Real estate agencies, property investors, data scientists analyzing the real estate market.

## Training Procedure
- **Preprocessing Steps**: Refer to `preprocessing/data_preprocessor.py` for detailed steps.
- **Training Environment**: The model was developed & trained on a MacBook, in a Python-based environment. See all libraries used in `requirements.txt`.

## Evaluation
- **Performance Metrics**: RMSE, MAE, R², per each model trained. Best performing model will be selected for deployment.
- **Validation Methods**: To be filled in...

## Caveats and Recommendations
- **Model Limitations**: To be filled-in..
- **Future Work**: Hyperparameter tuning and API deployment in a future stage of the project.
