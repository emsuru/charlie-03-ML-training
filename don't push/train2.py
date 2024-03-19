from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pandas as pd

#-- STEP 1: Load data, degine categorical & numerical list variables--
#
# # Load our data
df = pd.read_csv('data/properties.csv')

# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

#-- STEP 2: Define preprocessing steps --

# # Define preprocessing for numerical columns (impute then scale)
num_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Define preprocessing for categorical columns (impute then one-hot encode)
cat_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

#-- STEP 3: Combine Preprocessors with ColumnTransformer
# Use ColumnTransformer to apply the respective preprocessing steps to the numerical and categorical columns.

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_preprocessor, numerical_cols),
        ('cat', cat_preprocessor, categorical_cols)])

#-- STEP 4: Create a pipeline that combines the preprocessor with a linear regression model
from sklearn.linear_model import LinearRegression

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', LinearRegression())])

#-- STEP 5: Split the data
from sklearn.model_selection import train_test_split

X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#-- STEP 6: Train the pipeline on the training data
model_pipeline.fit(X_train, y_train)

#-- STEP 7: Save the trained pipeline to a file
import joblib

joblib.dump(model_pipeline, 'model_pipeline.joblib')

### -----

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 9. Make Predictions
y_train_pred = model_pipeline.predict(X_train)
y_test_pred = model_pipeline.predict(X_test)

# 10. Evaluate the Model
# Training set evaluation
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)

print("Training Set Evaluation:")
print(f"Mean Absolute Error: {train_mae}")
print(f"Mean Squared Error: {train_mse}")
print(f"Root Mean Squared Error: {train_rmse}")  # Display RMSE
print(f"R-squared: {train_r2}")

# Test set evaluation
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)

print("\nTest Set Evaluation:")
print(f"Mean Absolute Error: {test_mae}")
print(f"Mean Squared Error: {test_mse}")
print(f"Root Mean Squared Error: {test_rmse}")
print(f"R-squared: {test_r2}")
