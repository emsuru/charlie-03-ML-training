## --- CURRENTLY NOT WORKING AT ALL --

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

## --- STEP 1: Define Preprocessing Functions ---

# Custom transformer for dropping duplicates and rows with missing target values
class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # No need for target_column anymore

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop_duplicates()
        return X

# Custom transformer for dropping columns based on missing value threshold and specific columns
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=50, columns_to_drop=None):
        self.threshold = threshold
        self.columns_to_drop = columns_to_drop if columns_to_drop else []

    def fit(self, X, y=None):
        missing_values_count = X.isnull().sum()
        percent_missing = (missing_values_count / X.shape[0]) * 100
        self.columns_to_drop += percent_missing[percent_missing > self.threshold].index.tolist()
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)

# Custom transformer for encoding categorical columns
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.cat_cols:
            for col in self.cat_cols:
                X[col].fillna("MISSING", inplace=True)
            X = pd.get_dummies(X, columns=self.cat_cols, drop_first=True)
        return X

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(num_cols, cat_cols):
    num_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    cat_preprocessor = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_preprocessor, num_cols),
            ('cat', cat_preprocessor, cat_cols)])

    return preprocessor

from sklearn.linear_model import LinearRegression

def build_pipeline(num_cols, cat_cols):
    pipeline = Pipeline(steps=[
        ('cleaner', DataCleaner()),
        ('dropper', ColumnDropper()),
        ('preprocessor', build_preprocessor(num_cols, cat_cols)),
        ('regressor', LinearRegression())
    ])
    return pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your data
df = pd.read_csv('data/properties.csv')

# Split data into features and target
X = df.drop('price', axis=1)
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

# 'category']).columns

# Build the pipeline
pipeline = build_pipeline(num_cols=num_cols, cat_cols=cat_cols)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Evaluate the model
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
test_r2 = r2_score(y_test, y_test_pred)

print("Training Set Evaluation:")
print(f"Mean Absolute Error: {train_mae}")
print(f"Mean Squared Error: {train_mse}")
print(f"Root Mean Squared Error: {train_rmse}")
print(f"R-squared: {train_r2}")

print("\nTest Set Evaluation:")
print(f"Mean Absolute Error: {test_mae}")
print(f"Mean Squared Error: {test_mse}")
print(f"Root Mean Squared Error: {test_rmse}")
print(f"R-squared: {test_r2}")
