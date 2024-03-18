from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

# 1. Hello, data!
df = pd.read_csv('data/properties.csv')

# How big are ya?
print("Number of observations:", df.shape[0])
print("Number of features:", df.shape[1])

# 2. Let's clean you up before splitting into X and y

# Drop duplicate rows
df.drop_duplicates(inplace=True)
print("Dropped duplicates")

# Drop rows with missing target
df.dropna(axis=0, subset=['price'], inplace=True)
print("Dropped rows with missing target")

# Drop columns with **high** percentage of missing values (highly subjective: here >50)
missing_values_count = df.isnull().sum()
percent_missing = (missing_values_count / df.shape[0]) * 100
columns_to_drop = percent_missing[percent_missing > 50].index
print(f"Dropped columns: {list(columns_to_drop)}")

# Drop columns that are **unequivocally** not useful (ie: "ID")
df.drop(columns="id", inplace=True)
print("Dropped columns: id")

# 3. Let's split you up into features ("X") and target ("y")
X = df.drop('price', axis=1)
y = df['price']
print("X shape:", X.shape)
print("y shape:", y.shape)

# 4. Let's split your features ("X") into numerical & categorical
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns
print("Numerical features main X dataset:", num_cols)
print("Categorical features main X dataset:", cat_cols)

# 5. Let's split both "X" and "y" into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Let's start preprocessing on the training set: ENCODING categorical features into numerical

# Define numerical and categorical variables in training set only
num_cols_train = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols_train = X_train.select_dtypes(include=['object', 'category']).columns

# What are we working with?
for col in cat_cols_train:
    print(f"{col}: {df[col].unique()}")

# 6.1 Encoding ordinal features (the ones with inherent hierarchy)

# Map the 'epc' column to numbers
epc_mapping = {
    'MISSING': 0,
    'G': 1,
    'F': 2,
    'E': 3,
    'D': 4,
    'C': 5,
    'B': 6,
    'A': 7,
    'A+': 8,
    'A++': 9
}
X_train['epc'] = X_train['epc'].map(epc_mapping) # so this **replaces** 'epc'with the numerical column
# Verify the 'epc' column transformation
#print(X['epc'].head())

# binning 'state_building' into 3 categories
X_train['state_building_grouped'] = X_train['state_building'].replace({
    'AS_NEW': 'LIKE_NEW',
    'JUST_RENOVATED': 'LIKE_NEW',
    'TO_RESTORE': 'NEEDS_WORK',
    'TO_BE_DONE_UP': 'NEEDS_WORK',
    'TO_RENOVATE': 'NEEDS_WORK'
})

# mapping the categories to numbers
state_mapping = {
    'MISSING': 0,
    'NEEDS_WORK': 1,
    'GOOD': 2,
    'LIKE_NEW': 3
}

# applying the mapping to the new column
X_train['state_building_encoded'] = X_train['state_building_grouped'].map(state_mapping)

# Check the transformation
print(X_train[['state_building_grouped', 'state_building_encoded']].head())

# drop the original 'state_building' column and the temp grouping column
X_train.drop(['state_building', 'state_building_grouped'], axis=1, inplace=True)

# dataset should now only containthe encoded numerical column for 'state_building'
#print(X_train.head())

# Re-define num & cat variables, after dropping
num_cols_train = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols_train = X_train.select_dtypes(include=['object', 'category']).columns

# 6.2 Encoding nominal features in bulk (the ones without inherent hierarchy)
# Print before encoding
print("Before encoding:\n", X_train.head())
print("Number of columns before encoding:", X_train.shape[1])

# Apply encoding directly on X
X_train = pd.get_dummies(X_train, columns=cat_cols_train, drop_first=True)

# Print after encoding
print("\nAfter encoding:", X_train.head())
print("Number of columns after encoding:", X_train.shape[1])

# 7. More preprocessing on the training set: IMPUTING missing values





# Train the linear regression model using the preprocessed data

# # Initialize and train the linear regression model
# trained_model = LinearRegression()
# trained_model.fit(X_train_prepared, y_train)

# # Save the trained model
# joblib.dump(trained_model, 'model.joblib')

# # 7. Testing the model
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Predict on the testing set
# y_pred = trained_model.predict(X_test)

# # Calculate and print the metrics
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"R-squared (R²): {r2}")
