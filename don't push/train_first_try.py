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

# DROP duplicate rows
df.drop_duplicates(inplace=True)
print("Dropped duplicates")

# DROP rows with missing target
df.dropna(axis=0, subset=['price'], inplace=True)
print("Dropped rows with missing target")

# DROP columns with **high** percentage of missing values (highly subjective: here >50)
missing_values_count = df.isnull().sum()
percent_missing = (missing_values_count / df.shape[0]) * 100
columns_to_drop = percent_missing[percent_missing > 50].index
df.drop(columns=columns_to_drop, inplace=True)
print(f"Dropped columns: {list(columns_to_drop)}")

# DROP columns that are **unequivocally** not useful (ie: "ID")
df.drop(columns="id", inplace=True)
print("Dropped columns: id")


# IMPUTE missing values for CATEGORICAL COLS with "MISSING".
# --I've learned this can be done on the whole dataset,
# --since operation does not involve learning any parameters from the data

cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    df[f"{col}_was_missing"] = df[col].isnull()
    df[col].fillna("MISSING", inplace=True)

# ENCODE Categorical Cols, with BINNING - this also isn't considered "learning" from the data
# ENCODE ORDINAL features (the ones with inherent hierarchy): "epc" and "state_building"
for col in cat_cols:
    print(f"Categorical column '{col}' has unique values: {df[col].unique()}")
    print(f"Number of unique values: {df[col].nunique()}")

# Print before transformation
print(df[['state_building', 'epc']].head())

df['state_building_grouped'] = df['state_building'].replace({
    'AS_NEW': 'LIKE_NEW',
    'JUST_RENOVATED': 'LIKE_NEW',
    'TO_RESTORE': 'NEEDS_WORK',
    'TO_BE_DONE_UP': 'NEEDS_WORK',
    'TO_RENOVATE': 'NEEDS_WORK'
})

# Mapping the categories to numbers
state_mapping = {
    'MISSING': 0,
    'NEEDS_WORK': 1,
    'GOOD': 2,
    'LIKE_NEW': 3
}

# Applying the mapping to the new column
df['state_building_encoded'] = df['state_building_grouped'].map(state_mapping)

# DROP the original 'state_building' column and the temp grouping column
df.drop(['state_building', 'state_building_grouped'], axis=1, inplace=True)

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
df['epc'] = df['epc'].map(epc_mapping) #so this **replaces** 'epc'with the numerical column
print(df[['state_building_encoded', 'epc']].head())

print(f"Number of numerical features: {df.select_dtypes(include=['number']).shape[1]}")
print(f"Number of categorical features: {df.select_dtypes(include=['object', 'category']).shape[1]}")


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

# 6. PREPROCESSING. Let's start preprocessing on the training set:

# DEFINE numerical and categorical variables in training set only
num_cols_train = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols_train = X_train.select_dtypes(include=['object', 'category']).columns

# ENCODE NOMINAL features in bulk (the ones without inherent hierarchy)

# Print before encoding
print("Before encoding:\n", X_train.head())
print("Number of columns before encoding:", X_train.shape[1])

# Apply encoding directly on X_train
X_train_encoded = pd.get_dummies(X_train, columns=cat_cols_train, drop_first=True)

# Apply the same encoding to X_test
X_test_encoded = pd.get_dummies(X_test, columns=cat_cols_train, drop_first=True)

# Align X_train_encoded and X_test_encoded to ensure they have the same columns
X_train_aligned, X_test_aligned = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# Print after encoding and aligning
print("\nAfter encoding and aligning X_train:", X_train_aligned.head())
print("Number of columns in X_train after encoding and aligning:", X_train_aligned.shape[1])
print("\nAfter encoding and aligning X_test:", X_test_aligned.head())
print("Number of columns in X_test after encoding and aligning:", X_test_aligned.shape[1])

#FEATURE SELECTION - DROP columns with no correlation to target (subjective)

# Calculate correlation matrix on the numeric columns of the training set
numeric_cols_train = X_train.select_dtypes(include=['number'])
correlation_matrix_train = numeric_cols_train.join(y_train).corr()  # Joining y_train to include the target in correlation calculation
correlations_with_target_train = correlation_matrix_train['price'].abs().sort_values(ascending=False)

# Define a threshold for dropping columns based on correlation with the target variable
threshold = 0.14  # This is subjective and can be adjusted based on domain knowledge and specific requirements

# Identify columns that have low correlation with the target in the training set
columns_to_drop_due_to_low_correlation = correlations_with_target_train[correlations_with_target_train < threshold].index.tolist()

# Apply the feature selection to both training and test sets
X_train = X_train.drop(columns=columns_to_drop_due_to_low_correlation)
X_test = X_test.drop(columns=columns_to_drop_due_to_low_correlation)

print(f"Dropped columns due to low correlation with target: {columns_to_drop_due_to_low_correlation}")

# Update numeric_cols_train after dropping columns
numeric_cols_train = X_train.select_dtypes(include=['int64', 'float64']).columns



# # 6.3 IMPUTE NUMERICAL - Impute all numerical features

# # Identify missing values
# Identify missing values in numerical columns and print the min and max values
for col in numeric_cols_train:
    if X_train[col].isnull().any():
        print(f"{col}: Min = {X_train[col].min()}, Max = {X_train[col].max()}")


# IMPUTING #1: Statistical bulk imputing with median, because fuck it
# Initialize the imputer
num_imputer = SimpleImputer(strategy='median')

# Fit the imputer on the training data and transform it
X_train[numeric_cols_train] = num_imputer.fit_transform(X_train[numeric_cols_train])

# Transform the test data using the same imputer
X_test[numeric_cols_train] = num_imputer.transform(X_test[numeric_cols_train])

# Check if any missing values remain in the training set
print("Missing values in numerical columns of training set after imputation:\n", X_train[numeric_cols_train].isnull().sum())

# Optionally, check for missing values in the test set as well
print("Missing values in numerical columns of test set after imputation:\n", X_test[numeric_cols_train].isnull().sum())

# # IMPUTING #2: Arbitrary imputing with custom values
# # values chosen based on the feature's distribution
# arbitrary_values = {
#     'latitude': 999999999,
#     'longitude': 999999999,
#     'construction_year': 999999999,
#     'total_area_sqm': 999999999,
#     'surface_land_sqm': 999999999,
#     'nbr_frontages': 999999999 ,
#     'terrace_sqm': 999999999,
#     'garden_sqm': 999999999,
#     'primary_energy_consumption_sqm': 999999999
# }
# # Impute each feature with its corresponding arbitrary value
# for feature, value in arbitrary_values.items():
#     imputer = SimpleImputer(strategy='constant', fill_value=value)
#     X_train[feature] = imputer.fit_transform(X_train[[feature]])
# # Verify the imputation
# print(X_train.head())

# # IMPUTING #3  - Some other fancier imputing, KNN or geocoders


# # ---- MODEL TRAINING CODE BELOW ----

# Train the linear regression model using the preprocessed data

# # Initialize and train the linear regression model
trained_model = LinearRegression()
trained_model.fit(X_train, y_train)

# # Save the trained model
joblib.dump(trained_model, 'model.joblib')

# # 7. Testing the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Predict on the testing set
y_pred = trained_model.predict(X_test)

# # Calculate and print the metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
