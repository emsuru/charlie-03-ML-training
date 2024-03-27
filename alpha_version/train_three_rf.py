# --- TRAINING SCRIPT AND PREDICT SCRIPT IN ONE  ---
# First attempt to understand the whole process of training a model and making predictions, in one script.
# I'm using Random Forest Regressor here, as it's the best performing model so far, but I'm using it with a different, much simplified preproecssing pipeline
# than I do in the main project. This is just to get a feel of how the whole process works.

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import xgboost
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


## ---- 1. Hello, data! Who are you? ----
def hello_data(df):
    print("Number of observations:", df.shape[0])
    print("Number of features:", df.shape[1])
    return df


## ---- 2. Let's clean you up before splitting into X and y ----
def clean_drop(df):
    # DROP duplicate rows that have the same value in the 'id' column
    df.drop_duplicates(subset='id', inplace=True)
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

    # DROP columns that are **unequivocally** not useful (leaving the others commented out for transparency)
    df = df[[#'id',
         'price', 'property_type',
         #'subproperty_type',
         'region', 'province',
         # 'locality',
         #'zip_code', 'latitude', 'longitude',
       # 'construction_year',
       'total_area_sqm',
       # 'surface_land_sqm',
       'nbr_frontages', 'nbr_bedrooms',
       #'equipped_kitchen', 'fl_furnished',
       #'fl_open_fire', 'fl_terrace', 'terrace_sqm', 'fl_garden', 'garden_sqm',
       #'fl_swimming_pool', 'fl_floodzone', 'state_building',
       #'primary_energy_consumption_sqm', 'epc', 'heating_type',
       #'fl_double_glazing', 'cadastral_income'
       ]].copy()
    print("Dropped columns")
    return df

def clean_impute(df):
    # IMPUTE missing values for CATEGORICAL COLS with "MISSING".
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[f"{col}_was_missing"] = df[col].isnull()
        df[col].fillna("MISSING", inplace=True)
    return df

# def clean_encode(df): # commenting this method out for now (not needed, as I've decided to drop many columns, for a simplified version of the model,
# mostly interested here in makin the predictpart of the script work)

#     # ENCODE Categorical Cols, with BINNING - this also isn't considered "learning" from the data
#     # ENCODE ORDINAL features (the ones with inherent hierarchy): "epc" and "state_building"
#     cat_cols = df.select_dtypes(include=['object', 'category']).columns
#     for col in cat_cols:
#         print(f"Categorical column '{col}' has unique values: {df[col].unique()}")
#         print(f"Number of unique values: {df[col].nunique()}")

#     # Print before transformation
#     print(df[['state_building', 'epc']].head())

#     df['state_building_grouped'] = df['state_building'].replace({
#         'AS_NEW': 'LIKE_NEW',
#         'JUST_RENOVATED': 'LIKE_NEW',
#         'TO_RESTORE': 'NEEDS_WORK',
#         'TO_BE_DONE_UP': 'NEEDS_WORK',
#         'TO_RENOVATE': 'NEEDS_WORK'
#     })

#     # Mapping the categories to numbers
#     state_mapping = {
#         'MISSING': 0,
#         'NEEDS_WORK': 1,
#         'GOOD': 2,
#         'LIKE_NEW': 3
#     }

#     # Applying the mapping to the new column
#     df['state_building_encoded'] = df['state_building_grouped'].map(state_mapping)

#     # DROP the original 'state_building' column and the temp grouping column
#     df.drop(['state_building', 'state_building_grouped'], axis=1, inplace=True)

#     epc_mapping = {
#         'MISSING': 0,
#         'G': 1,
#         'F': 2,
#         'E': 3,
#         'D': 4,
#         'C': 5,
#         'B': 6,
#         'A': 7,
#         'A+': 8,
#         'A++': 9
#     }
#     df['epc'] = df['epc'].map(epc_mapping) #so this **replaces** 'epc'with the numerical column
#     print(df[['state_building_encoded', 'epc']].head())

#     print(f"Number of numerical features: {df.select_dtypes(include=['number']).shape[1]}")
#     print(f"Number of categorical features: {df.select_dtypes(include=['object', 'category']).shape[1]}")
#     return df

## ---- 3. Let's split you up and preprocess you ----
def preprocess_split(df, target='price', test_size=0.2, random_state=42):
    X = df.drop(target, axis=1)
    y = df[target]
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# def preprocess_encode(X_train, X_test):
#     cat_cols_train = X_train.select_dtypes(include=['object', 'category']).columns
#     print("Before encoding:\n", X_train.head())
#     print("Number of columns before encoding:", X_train.shape[1])
#     X_train_encoded = pd.get_dummies(X_train, columns=cat_cols_train, drop_first=True)
#     X_test_encoded = pd.get_dummies(X_test, columns=cat_cols_train, drop_first=True)
#     X_train_aligned, X_test_aligned = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
#     print("\nAfter encoding and aligning X_train:", X_train_aligned.head())
#     print("Number of columns in X_train after encoding and aligning:", X_train_aligned.shape[1])
#     print("\nAfter encoding and aligning X_test:", X_test_aligned.head())
#     print("Number of columns in X_test after encoding and aligning:", X_test_aligned.shape[1])
#     return X_train_aligned, X_test_aligned

def preprocess_encode(X_train, X_test):
    cat_cols_train = X_train.select_dtypes(include=['object', 'category']).columns
    print("Before encoding:\n", X_train.head())
    print("Number of columns before encoding:", X_train.shape[1])

    # Initialize OneHotEncoder
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore')

    # Fit and transform the training data
    X_train_encoded = ohe.fit_transform(X_train[cat_cols_train])
    X_train_encoded_dense = X_train_encoded.toarray()   # Convert the sparse matrix to a dense format (I got errors when passing the 'sparse' argument when initialisign the OHE... so used this instead)

    # Save the OneHotEncoder object to disk
    joblib.dump(ohe, "alpha_version/ohe_encoder_t3.pkl")
    print("OneHotEncoder saved to 'alpha_version/ohe_encoder_t3.pkl'")

    # Now create the DataFrame for train set with the dense matrix
    X_train_encoded_df = pd.DataFrame(X_train_encoded_dense, columns=ohe.get_feature_names_out(cat_cols_train), index=X_train.index)

    # Transform the test data
    X_test_encoded = ohe.transform(X_test[cat_cols_train])
    X_test_encoded_dense = X_test_encoded.toarray()  # Convert the sparse matrix to a dense format (I got errors when passing the 'sparse' argument when initialisign the OHE... so used this instead)
    # Now create the DataFrame for test set with the dense matrix
    X_test_encoded_df = pd.DataFrame(X_test_encoded_dense, columns=ohe.get_feature_names_out(cat_cols_train), index=X_test.index)

    # Drop original categorical columns and concatenate the encoded ones
    X_train_aligned = X_train.drop(columns=cat_cols_train).join(X_train_encoded_df)
    X_test_aligned = X_test.drop(columns=cat_cols_train).join(X_test_encoded_df)

    print("\nAfter encoding X_train:", X_train_aligned.head())
    print("Number of columns in X_train after encoding:", X_train_aligned.shape[1])
    print("\nAfter encoding X_test:", X_test_aligned.head())
    print("Number of columns in X_test after encoding:", X_test_aligned.shape[1])

    return X_train_aligned, X_test_aligned



def preprocess_feat_select(X_train_aligned, X_test_aligned, y_train, threshold=0.14):
    numeric_cols_train = X_train_aligned.select_dtypes(include=['number'])
    correlation_matrix_train = numeric_cols_train.join(y_train).corr()
    correlations_with_target_train = correlation_matrix_train['price'].abs().sort_values(ascending=False)
    columns_to_drop_due_to_low_correlation = correlations_with_target_train[correlations_with_target_train < threshold].index.tolist()

    # Drop the same columns from both X_train_aligned and X_test_aligned
    X_train_aligned = X_train_aligned.drop(columns=columns_to_drop_due_to_low_correlation)
    X_test_aligned = X_test_aligned.drop(columns=columns_to_drop_due_to_low_correlation)

    print(f"Dropped columns due to low correlation with target: {columns_to_drop_due_to_low_correlation}")

    columns_to_keep = X_train_aligned.columns.tolist()
    joblib.dump(columns_to_keep, "alpha_version/columns_to_keep_t3.pkl")

    return X_train_aligned, X_test_aligned

def preprocess_impute(X_train_aligned, X_test_aligned, strategy='median'):
    numeric_cols_train = X_train_aligned.select_dtypes(include=['int64', 'float64']).columns
    num_imputer = SimpleImputer(strategy=strategy)
    X_train_aligned[numeric_cols_train] = num_imputer.fit_transform(X_train_aligned[numeric_cols_train])
    X_test_aligned[numeric_cols_train] = num_imputer.transform(X_test_aligned[numeric_cols_train])
    print("Missing values in numerical columns of training set after imputation:\n", X_train_aligned[numeric_cols_train].isnull().sum())
    print("Missing values in numerical columns of test set after imputation:\n", X_test_aligned[numeric_cols_train].isnull().sum())
    # Save the num_imputer object to disk
    joblib.dump(num_imputer, "alpha_version/num_imputer_t3.pkl")
    print("Imputer saved to 'alpha_version/num_imputer_t3.pkl'")

    return X_train_aligned, X_test_aligned


# Start the pipeline

df = pd.read_csv('input_data/properties.csv')
df = hello_data(df)

df = clean_drop(df)
df = clean_impute(df)
# df = clean_encode(df)
X_train, X_test, y_train, y_test = preprocess_split(df)
X_train_aligned, X_test_aligned = preprocess_encode(X_train, X_test)
X_train_aligned, X_test_aligned = preprocess_feat_select(X_train_aligned, X_test_aligned, y_train)
X_train_aligned, X_test_aligned = preprocess_impute(X_train_aligned, X_test_aligned)

# 6. Train the Model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # 7. Initialize the Model
# model = LinearRegression()
# model = XGBRegressor()
model = RandomForestRegressor()

# 8. Train the Model
model.fit(X_train_aligned, y_train)

# 9. Make Predictions
y_train_pred = model.predict(X_train_aligned)
y_test_pred = model.predict(X_test_aligned)

# 10. Evaluate the Model
# Training set evaluation
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

print("Training Set Evaluation RF t3:")
print(f"Mean Absolute Error: {train_mae}")
print(f"Mean Squared Error: {train_mse}")
print(f"Root Mean Squared Error: {train_rmse}")  # Display RMSE
print(f"R-squared: {train_r2}")

# Test set evaluation
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\nTest Set Evaluation RF t3:")
print(f"Mean Absolute Error: {test_mae}")
print(f"Mean Squared Error: {test_mse}")
print(f"Root Mean Squared Error: {test_rmse}")
print(f"R-squared: {test_r2}")

# 11. ---- SAVE MODEL ----

# Save the model to disk in the 'saved_models' directory
joblib_file = "alpha_version/randomforest_t3.pkl"
joblib.dump(model, joblib_file)

print(f"Model saved to {joblib_file}")


# # 12. define PREDICTION function ---

def predict_new_data(new_data_path):
    # Load the saved model and preprocessing objects
    model = joblib.load("alpha_version/randomforest_t3.pkl")
    num_imputer = joblib.load("alpha_version/num_imputer_t3.pkl")
    ohe_encoder = joblib.load("alpha_version/ohe_encoder_t3.pkl")
    model_columns = joblib.load("alpha_version/columns_to_keep_t3.pkl")

    # Load new data
    new_data = pd.read_csv(new_data_path)

    # Preprocess new data (these steps don't have any fitted objects)

    new_data = clean_drop(new_data)  # do I need to call this on newdata or not?!!
    new_data = clean_impute(new_data)  # do I need to call this on newdata or not?!!

    # Apply OneHotEncoder to the categorical columns
    cat_cols = new_data.select_dtypes(include=['object', 'category']).columns
    new_data_encoded = ohe_encoder.transform(new_data[cat_cols])
    new_data_encoded_df = pd.DataFrame(new_data_encoded.toarray(), columns=ohe_encoder.get_feature_names_out(cat_cols), index=new_data.index)

    # Drop original categorical columns and concatenate the encoded ones
    new_data = new_data.drop(columns=cat_cols).join(new_data_encoded_df)

    # Ensure the new data has the same columns as the model was trained on, filling missing columns with zeros
    new_data = new_data.reindex(columns=model_columns, fill_value=0)

    # Impute missing values in numerical columns
    numeric_cols = new_data.select_dtypes(include=['int64', 'float64']).columns
    new_data[numeric_cols] = num_imputer.transform(new_data[numeric_cols])

    # Predict
    predictions = model.predict(new_data)
    return predictions

# 13. Test the prediction function
predictions = predict_new_data("input_data/newdata_25.csv")


# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
# Save the predictions to a CSV file
predictions_df.to_csv("alpha_version/predictions_25.csv", index=False)

print("Predictions saved to alpha_version/predictions_25.csv")
