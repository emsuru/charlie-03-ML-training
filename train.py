from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

# 1. Load data
df = pd.read_csv('data/properties.csv')

# 2. Split the data set into features ("X") and target ("y")
X = df.drop('price', axis=1)
y = df['price']

# 3. Define the numerical_cols and categorical_cols variables from X
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessing 1: Imputing missing values

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])

X_test[num_cols] = num_imputer.transform(X_test[num_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

# 5. Preprocessing 2: Encoding categorical data

# OneHotEncoder for categorical data
encoder = OneHotEncoder(handle_unknown='ignore')

# Fit and transform the training data, and transform the testing data
X_train_encoded = encoder.fit_transform(X_train[cat_cols])
X_test_encoded = encoder.transform(X_test[cat_cols])

# Convert encoded data back to DataFrame
# Convert encoded data back to DataFrame
X_train_encoded = pd.DataFrame(X_train_encoded.toarray(), columns=encoder.get_feature_names_out(cat_cols))
X_test_encoded = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names_out(cat_cols))

# Concatenate encoded categorical data with the rest of the data
X_train = pd.concat([X_train.drop(cat_cols, axis=1), X_train_encoded], axis=1)
X_test = pd.concat([X_test.drop(cat_cols, axis=1), X_test_encoded], axis=1)

# 6. Train the linear regression model using the preprocessed data

# Initialize and train the linear regression model
trained_model = LinearRegression()
trained_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(trained_model, 'model.joblib')

# 7. Testing the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict on the testing set
y_pred = trained_model.predict(X_test)

# Calculate and print the metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
