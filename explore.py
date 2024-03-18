import pandas as pd

# 1. Hello, data!
df = pd.read_csv('data/properties.csv')

# What do you look like?
print("Number of observations:", df.shape[0])
print("Number of features:", df.shape[1])

feature_types = df.dtypes
numerical_features = feature_types[feature_types != 'object']
categorical_features = feature_types[feature_types == 'object']
print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

# Do you have any duplicates?
duplicates = df.duplicated().sum()
print("Duplicates:", duplicates)

# Do you have any missing values?
missing_target = df['price'].isnull().sum()
print("Rows with missing target value:", missing_target)

missing_values_count = df.isnull().sum()
missing_cols = missing_values_count[missing_values_count > 0]
print(f"Columns with missing values: {len(missing_cols)} out of {df.shape[1]}")

percent_missing = (missing_cols / df.shape[0]) * 100
percent_missing_sorted = percent_missing.sort_values(ascending=False)

print("Index", "| Feature |", "| Missing values |", "| Type |", sep="\t")
for index, (feature, percent) in enumerate(percent_missing_sorted.items(), start=1):
    feature_type = "Categorical" if feature_types[feature] == 'object' else "Numerical"
    print(index, feature, f"{percent:.2f}%", feature_type, sep="\t")

# 2. Data viz (exploring distributions, correlations, etc.)

# 3. Let's clean you up before splitting into X and y

# # Drop duplicate rows

duplicates = df.duplicated().sum()
if duplicates > 0:
    shape_before = df.shape
    df = df.drop_duplicates(inplace=True)
    shape_after = df.shape
    print(f"{duplicates} duplicates removed. Shape before: {shape_before}, Shape after: {shape_after}")
else:
    print("No duplicates found.")

# Drop rows with missing target
df.dropna(axis=0, subset=['price'], inplace=True)
print("Dropped rows with missing target")

# Drop columns with high percentage of missing values (highly subjective: here >50)
columns_to_drop = percent_missing[percent_missing > 50].index
df.drop(columns=columns_to_drop, inplace=True)
print(f"Dropped columns: {list(columns_to_drop)}")

# Drop columns that are not useful for the model (ie: "ID")
df.drop(columns="id", inplace=True)
print("Dropped columns: id")

# 4. Encoding categorical data (in train.py, this would be done on test set only)
