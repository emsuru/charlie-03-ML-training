import pandas as pd


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

# DROP columns with no correlation to target (subjective)

# -- Ensure df contains only numeric columns before computing correlation matrix
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr() # Compute correlation matrix
correlations_with_target = correlation_matrix['price'].abs().sort_values(ascending=False)
print("Correlation with target (descending order):\n", correlations_with_target)
threshold = 0.14 #highly subjective

# -- Identify columns that have low correlation with the target
columns_to_drop_due_to_low_correlation = correlations_with_target[correlations_with_target < threshold].index.tolist()

df.drop(columns=columns_to_drop_due_to_low_correlation, inplace=True)

print(f"Dropped columns due to low correlation with target: {columns_to_drop_due_to_low_correlation}")

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
