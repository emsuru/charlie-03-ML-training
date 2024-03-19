import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def clean_drop(self):
        self.df.drop_duplicates(inplace=True) # DROP duplicate rows
        print("Dropped duplicates")

        self.dropna(axis=0, subset=['price'], inplace=True)   # DROP rows with missing target
        print("Dropped rows with missing target")

        # DROP columns with **high** percentage of missing values (highly subjective: here >50)
        missing_values_count = self.df.isnull().sum()
        percent_missing = (missing_values_count / self.df.shape[0]) * 100
        columns_to_drop = percent_missing[percent_missing > 50].index
        self.df.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns: {list(columns_to_drop)}")

        # DROP columns that are **unequivocally** not useful (ie: "ID")
        self.df.drop(columns="id", inplace=True)
        print("Dropped columns: id")
        return self

    def clean_impute(self):  # IMPUTE missing values for CATEGORICAL COLS with "MISSING".
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            self.df[f"{col}_was_missing"] = self.df[col].isnull()
            self.df[col].fillna("MISSING", inplace=True)
        return self

    def clean_encode(self):
        # ENCODE Categorical Cols, with BINNING - this also isn't considered "learning" from the data
        # ENCODE ORDINAL features (the ones with inherent hierarchy): "epc" and "state_building"
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            print(f"Categorical column '{col}' has unique values: {self.df[col].unique()}")
            print(f"Number of unique values: {self.df[col].nunique()}")
        # Print before transformation
        print(self.df[['state_building', 'epc']].head())

        self.df['state_building_grouped'] = self.df['state_building'].replace({
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
        self.df['state_building_encoded'] = self.df['state_building_grouped'].map(state_mapping)

        # DROP the original 'state_building' column and the temp grouping column
        self.df.drop(['state_building', 'state_building_grouped'], axis=1, inplace=True)

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

        self.df['epc'] = self.df['epc'].map(epc_mapping) #so this **replaces** 'epc'with the numerical column

        print(self.df[['state_building_encoded', 'epc']].head())

        print(f"Number of numerical features: {self.df.select_dtypes(include=['number']).shape[1]}")
        print(f"Number of categorical features: {self.df.select_dtypes(include=['object', 'category']).shape[1]}")
        return self

    def preprocess_split(self, target='price', test_size=0.2, random_state=42):
        X = self.df.drop(target, axis=1)
        y = self.df[target]
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
