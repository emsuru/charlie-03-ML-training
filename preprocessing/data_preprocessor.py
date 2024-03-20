import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def clean_drop(self):
        self.df.drop_duplicates(inplace=True) # DROP duplicate rows
        print("Dropped duplicates")

        self.df.dropna(axis=0, subset=['price'], inplace=True)   # DROP rows with missing target
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

    def save_training_columns(self, filepath='training/training_columns.txt'):
        """
        Saves the column names of the processed dataframe to a file, excluding the target variable.
        """
        # Exclude the target variable 'price' from the columns
        training_columns = [col for col in self.df.columns if col != 'price']
        # Save the column names to a file
        with open(filepath, 'w') as file:
            for column in training_columns:
                file.write(f"{column}\n")
        print(f"Training columns saved to {filepath}")

    def preprocess_split(self, target='price', test_size=0.2, random_state=42):
        X = self.df.drop(target, axis=1)
        y = self.df[target]
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test


    def preprocess_encode(self, X_train, X_test):
        cat_cols_train = X_train.select_dtypes(include=['object', 'category']).columns
        print("Before encoding:\n", X_train.head())
        print("Number of columns before encoding:", X_train.shape[1])
        X_train_encoded = pd.get_dummies(X_train, columns=cat_cols_train, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test, columns=cat_cols_train, drop_first=True)
        X_train_aligned, X_test_aligned = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
        print("\nAfter encoding and aligning X_train:", X_train_aligned.head())
        print("Number of columns in X_train after encoding and aligning:", X_train_aligned.shape[1])
        print("\nAfter encoding and aligning X_test:", X_test_aligned.head())
        print("Number of columns in X_test after encoding and aligning:", X_test_aligned.shape[1])
        return X_train_aligned, X_test_aligned


    def preprocess_feat_select(self, X_train_aligned, X_test_aligned, y_train, threshold=0.14):
        numeric_cols_train = X_train_aligned.select_dtypes(include=['number'])
        correlation_matrix_train = numeric_cols_train.join(y_train).corr()
        correlations_with_target_train = correlation_matrix_train['price'].abs().sort_values(ascending=False)
        columns_to_drop_due_to_low_correlation = correlations_with_target_train[correlations_with_target_train < threshold].index.tolist()

        # Drop the same columns from both X_train_aligned and X_test_aligned
        X_train_aligned = X_train_aligned.drop(columns=columns_to_drop_due_to_low_correlation)
        X_test_aligned = X_test_aligned.drop(columns=columns_to_drop_due_to_low_correlation)

        print(f"Dropped columns due to low correlation with target: {columns_to_drop_due_to_low_correlation}")
        return X_train_aligned, X_test_aligned

    def preprocess_impute(self, X_train_aligned, X_test_aligned, strategy='median'):
        numeric_cols_train = X_train_aligned.select_dtypes(include=['int64', 'float64']).columns
        num_imputer = SimpleImputer(strategy=strategy)
        X_train_aligned[numeric_cols_train] = num_imputer.fit_transform(X_train_aligned[numeric_cols_train])
        X_test_aligned[numeric_cols_train] = num_imputer.transform(X_test_aligned[numeric_cols_train])
        print("Missing values in numerical columns of training set after imputation:\n", X_train_aligned[numeric_cols_train].isnull().sum())
        print("Missing values in numerical columns of test set after imputation:\n", X_test_aligned[numeric_cols_train].isnull().sum())
        return X_train_aligned, X_test_aligned
