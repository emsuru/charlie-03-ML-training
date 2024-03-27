import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

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


    def encode_state_building(self):
        # Group 'state_building' values
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
        # # Save the state_mapping to a pkl file to call later in predictions
        # joblib.dump(state_mapping, 'preprocessing/state_mapping.pkl')
        # print("Saved state_mapping to 'preprocessing/state_mapping.pkl'")
        return self

    def encode_epc(self):
        # Mapping for 'epc'
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
        # Apply mapping to 'epc'
        self.df['epc'] = self.df['epc'].map(epc_mapping)

        # # Save the epc_mapping to a file to use later in predictions
        # joblib.dump(epc_mapping, 'preprocessing/epc_mapping.pkl')
        # print("Saved epc_mapping to 'preprocessing/epc_mapping.pkl'")
        return self


    def preprocess_split(self, target='price', test_size=0.2, random_state=42):
        X = self.df.drop(target, axis=1)
        y = self.df[target]
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    # encoding with pandas get_dummies() is commented out
    # because I'm using the OneHotEncoder instead (but with get_dummies I get better model scores,
    # so keeping it here for reference.. and in case I want to switch back

    # def preprocess_encode(self, X_train, X_test):
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


    def preprocess_encode(self, X_train, X_test):
        cat_cols_train = X_train.select_dtypes(include=['object', 'category']).columns
        print("Before encoding:\n", X_train.head())
        print("Number of columns before encoding:", X_train.shape[1])

        # Initialize OneHotEncoder
        ohe = OneHotEncoder(drop='first')

        # Fit and transform the training data
        X_train_encoded = ohe.fit_transform(X_train[cat_cols_train])
        X_train_encoded_dense = X_train_encoded.toarray()   # Convert the sparse matrix to a dense format (I got errors when passing the 'sparse' argument when initialisign the OHE... so used this instead)

        # Save the fitted OneHotEncoder model to a file immediately after fitting, to use later for predictions
        joblib.dump(ohe, 'preprocessing/onehotencoder.pkl')
        print("Saved OneHotEncoder model to 'preprocessing/onehotencoder.pkl'")

        # Now create the DataFrame for train set with the dense matrix
        X_train_encoded_df = pd.DataFrame(X_train_encoded_dense, columns=ohe.get_feature_names_out(cat_cols_train), index=X_train.index)

        # Transform the test data
        X_test_encoded = ohe.transform(X_test[cat_cols_train])
        X_test_encoded_dense = X_test_encoded.toarray()  # Convert the sparse matrix to a dense format
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


    def preprocess_feat_select(self, X_train_aligned, X_test_aligned, y_train, threshold=0.14):
        numeric_cols_train = X_train_aligned.select_dtypes(include=['number'])
        correlation_matrix_train = numeric_cols_train.join(y_train).corr()
        correlations_with_target_train = correlation_matrix_train['price'].abs().sort_values(ascending=False)
        columns_to_drop_due_to_low_correlation = correlations_with_target_train[correlations_with_target_train < threshold].index.tolist()

        # Drop the same columns from both X_train_aligned and X_test_aligned
        X_train_aligned = X_train_aligned.drop(columns=columns_to_drop_due_to_low_correlation)
        X_test_aligned = X_test_aligned.drop(columns=columns_to_drop_due_to_low_correlation)

        print(f"Dropped columns due to low correlation with target: {columns_to_drop_due_to_low_correlation}")

        #Save to use later for predictions
        columns_to_keep = X_train_aligned.columns.tolist()
        joblib.dump(columns_to_keep, "preprocessing/columns_to_keep.pkl")
        print("Saved columns to keep to 'preprocessing/columns_to_keep.pkl'")

        return X_train_aligned, X_test_aligned

    def preprocess_impute(self, X_train_aligned, X_test_aligned, strategy='median'):
        numeric_cols_train = X_train_aligned.select_dtypes(include=['int64', 'float64']).columns
        num_imputer = SimpleImputer(strategy=strategy)
        X_train_aligned[numeric_cols_train] = num_imputer.fit_transform(X_train_aligned[numeric_cols_train])
        X_test_aligned[numeric_cols_train] = num_imputer.transform(X_test_aligned[numeric_cols_train])
        print("Missing values in numerical columns of TRAINING set after imputation:\n", X_train_aligned[numeric_cols_train].isnull().sum())
        print("Missing values in numerical columns of TEST set after imputation:\n", X_test_aligned[numeric_cols_train].isnull().sum())
          # Save the num_imputer object to disk, to use later for predictions
        joblib.dump(num_imputer, "preprocessing/num_imputer.pkl")
        print("Imputer saved to 'preprocessing/num_imputer.pkl'")

        return X_train_aligned, X_test_aligned
