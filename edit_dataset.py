import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
import random as rd
import matplotlib.pyplot as plt

def to_number(attribute, dataframe):
    enc = LabelEncoder()
    enc.fit(dataframe[attribute])
    dataframe[attribute] = enc.transform(dataframe[attribute])

def handle_missing_values(dataframe):
    # Use lowercase 'abondance' as per normalized column names
    sum_values = dataframe.loc[dataframe['abondance'] != 'NSP', 'abondance'].astype(float).sum()
    count_values = dataframe.loc[dataframe['abondance'] != 'NSP', 'abondance'].astype(float).count()
    final_number = round(sum_values / count_values, 2)
    dataframe['abondance'] = dataframe['abondance'].replace('NSP', final_number)

    reviews = [val for val in dataframe["plus"] if isinstance(val, str) and len(val) > 4]
    for i, val in dataframe["plus"].items():
        if not isinstance(val, str) or len(val) < 4:
            random_review = rd.choice(reviews)
            dataframe.at[i, 'plus'] = random_review


def preprocess_features(dataframe, target_column):
    # Debugging: Check initial columns
    print(f"Columns before preprocessing: {dataframe.columns.tolist()}")
    
    # Drop non-numeric columns but exclude target_column to keep it intact
    if target_column in dataframe.columns:
        dataframe = dataframe.drop(['horodateur', 'row.names'], axis=1, errors='ignore')
    else:
        raise KeyError(f"Target column '{target_column}' is missing before preprocessing.")

    # Separate features and target
    y = dataframe[target_column]
    X = dataframe.drop(target_column, axis=1)
    
    # Debugging: Check that target column exists
    print(f"Columns after preprocessing (features): {X.columns.tolist()}")
    print(f"Target column: {target_column}")
    
    return X, y


def apply_smote(dataframe, target_column):
    try:
        # Preprocess features
        X, y = preprocess_features(dataframe, target_column)
    except KeyError as e:
        print(f"Error in preprocessing: {e}")
        print(f"Available columns: {dataframe.columns.tolist()}")
        raise

    print(f"Original class distribution: {Counter(y)}")

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

    print(f"Resampled class distribution: {Counter(y_resampled)}")

    # Decode target labels back to original
    y_resampled = label_encoder.inverse_transform(y_resampled)

    # Combine resampled data into a new DataFrame
    resampled_dataframe = pd.concat(
        [pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[target_column])],
        axis=1,
    )
    return resampled_dataframe

def main():
    file_path = r"./Data cat personality and predation Cordonnier et al.xlsx"
    dataframe1 = pd.read_excel(file_path, sheet_name='Data')

    # Normalize column names
    dataframe1.columns = dataframe1.columns.str.strip().str.lower()
    print("Normalized Columns:", dataframe1.columns.tolist())

    target_column = 'race'

    if target_column not in dataframe1.columns:
        print(f"Error: Target column '{target_column}' is not in the dataset.")
        return

    to_number('sexe', dataframe1)
    to_number('nombre', dataframe1)
    to_number('age', dataframe1)
    to_number('zone', dataframe1)
    to_number('logement', dataframe1)

    handle_missing_values(dataframe1)

    # Confirm target column still exists
    if target_column not in dataframe1.columns:
        print(f"Error: Target column '{target_column}' is missing after preprocessing.")
        print("Available columns:", dataframe1.columns.tolist())
        return

    print(f"Target Column Distribution Before SMOTE:\n{dataframe1[target_column].value_counts()}")

    dataframe1 = apply_smote(dataframe1, target_column)

    modified_file_path = r"./Modified_Data_with_SMOTE.xlsx"
    dataframe1.to_excel(modified_file_path, index=False)
    print(f"Modified dataset saved to {modified_file_path}")
