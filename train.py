"""
preprocessing.py
----------------
Cleaning, encoding, train/test splitting, and SMOTE balancing
for the Telco Customer Churn dataset.
"""

import pickle
from typing import Tuple, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier column, fix TotalCharges, encode target as 0/1."""
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Blank strings in TotalCharges correspond to tenure == 0 (new customers)
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    df["Churn"] = df["Churn"].replace({"No": 0, "Yes": 1})

    return df


def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode all object (categorical) columns and return the fitted encoders."""
    df = df.copy()
    object_cols = df.select_dtypes(include="object").columns

    encoders: Dict[str, LabelEncoder] = {}
    for column in object_cols:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoders[column] = le

    return df, encoders


def save_encoders(encoders: Dict[str, LabelEncoder], path: str = "models/encoders.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump(encoders, f)


def split_and_balance(
    df: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2, random_state: int = 42
):
    """
    Train/test split, then apply SMOTE to the training set ONLY.
    The test set is left untouched to give a realistic evaluation.
    """
    x = df.drop(columns=[target_col])
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    smote = SMOTE(random_state=random_state)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

    return x_train_smote, x_test, y_train_smote, y_test


def prepare_data(raw_df: pd.DataFrame, encoders_path: str = "models/encoders.pkl"):
    """Full preprocessing pipeline: clean -> encode -> save encoders -> split/balance."""
    df = clean_data(raw_df)
    df, encoders = encode_categoricals(df)
    save_encoders(encoders, encoders_path)

    x_train, x_test, y_train, y_test = split_and_balance(df)
    feature_names = df.drop(columns=["Churn"]).columns.tolist()

    return x_train, x_test, y_train, y_test, feature_names
