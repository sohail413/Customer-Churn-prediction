"""
data_loader.py
--------------
Downloads the Telco Customer Churn dataset from Kaggle (via kagglehub)
and loads it into a pandas DataFrame.
"""

import os
import kagglehub
import pandas as pd


def download_dataset() -> str:
    """Download the dataset and return the local folder path."""
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    print(f"Dataset downloaded to: {path}")
    return path


def load_data() -> pd.DataFrame:
    """Download (if needed) and load the raw churn dataset as a DataFrame."""
    path = download_dataset()
    csv_path = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    df = load_data()
    print(df.shape)
    print(df.head())
