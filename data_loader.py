"""
eda.py
------
Exploratory Data Analysis for the Telco Customer Churn dataset.

Run this script directly to print summary stats and pop up
matplotlib figures for numeric and categorical feature distributions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_loader import load_data


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges has 11 blank-string rows, all belonging to customers
    with tenure == 0 (brand new customers with no billing history yet).
    We fill these with 0.0 and cast the column to float.
    """
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    return df


def basic_overview(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nTarget distribution:\n", df["Churn"].value_counts())
    print("\nDescribe (numeric):\n", df.describe())
    print("\nCorrelation (numeric):\n", df.select_dtypes(include=["float64", "int64"]).corr())


def plot_histogram(df: pd.DataFrame, column_name: str) -> None:
    plt.figure(figsize=(5, 3))
    sns.histplot(df[column_name], kde=True)
    plt.title(f"Distribution of {column_name}")

    col_mean = df[column_name].mean()
    col_median = df[column_name].median()
    plt.axvline(col_mean, color="red", linestyle="--", label=f"Mean: {col_mean:.2f}")
    plt.axvline(col_median, color="green", linestyle="--", label=f"Median: {col_median:.2f}")
    plt.legend()
    plt.show()


def plot_boxplot(df: pd.DataFrame, column_name: str) -> None:
    plt.figure(figsize=(5, 3))
    sns.boxplot(y=df[column_name])
    plt.title(f"Boxplot of {column_name}")
    plt.ylabel(column_name)
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 6))
    sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()


def plot_categorical_counts(df: pd.DataFrame) -> None:
    object_cols = df.select_dtypes(include="object").columns.to_list()
    object_cols = ["SeniorCitizen"] + object_cols

    for col in object_cols:
        plt.figure(figsize=(4, 3))
        sns.countplot(x=df[col])
        plt.title(f"Count plot of {col}")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()


def run_eda() -> None:
    df = load_data()
    df = df.drop(columns=["customerID"])
    df = clean_total_charges(df)

    basic_overview(df)

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        plot_histogram(df, col)
        plot_boxplot(df, col)

    plot_correlation_heatmap(df)
    plot_categorical_counts(df)


if __name__ == "__main__":
    run_eda()
