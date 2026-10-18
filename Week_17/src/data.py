"""Shared data loading and preprocessing for training and monitoring."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"


def load_raw_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].replace(" ", pd.NA), errors="coerce")
    data[TARGET] = data[TARGET].map({"Yes": 1, "No": 0}).astype("int64")
    return data


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = data.drop(columns=[TARGET, "customerID"])
    return features, data[TARGET]


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(include=["object"]).columns.tolist()
    numeric = features.select_dtypes(exclude=["object"]).columns.tolist()
    numeric_pipeline = Pipeline([( "imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])
