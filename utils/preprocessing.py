"""Small preprocessing helpers used by training and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUMERICAL_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERICAL_COLUMNS
TARGET_COLUMN = "Loan_Status"


@dataclass
class PreprocessingArtifacts:
    """Objects and values learned from the training set."""

    encoders: dict[str, LabelEncoder]
    scaler: StandardScaler
    categorical_fill_values: dict[str, str]
    numerical_fill_values: dict[str, float]
    feature_columns: list[str]
    categorical_columns: list[str]
    numerical_columns: list[str]


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with clean, consistent column names."""

    clean_df = dataframe.copy()
    clean_df.columns = clean_df.columns.str.strip()
    return clean_df


def validate_columns(dataframe: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise a clear error if a required column is missing."""

    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required column(s): {missing_text}")


def get_fill_values(
    dataframe: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, float]]:
    """Calculate fill values from the training data only."""

    categorical_values: dict[str, str] = {}
    numerical_values: dict[str, float] = {}

    for column in CATEGORICAL_COLUMNS:
        mode_value = dataframe[column].mode(dropna=True)
        if mode_value.empty:
            categorical_values[column] = "Unknown"
        else:
            categorical_values[column] = str(mode_value.iloc[0])

    for column in NUMERICAL_COLUMNS:
        numerical_values[column] = float(dataframe[column].median())

    return categorical_values, numerical_values


def fill_missing_values(
    dataframe: pd.DataFrame,
    categorical_values: dict[str, str],
    numerical_values: dict[str, float],
) -> pd.DataFrame:
    """Fill missing values using values learned from training data."""

    clean_df = dataframe.copy()

    for column in CATEGORICAL_COLUMNS:
        if column in clean_df.columns:
            clean_df[column] = clean_df[column].fillna(categorical_values[column])

    for column in NUMERICAL_COLUMNS:
        if column in clean_df.columns:
            clean_df[column] = clean_df[column].fillna(numerical_values[column])

    return clean_df


def fit_preprocessor(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessingArtifacts]:
    """Fit encoders and scaler, then return transformed features."""

    clean_df = normalize_columns(dataframe)
    validate_columns(clean_df, FEATURE_COLUMNS)
    categorical_values, numerical_values = get_fill_values(clean_df)
    clean_df = fill_missing_values(clean_df, categorical_values, numerical_values)

    encoded_df = clean_df[FEATURE_COLUMNS].copy()
    encoders: dict[str, LabelEncoder] = {}

    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        encoded_df[column] = encoder.fit_transform(encoded_df[column].astype(str))
        encoders[column] = encoder

    scaler = StandardScaler()
    encoded_df[NUMERICAL_COLUMNS] = scaler.fit_transform(encoded_df[NUMERICAL_COLUMNS])

    artifacts = PreprocessingArtifacts(
        encoders=encoders,
        scaler=scaler,
        categorical_fill_values=categorical_values,
        numerical_fill_values=numerical_values,
        feature_columns=FEATURE_COLUMNS.copy(),
        categorical_columns=CATEGORICAL_COLUMNS.copy(),
        numerical_columns=NUMERICAL_COLUMNS.copy(),
    )

    return encoded_df, artifacts


def transform_with_preprocessor(
    dataframe: pd.DataFrame,
    preprocessing: dict[str, Any],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Apply fitted preprocessing objects to new data."""

    clean_df = normalize_columns(dataframe)
    validate_columns(clean_df, FEATURE_COLUMNS)

    encoders = preprocessing["encoders"]
    categorical_values = preprocessing["categorical_fill_values"]
    numerical_values = preprocessing["numerical_fill_values"]

    clean_df = fill_missing_values(clean_df, categorical_values, numerical_values)

    transformed_df = clean_df[FEATURE_COLUMNS].copy()

    for column in CATEGORICAL_COLUMNS:
        encoder = encoders[column]
        transformed_df[column] = transformed_df[column].astype(str).apply(
            lambda value: value if value in encoder.classes_ else encoder.classes_[0]
        )
        transformed_df[column] = encoder.transform(transformed_df[column])

    transformed_df[NUMERICAL_COLUMNS] = scaler.transform(
        transformed_df[NUMERICAL_COLUMNS]
    )

    return transformed_df


def encode_target(target: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    """Encode target labels such as Y/N into numbers."""

    encoder = LabelEncoder()
    encoded_target = pd.Series(
        encoder.fit_transform(target.astype(str)),
        index=target.index,
        name=target.name,
    )
    return encoded_target, encoder
