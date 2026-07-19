"""Train and save the loan approval model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from utils.preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    encode_target,
    fit_preprocessor,
    transform_with_preprocessor,
    validate_columns,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan.csv"
MODEL_DIR = BASE_DIR / "model"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the loan dataset from CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    return pd.read_csv(path)


def train() -> None:
    """Train, evaluate, and save the model and preprocessing objects."""

    dataframe = load_dataset()
    validate_columns(dataframe, [*FEATURE_COLUMNS, TARGET_COLUMN])

    # Loan_ID is only an identifier, so it should not be used as a model feature.
    dataframe = dataframe.drop(columns=["Loan_ID"], errors="ignore")
    dataframe = dataframe.dropna(subset=[TARGET_COLUMN])

    target, target_encoder = encode_target(dataframe[TARGET_COLUMN])
    raw_features = dataframe[FEATURE_COLUMNS]

    raw_train_features, raw_test_features, y_train, y_test = train_test_split(
        raw_features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    # Fit preprocessing on the training set only to avoid data leakage.
    x_train, artifacts = fit_preprocessor(raw_train_features)
    x_test = transform_with_preprocessor(
        raw_test_features,
        {
            "encoders": artifacts.encoders,
            "categorical_fill_values": artifacts.categorical_fill_values,
            "numerical_fill_values": artifacts.numerical_fill_values,
        },
        artifacts.scaler,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print("Model Evaluation")
    print("----------------")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(y_test, predictions, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    joblib.dump(
        {
            "encoders": artifacts.encoders,
            "categorical_fill_values": artifacts.categorical_fill_values,
            "numerical_fill_values": artifacts.numerical_fill_values,
        },
        MODEL_DIR / "encoder.joblib",
    )
    joblib.dump(artifacts.scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(target_encoder, MODEL_DIR / "target_encoder.joblib")

    print()
    print(f"Saved model to: {MODEL_DIR / 'model.joblib'}")
    print(f"Saved encoders to: {MODEL_DIR / 'encoder.joblib'}")
    print(f"Saved scaler to: {MODEL_DIR / 'scaler.joblib'}")


if __name__ == "__main__":
    train()
