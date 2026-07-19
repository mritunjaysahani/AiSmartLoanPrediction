"""Prediction utilities used by the Streamlit app and quick scripts."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from utils.preprocessing import FEATURE_COLUMNS, transform_with_preprocessor


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


def load_artifacts(model_dir: Path = MODEL_DIR) -> tuple[object, dict, object]:
    """Load the trained model and preprocessing objects from disk."""

    model_path = model_dir / "model.joblib"
    encoder_path = model_dir / "encoder.joblib"
    scaler_path = model_dir / "scaler.joblib"

    if not model_path.exists() or not encoder_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Model files are missing. Run 'python train_model.py' before prediction."
        )

    model = joblib.load(model_path)
    preprocessing = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)

    required_keys = {
        "encoders",
        "categorical_fill_values",
        "numerical_fill_values",
    }
    if not isinstance(preprocessing, dict) or not required_keys.issubset(preprocessing):
        raise ValueError(
            "Preprocessing artifacts are outdated. Run 'python train_model.py' again."
        )

    return model, preprocessing, scaler


def predict_loan_status(applicant_data: dict) -> dict:
    """Predict loan approval for a single applicant."""

    missing_fields = [field for field in FEATURE_COLUMNS if field not in applicant_data]
    if missing_fields:
        raise ValueError(f"Missing input field(s): {', '.join(missing_fields)}")

    model, preprocessing, scaler = load_artifacts()
    input_df = pd.DataFrame([applicant_data], columns=FEATURE_COLUMNS)
    processed_input = transform_with_preprocessor(input_df, preprocessing, scaler)

    prediction = int(model.predict(processed_input)[0])
    probabilities = model.predict_proba(processed_input)[0]
    confidence = float(probabilities[prediction])

    return {
        "prediction": "Approved" if prediction == 1 else "Rejected",
        "confidence": round(confidence * 100, 2),
        "approval_probability": round(float(probabilities[1]) * 100, 2),
        "rejection_probability": round(float(probabilities[0]) * 100, 2),
    }


if __name__ == "__main__":
    sample_applicant = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "Property_Area": "Semiurban",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 140,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
    }

    result = predict_loan_status(sample_applicant)
    print(result)
