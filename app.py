"""Streamlit app for the loan approval prediction project."""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from predict import MODEL_DIR, predict_loan_status
from utils.preprocessing import FEATURE_COLUMNS


MODEL_PATH = MODEL_DIR / "model.joblib"


st.set_page_config(
    page_title="Smart Loan Approval Prediction",
    page_icon="SL",
    layout="wide",
)


def load_model_for_importance() -> object | None:
    """Load the trained model for the feature importance chart."""

    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def render_sidebar() -> None:
    """Show project details in the sidebar."""

    st.sidebar.title("Project Info")
    st.sidebar.info(
        "A student machine learning project that predicts loan approval with "
        "a Random Forest model."
    )
    st.sidebar.markdown(
        """
        **Built with**
        - Python
        - Pandas, NumPy
        - Scikit-learn
        - Streamlit
        """
    )
    st.sidebar.warning(
        "This app is for learning purposes only. It should not be used for "
        "real lending decisions."
    )


def applicant_form() -> dict | None:
    """Collect applicant details from Streamlit widgets."""

    with st.form("loan_form"):
        st.subheader("Enter Applicant Details")

        left, right = st.columns(2)

        with left:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            property_area = st.selectbox(
                "Property Area",
                ["Urban", "Semiurban", "Rural"],
            )

        with right:
            applicant_income = st.number_input(
                "Applicant Income",
                min_value=0,
                value=5000,
                step=500,
            )
            coapplicant_income = st.number_input(
                "Coapplicant Income",
                min_value=0,
                value=1500,
                step=500,
            )
            loan_amount = st.number_input(
                "Loan Amount (in thousands)",
                min_value=1,
                value=140,
                step=5,
            )
            loan_amount_term = st.selectbox(
                "Loan Amount Term (months)",
                [60, 120, 180, 240, 300, 360, 480],
                index=5,
            )
            credit_history = st.selectbox(
                "Credit History",
                options=[1, 0],
                format_func=lambda value: (
                    "Good / Available" if value == 1 else "Poor / Missing"
                ),
            )

        submitted = st.form_submit_button("Predict Loan Approval")

    if not submitted:
        return None

    total_income = applicant_income + coapplicant_income
    if total_income <= 0:
        st.error("Total income must be greater than zero.")
        return None

    return {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }


def show_feature_importance() -> None:
    """Display feature importance when the trained model supports it."""

    model = load_model_for_importance()
    if model is None or not hasattr(model, "feature_importances_"):
        st.info("Feature importance will appear after the model is trained.")
        return

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    st.subheader("Feature Importance")
    st.bar_chart(importance_df.set_index("Feature"))


def main() -> None:
    """Run the Streamlit application."""

    render_sidebar()

    st.title("Smart Loan Approval Prediction")
    st.write(
        "Fill in the applicant details below. The app uses the same preprocessing "
        "steps that were used during model training."
    )

    if not MODEL_PATH.exists():
        st.error("Model files were not found. Run `python train_model.py` first.")
        return

    applicant_data = applicant_form()

    if applicant_data is not None:
        try:
            result = predict_loan_status(applicant_data)
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
            return
        except Exception as error:
            st.error(f"Prediction failed: {error}")
            return

        st.divider()
        st.subheader("Result")

        if result["prediction"] == "Approved":
            st.success("Loan Status: Approved")
        else:
            st.error("Loan Status: Rejected")

        metric_left, metric_middle, metric_right = st.columns(3)
        metric_left.metric("Confidence", f"{result['confidence']}%")
        metric_middle.metric(
            "Approval Probability",
            f"{result['approval_probability']}%",
        )
        metric_right.metric(
            "Rejection Probability",
            f"{result['rejection_probability']}%",
        )

    st.divider()
    show_feature_importance()


if __name__ == "__main__":
    main()
