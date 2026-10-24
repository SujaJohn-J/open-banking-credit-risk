print("SCRIPT STARTED")
import os
import joblib
import pandas as pd


# Get the project root directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# Path to the saved model
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "logistic_regression_credit_risk.joblib"
)


# Load the trained model
model = joblib.load(MODEL_PATH)


# Selected classification threshold
THRESHOLD = 0.40


def predict_credit_risk(customer_data):
    """
    Predict credit risk for a customer.

    Parameters
    ----------
    customer_data : dict
        Dictionary containing customer features.

    Returns
    -------
    dict
        Prediction result containing risk probability
        and predicted credit risk class.
    """

    # Convert dictionary to DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Predict probability for credit_risk = 1
    risk_probability = model.predict_proba(customer_df)[0][1]

    # Apply threshold
    prediction = int(risk_probability >= THRESHOLD)

    return {
        "risk_probability": round(float(risk_probability), 4),
        "credit_risk_prediction": prediction
    }
if __name__ == "__main__":

    # Example customer data
    customer_data = {
        "income_mean": 15000,
        "income_std": 2000,
        "income_volatility": 0.13,
        "avg_monthly_expenses": 10000,
        "avg_cash_flow": 5000,
        "avg_loan_payment": 2000,
        "avg_dti": 0.25,
        "avg_expense_to_income_ratio": 0.65,
        "avg_essential_expenses": 6000,
        "avg_discretionary_expenses": 2000,
        "age": 35,
        "annual_income": 180000,
        "transaction_count": 250,
        "savings_rate": 0.20,
        "negative_cash_flow_months": 1,
        "avg_discretionary_spending_ratio": 0.20,
        "avg_shopping_expenses": 1000,
        "avg_subscription_expenses": 500,
        "shopping_ratio": 0.10,
        "subscription_ratio": 0.05,
        "employment_status_Part-time": 0,
        "employment_status_Self-employed": 0
    }

    result = predict_credit_risk(customer_data)

    print("\nCredit Risk Prediction")
    print("=" * 40)
    print("Risk Probability:", result["risk_probability"])
    print(
        "Credit Risk Prediction:",
        result["credit_risk_prediction"]
    )