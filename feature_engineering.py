import pandas as pd
import numpy as np


def create_features(data):
    """
    Prepare customer features for the credit risk model.

    Parameters
    ----------
    data : pandas DataFrame
        Customer-level dataset.

    Returns
    -------
    pandas DataFrame
        Dataset containing engineered features.
    """

    df = data.copy()


    # --------------------------------
    # Income features
    # --------------------------------

    if "income_mean" in df.columns:
        df["income_volatility"] = (
            df["income_std"] /
            df["income_mean"].replace(0, np.nan)
        )

        df["income_volatility"] = (
            df["income_volatility"]
            .fillna(0)
        )


    # --------------------------------
    # Expense to income ratio
    # --------------------------------

    if (
        "avg_monthly_expenses" in df.columns
        and "income_mean" in df.columns
    ):

        df["avg_expense_to_income_ratio"] = (
            df["avg_monthly_expenses"] /
            df["income_mean"].replace(0, np.nan)
        )

        df["avg_expense_to_income_ratio"] = (
            df["avg_expense_to_income_ratio"]
            .fillna(0)
        )


    # --------------------------------
    # Debt to income ratio
    # --------------------------------

    if (
        "avg_loan_payment" in df.columns
        and "income_mean" in df.columns
    ):

        df["avg_dti"] = (
            df["avg_loan_payment"] /
            df["income_mean"].replace(0, np.nan)
        )

        df["avg_dti"] = (
            df["avg_dti"]
            .fillna(0)
        )


    # --------------------------------
    # Savings rate
    # --------------------------------

    if (
        "avg_cash_flow" in df.columns
        and "income_mean" in df.columns
    ):

        df["savings_rate"] = (
            df["avg_cash_flow"] /
            df["income_mean"].replace(0, np.nan)
        )

        df["savings_rate"] = (
            df["savings_rate"]
            .fillna(0)
        )


    # --------------------------------
    # Shopping ratio
    # --------------------------------

    if (
        "avg_shopping_expenses" in df.columns
        and "avg_monthly_expenses" in df.columns
    ):

        df["shopping_ratio"] = (
            df["avg_shopping_expenses"] /
            df["avg_monthly_expenses"].replace(0, np.nan)
        )

        df["shopping_ratio"] = (
            df["shopping_ratio"]
            .fillna(0)
        )


    # --------------------------------
    # Subscription ratio
    # --------------------------------

    if (
        "avg_subscription_expenses" in df.columns
        and "avg_monthly_expenses" in df.columns
    ):

        df["subscription_ratio"] = (
            df["avg_subscription_expenses"] /
            df["avg_monthly_expenses"].replace(0, np.nan)
        )

        df["subscription_ratio"] = (
            df["subscription_ratio"]
            .fillna(0)
        )


    return df