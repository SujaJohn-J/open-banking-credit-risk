import pandas as pd
from pathlib import Path


def load_raw_data(data_dir="data/raw"):
    """
    Load raw customer and transaction data.

    Parameters
    ----------
    data_dir : str or Path
        Path to the raw data directory.

    Returns
    -------
    customers : pandas DataFrame
        Customer demographic data.

    transactions : pandas DataFrame
        Banking transaction data.
    """

    data_dir = Path(data_dir)

    customers_path = data_dir / "customers.csv"
    transactions_path = data_dir / "transactions.csv"

    customers = pd.read_csv(customers_path)

    transactions = pd.read_csv(
        transactions_path,
        parse_dates=["date"]
    )

    return customers, transactions


def validate_data(customers, transactions):
    """
    Validate required columns in the datasets.
    """

    customer_columns = [
        "customer_id",
        "age",
        "annual_income",
        "employment_status"
    ]

    transaction_columns = [
        "customer_id",
        "date",
        "description",
        "category",
        "amount"
    ]

    missing_customer_columns = [
        column
        for column in customer_columns
        if column not in customers.columns
    ]

    missing_transaction_columns = [
        column
        for column in transaction_columns
        if column not in transactions.columns
    ]

    if missing_customer_columns:
        raise ValueError(
            f"Missing customer columns: "
            f"{missing_customer_columns}"
        )

    if missing_transaction_columns:
        raise ValueError(
            f"Missing transaction columns: "
            f"{missing_transaction_columns}"
        )

    print("Data validation successful!")


def clean_customers(customers):
    """
    Clean customer data.
    """

    customers = customers.copy()

    # Remove duplicate customers
    customers = customers.drop_duplicates(
        subset="customer_id"
    )

    # Remove rows with missing customer IDs
    customers = customers.dropna(
        subset=["customer_id"]
    )

    # Fill missing employment status
    customers["employment_status"] = (
        customers["employment_status"]
        .fillna("Unknown")
    )

    # Remove invalid ages
    customers = customers[
        customers["age"].between(18, 100)
    ]

    # Remove invalid income values
    customers = customers[
        customers["annual_income"] > 0
    ]

    return customers


def clean_transactions(transactions):
    """
    Clean transaction data.
    """

    transactions = transactions.copy()

    # Remove duplicate transactions
    transactions = transactions.drop_duplicates()

    # Convert date to datetime
    transactions["date"] = pd.to_datetime(
        transactions["date"],
        errors="coerce"
    )

    # Convert amount to numeric
    transactions["amount"] = pd.to_numeric(
        transactions["amount"],
        errors="coerce"
    )

    # Remove missing important values
    transactions = transactions.dropna(
        subset=[
            "customer_id",
            "date",
            "category",
            "amount"
        ]
    )

    # Remove zero-value transactions
    transactions = transactions[
        transactions["amount"] != 0
    ]

    # Sort transactions
    transactions = transactions.sort_values(
        by=["customer_id", "date"]
    ).reset_index(drop=True)

    return transactions


def process_raw_data(data_dir="data/raw"):
    """
    Complete raw data processing pipeline.

    Returns
    -------
    customers : pandas DataFrame
        Cleaned customer data.

    transactions : pandas DataFrame
        Cleaned transaction data.
    """

    # Load data
    customers, transactions = load_raw_data(
        data_dir
    )

    print(
        f"Customers loaded: {len(customers):,}"
    )

    print(
        f"Transactions loaded: "
        f"{len(transactions):,}"
    )

    # Validate data
    validate_data(
        customers,
        transactions
    )

    # Clean data
    customers = clean_customers(
        customers
    )

    transactions = clean_transactions(
        transactions
    )

    print(
        f"Customers after cleaning: "
        f"{len(customers):,}"
    )

    print(
        f"Transactions after cleaning: "
        f"{len(transactions):,}"
    )

    return customers, transactions