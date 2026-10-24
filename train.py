import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from evaluate import evaluate_model


# --------------------------------
# Project paths
# --------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "customer_features.csv"
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models_S"
)

os.makedirs(MODELS_DIR, exist_ok=True)


# --------------------------------
# Load data
# --------------------------------

print("Loading processed data...")

data = pd.read_csv(DATA_PATH)

print("Data shape:", data.shape)


# --------------------------------
# Prepare target
# --------------------------------

y = data["credit_risk"]


# --------------------------------
# Encode employment status
# --------------------------------

if "employment_status" in data.columns:

    data = pd.get_dummies(
        data,
        columns=["employment_status"],
        drop_first=True
    )


# --------------------------------
# Select features
# --------------------------------

FEATURE_COLUMNS = [
    "income_mean",
    "income_std",
    "income_volatility",
    "avg_monthly_expenses",
    "avg_cash_flow",
    "avg_loan_payment",
    "avg_dti",
    "avg_expense_to_income_ratio",
    "avg_essential_expenses",
    "avg_discretionary_expenses",
    "age",
    "annual_income",
    "transaction_count",
    "savings_rate",
    "negative_cash_flow_months",
    "avg_discretionary_spending_ratio",
    "avg_shopping_expenses",
    "avg_subscription_expenses",
    "shopping_ratio",
    "subscription_ratio",
    "employment_status_Part-time",
    "employment_status_Self-employed"
]


X = data.reindex(
    columns=FEATURE_COLUMNS,
    fill_value=0
)


print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# --------------------------------
# Train/Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])


# =================================
# Logistic Regression
# =================================

print("\nTraining Logistic Regression...")

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train,
    y_train
)


# --------------------------------
# Evaluate Logistic Regression
# --------------------------------

print("\nLOGISTIC REGRESSION RESULTS")

logistic_results = evaluate_model(
    logistic_model,
    X_test,
    y_test,
    threshold=0.40
)


# --------------------------------
# Save Logistic Regression
# --------------------------------

logistic_path = os.path.join(
    MODELS_DIR,
    "logistic_regression_credit_risk.joblib"
)

joblib.dump(
    logistic_model,
    logistic_path
)

print("\nLogistic Regression saved:")
print(logistic_path)


# =================================
# Random Forest
# =================================

print("\nTraining Random Forest...")

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)


# --------------------------------
# Evaluate Random Forest
# --------------------------------

print("\nRANDOM FOREST RESULTS")

random_forest_results = evaluate_model(
    random_forest_model,
    X_test,
    y_test,
    threshold=0.40
)


# --------------------------------
# Save Random Forest
# --------------------------------

random_forest_path = os.path.join(
    MODELS_DIR,
    "random_forest_credit_risk.joblib"
)

joblib.dump(
    random_forest_model,
    random_forest_path
)

print("\nRandom Forest saved:")
print(random_forest_path)


# =================================
# Model Comparison
# =================================

print("\nMODEL COMPARISON")
print("=" * 50)

print(
    f"Logistic Regression ROC-AUC: "
    f"{logistic_results['roc_auc']:.3f}"
)

print(
    f"Random Forest ROC-AUC:       "
    f"{random_forest_results['roc_auc']:.3f}"
)


# =================================
# Final Model
# =================================

if logistic_results["roc_auc"] >= random_forest_results["roc_auc"]:

    final_model = logistic_model

    final_model_name = "Logistic Regression"

else:

    final_model = random_forest_model

    final_model_name = "Random Forest"


# Save final model

final_model_path = os.path.join(
    MODELS_DIR,
    "final_credit_risk_model.joblib"
)

joblib.dump(
    final_model,
    final_model_path
)


print("\nFINAL MODEL")
print("=" * 50)

print("Selected Model:", final_model_name)

print("Saved as:")
print(final_model_path)