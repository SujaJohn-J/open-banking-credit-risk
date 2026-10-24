# Open Banking Credit Risk Prediction

An end-to-end machine learning project that analyzes synthetic Open Banking transaction data to identify potential credit risk. The project includes transaction analysis, customer-level feature engineering, predictive modeling, model evaluation, explainability, and an interactive dashboard.

## Project Overview

Open Banking enables financial institutions and authorized providers to analyze customers' financial transaction data, subject to appropriate consent and applicable regulations. Understanding income stability, spending behavior, savings, and debt obligations can help inform credit risk assessment.

This project demonstrates how transaction-level data can be transformed into meaningful financial indicators and used to train machine learning models that classify customers into potential high-risk and low-risk groups.

**Important:** This project uses synthetic data for educational and portfolio purposes. It is not a production credit scoring system and must not be used to make real lending decisions.

## Objectives

- Analyze customer income and spending patterns.
- Explore transaction categories and monthly financial behavior.
- Engineer customer-level financial features from transaction data.
- Train and compare machine learning classification models.
- Evaluate model performance using appropriate classification metrics.
- Explore feature importance and model explainability.
- Build an interactive dashboard for exploring results.
- Demonstrate a reproducible, end-to-end data science workflow.

## Key Features

- **Data processing:** Load, validate, and clean customer and transaction datasets.
- **Exploratory data analysis:** Investigate transaction amounts, spending categories, income patterns, and customer characteristics.
- **Feature engineering:** Calculate financial indicators such as income volatility, average monthly expenses, savings rate, cash flow, and debt-to-income ratio.
- **Machine learning:** Train Logistic Regression and Random Forest classifiers.
- **Model evaluation:** Assess ROC-AUC, Gini coefficient, KS statistic, precision, recall, F1-score, and confusion matrices.
- **Explainability:** Examine model coefficients and feature importance to understand which financial indicators influence predictions.
- **Prediction:** Generate estimated risk probabilities and high-risk/low-risk classifications.
- **Interactive dashboard:** Explore customer financial indicators and model results through Streamlit.

## Dataset

The project uses synthetic customer and banking transaction data generated for demonstration purposes.

### Dataset summary

| Item | Description |
|---|---|
| Number of customers | Approximately 5,000 |
| Transaction period | September 2025 – August 2026 |
| Customer data | Demographics, income, employment status |
| Transaction data | Dates, categories, amounts, customer identifiers |
| Target variable | `credit_risk` |
| Target value `0` | Low-risk class |
| Target value `1` | High-risk class |

The exact row counts and class distribution may change if the synthetic dataset is regenerated.

### Transaction categories

The synthetic transactions include categories such as:

- Salary and income
- Housing
- Groceries
- Utilities
- Subscriptions
- Transport
- Shopping
- Entertainment
- Healthcare
- Loan payments

### Data privacy

All records are synthetic. No real customers' bank statements or personal financial records are required for this project.

The raw and processed CSV files and saved model artifacts are excluded from Git through `.gitignore`. Anyone reproducing the project should generate or supply compatible local datasets before running the training and evaluation scripts.

## Feature Engineering

Transaction-level records are aggregated into customer-level financial indicators.

| Feature | Description |
|---|---|
| `income_mean` | Average income over the observed period |
| `income_std` | Standard deviation of income |
| `income_volatility` | Measure of income variability |
| `avg_monthly_expenses` | Average monthly expenditure |
| `avg_cash_flow` | Average difference between income and expenses |
| `avg_loan_payment` | Average loan-related payments |
| `avg_dti` | Average debt-to-income ratio |
| `avg_expense_to_income_ratio` | Expenses relative to income |
| `avg_essential_expenses` | Average spending on essential categories |
| `avg_discretionary_expenses` | Average spending on discretionary categories |
| `savings_rate` | Proportion of income retained as savings, according to the project's calculation |
| `negative_cash_flow_months` | Number of months with negative cash flow |
| `avg_discretionary_spending_ratio` | Average discretionary spending relative to income |
| `avg_shopping_expenses` | Average shopping expenditure |
| `avg_subscription_expenses` | Average subscription expenditure |
| `shopping_ratio` | Shopping expenditure ratio |
| `subscription_ratio` | Subscription expenditure ratio |
| `transaction_count` | Number of observed transactions |
| `age` | Customer age |
| `annual_income` | Customer annual income |

Employment status is also encoded into numerical indicator variables before model training.

These features are intended to demonstrate how raw financial transactions can be transformed into useful inputs for machine learning.

## Machine Learning Models

### 1. Logistic Regression

Logistic Regression is used as an interpretable baseline classification model. It estimates the probability that a customer belongs to the high-risk class.

### 2. Random Forest

Random Forest is used as a nonlinear comparison model. It combines multiple decision trees to capture relationships between financial features and the target.

### Model evaluation

The project uses the following metrics:

- **ROC-AUC:** Measures how well the model ranks positive cases above negative cases across classification thresholds.
- **Gini coefficient:** Calculated as `2 × ROC-AUC − 1`.
- **KS statistic:** Measures the maximum separation between the predicted-score distributions of the two classes.
- **Precision:** Measures the proportion of predicted high-risk customers who belong to the high-risk class.
- **Recall:** Measures the proportion of actual high-risk customers identified by the model.
- **F1-score:** Combines precision and recall.
- **Confusion matrix:** Shows correct and incorrect classifications.

A classification threshold of `0.40` has been explored to illustrate the trade-off between detecting high-risk cases and generating false positives. The appropriate threshold depends on the use case and the relative costs of different errors.

Actual evaluation results should be generated from the current saved models and held-out test dataset rather than assumed from earlier experiments.

## Model Explainability

Model explainability helps investigate how the model uses financial indicators when generating predictions.

The project explores:

- Logistic Regression coefficients
- Random Forest feature importance
- Ranking of influential financial indicators
- Visual comparisons of feature contributions or importance

These analyses help explain model behavior, but they do not establish causation or prove that the model is fair or suitable for real lending decisions.

## Project Structure

```text
open-banking-credit-risk/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── customers.csv
│   │   └── transactions.csv
│   └── processed/
│       └── customer_features.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_transaction_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_explainability.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── build_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   ├── logistic_regression_credit_risk.joblib
│   ├── random_forest_credit_risk.joblib
│   └── scaler.joblib
│
└── dashboard/
    └── app.py
```

*Note: The structure above describes the intended project layout. Local data files and model artifacts may be absent from the public GitHub repository because they are excluded by `.gitignore`.*

## Technologies Used

- **Python** — Main programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical calculations
- **Scikit-learn** — Model training, preprocessing, and evaluation
- **Matplotlib and Seaborn** — Data visualization
- **Plotly** — Interactive visualizations
- **Jupyter Notebook** — Exploratory analysis and experimentation
- **Joblib** — Saving and loading trained models
- **Streamlit** — Interactive dashboard
- **Git and GitHub** — Version control and portfolio presentation

## Installation

### Prerequisites

- Python 3.12 or a compatible Python version
- Git
- Visual Studio Code or another Python editor

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/open-banking-credit-risk.git
cd open-banking-credit-risk
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks environment activation, you can use the environment's Python executable directly or follow your organization's approved PowerShell settings.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure `requirements.txt` includes the packages needed by the scripts you intend to run, including `streamlit` if you use the dashboard.

### 4. Prepare the data

The raw customer and transaction datasets are excluded from Git. Place compatible synthetic datasets in:

```text
data/raw/customers.csv
data/raw/transactions.csv
```

The columns must match the expectations in the data-processing and feature-engineering modules.

If your repository includes a working synthetic-data generation script, you can use that instead of supplying the raw CSV files.

## Usage

Run the following commands from the project root directory.

### 1. Build the feature dataset

```bash
python -m src.build_dataset
```

This step processes the raw data, creates customer-level financial features, generates the synthetic target, and saves the processed dataset.

**Reproducibility note:** Rebuilding the dataset can change the generated target and model results if the random seed, row ordering, or feature calculations change. Retrain the models after rebuilding the dataset.

### 2. Train the models

```bash
python -m src.train
```

This script trains the configured classification models, evaluates their performance, and saves model artifacts and supporting results to the appropriate local directories.

### 3. Evaluate saved models

```bash
python -m src.evaluate
```

The evaluation script uses the saved held-out test dataset and reports the supported classification metrics. It also saves an evaluation summary when the required model and test files are available.

### 4. Generate predictions

```bash
python -m src.predict
```

The prediction script loads a saved model, prepares the customer's features, and returns an estimated high-risk probability and predicted class.

### 5. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard opens in your browser, provided the Streamlit app and its required local data/model files are available.

## Results and Interpretation

The project compares Logistic Regression and Random Forest using held-out data. It also examines the effect of the classification threshold on high-risk recall and false-positive predictions.

Performance metrics should be interpreted in the context of the dataset's construction. Because the target label is generated from a synthetic financial stress score, the model learns patterns associated with that score rather than independently verified real-world credit defaults.

Consequently, strong performance on this dataset would not demonstrate that the model predicts actual loan defaults or is appropriate for real credit decisions.

## Limitations and Future Improvements

Potential future improvements include:

- Use a more realistic, independently defined target variable.
- Validate the model using appropriate real-world outcomes and legally obtained data.
- Add time-based validation to reduce temporal leakage.
- Use cross-validation and systematic hyperparameter tuning.
- Compare threshold choices using business costs and risk requirements.
- Add SHAP-based explainability where appropriate.
- Test calibration of predicted probabilities.
- Assess fairness, bias, privacy, and regulatory compliance.
- Add automated unit tests and data-quality checks.
- Track experiments and model versions.
- Deploy the dashboard with clear warnings and input validation.

## Responsible Use

This is an educational machine learning portfolio project, not a financial product or credit decision service. The generated risk classifications are illustrative only. Real-world credit risk modeling requires appropriate data governance, robust validation, fairness assessment, privacy protections, explainability, and compliance with applicable financial and data-protection laws.

## Author

**Suja John**

Aspiring Data Scientist interested in machine learning, financial data analytics, feature engineering, and explainable AI.

This project demonstrates practical skills in Python, data analysis, predictive modeling, model evaluation, and communicating machine learning results.

## License

This project may be published under the MIT License if you choose to include one. Before adding a license, create a `LICENSE` file with the appropriate license text and confirm that the project's dependencies and any additional assets permit your intended use.
# Open Banking Credit Risk & Affordability Prediction

## Overview

This project explores how Open Banking transaction data can be used to understand customers' financial behaviour and predict credit risk.

The project uses synthetic transaction data inspired by real-world financial data.

## Objectives

- Analyse customer transaction behaviour
- Categorise financial transactions
- Create financial and behavioural features
- Build machine learning models for credit risk prediction
- Evaluate model performance
- Explain model predictions
- Create a dashboard to visualise financial insights

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- SQL
- Power BI / Streamlit

## Machine Learning

The project will experiment with models such as:

- Logistic Regression
- XGBoost
- LightGBM

## Project Structure

```text
data/
notebooks/
src/
models/
dashboard/
reports/

**For now, don't worry about making the README perfect.** We'll improve it as we build the project.

Once you have created `README.md`, the next thing is to create **`requirements.txt`** and put the Python libraries inside it.
