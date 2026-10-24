import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


def evaluate_model(model, X_test, y_test, threshold=0.40):
    """
    Evaluate a classification model for credit risk.

    Parameters
    ----------
    model : trained machine learning model
        The trained model to evaluate.

    X_test : pandas DataFrame
        Test features.

    y_test : pandas Series
        Actual target values.

    threshold : float, default=0.40
        Classification threshold.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    # Predict probabilities for class 1
    y_prob = model.predict_proba(X_test)[:, 1]

    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)

    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_prob)

    # Gini Score
    gini = 2 * roc_auc - 1

    # KS Statistic
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    ks = np.max(tpr - fpr)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification Report
    report = classification_report(
        y_test,
        y_pred
    )

    # Print results
    print("\nModel Evaluation")
    print("=" * 60)

    print("\nClassification Report")
    print("-" * 60)
    print(report)

    print("\nROC-AUC :", round(roc_auc, 3))
    print("Gini    :", round(gini, 3))
    print("KS      :", round(ks, 3))

    print("\nConfusion Matrix")
    print(cm)

    # Return results
    return {
        "roc_auc": roc_auc,
        "gini": gini,
        "ks": ks,
        "confusion_matrix": cm,
        "y_prob": y_prob,
        "y_pred": y_pred
    }