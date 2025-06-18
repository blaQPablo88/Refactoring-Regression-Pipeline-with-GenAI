import pandas as pd
import numpy as np

def train_and_evaluate(models: dict, X_train, y_train, X_test, y_test, test_final, id_column):
    """
    Trains multiple models, evaluates performance using Pearson correlation, and saves predictions.

    Parameters:
    ----------
    models : dict
        Dictionary of model names and instantiated regressors.
    X_train : array-like
        Scaled or processed training features.
    y_train : array-like
        Training labels.
    X_test : array-like
        Scaled or processed test features.
    y_test : array-like
        Test labels for evaluation.
    test_final : array-like
        Final test set to generate predictions for submission.
    id_column : array-like
        ID column for use in output prediction files.

    Returns:
    -------
    pd.DataFrame
        A summary table of model names and their Pearson correlation coefficients.

    Raises:
    ------
    Exception
        If model training or prediction fails.

    Example:
    -------
    >>> results_df = train_and_evaluate(models, X_train, y_train, X_test, y_test, test_final, id_column)

    Notes:
    -----
    Saves each model's prediction to a CSV named `{model_name}_prediction.csv`.
    """
    results = []
    
    for name, model in models.items():
        print(f"\n{name} ➜")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            corr = np.corrcoef(y_test, y_pred)[0, 1]
            print(f"Pearson Correlation: {corr:.4f}")
            results.append((name, corr))

            preds = model.predict(test_final)
            pd.DataFrame({"ID": id_column, "prediction": preds}).to_csv(f"{name}_prediction.csv", index=False)
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    return pd.DataFrame(results, columns=["Model", "Pearson Correlation"])