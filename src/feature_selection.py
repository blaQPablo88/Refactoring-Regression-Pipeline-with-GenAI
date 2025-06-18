from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

def select_features(X, y, k=200):
    """
    Selects the top `k` features from a dataset using univariate linear regression (f_regression).

    Parameters:
    ----------
    X : pd.DataFrame
        Input features (independent variables).
    y : pd.Series or np.ndarray
        Target variable (dependent variable).
    k : int, optional (default=200)
        Number of top features to select based on F-score.

    Returns:
    -------
    tuple:
        - pd.DataFrame: Transformed dataset with only the top `k` features.
        - SelectKBest: Fitted feature selector object.
        - list: Names of the selected features.

    Raises:
    ------
    ValueError
        If `k` is greater than the number of features in `X`.

    Example:
    -------
    >>> X_new, selector, features = select_features(X, y, k=50)

    Notes:
    -----
    This function uses sklearn's `SelectKBest` with `f_regression` as the scoring function.
    """
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()]
    return pd.DataFrame(X_selected, columns=selected_columns), selector, selected_columns