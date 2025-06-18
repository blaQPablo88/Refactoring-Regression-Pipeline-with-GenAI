import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a DataFrame by replacing infinite values with NaN and dropping columns containing NaNs.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame with potential infinite values.

    Returns:
    -------
    pd.DataFrame
        A cleaned DataFrame with no infinite or NaN values.

    Example:
    -------
    >>> df_cleaned = clean_infinite_values(df)

    Notes:
    -----
    This method performs in-place replacement but returns a new DataFrame without columns containing NaNs.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna(axis=1)

def standard_scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Applies standard scaling to numeric features in training and test sets.

    Parameters:
    ----------
    X_train : pd.DataFrame
        The training feature set.
    X_test : pd.DataFrame
        The test feature set.

    Returns:
    -------
    tuple:
        - np.ndarray: Scaled training data.
        - np.ndarray: Scaled test data.
        - ColumnTransformer: The fitted transformer for reuse.

    Example:
    -------
    >>> X_train_scaled, X_test_scaled, transformer = standard_scale(X_train, X_test)

    Notes:
    -----
    Non-numeric columns are passed through unchanged using `remainder='passthrough'`.
    """
    scaler = StandardScaler()
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    transformer = ColumnTransformer([
        ("scaler", scaler, num_features)
    ], remainder='passthrough')

    X_train_scaled = transformer.fit_transform(X_train)
    X_test_scaled = transformer.transform(X_test)
    return X_train_scaled, X_test_scaled, transformer