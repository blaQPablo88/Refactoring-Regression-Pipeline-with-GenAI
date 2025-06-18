import pandas as pd

def load_data(train_path: str, test_path: str, sample_path: str):
    """
    Loads training, test, and sample submission data from specified file paths.

    This function reads the training and test datasets from `.parquet` files using the
    'pyarrow' engine, and reads the sample submission file from a CSV format.
    It returns all three as pandas DataFrames.

    Parameters:
    ----------
    train_path : str
        The file path to the training dataset in Parquet format.
    test_path : str
        The file path to the test dataset in Parquet format.
    sample_path : str
        The file path to the sample submission CSV file.

    Returns:
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing three pandas DataFrames:
        - train_df: The training dataset
        - test_df: The test dataset
        - sample_df: The sample submission data

    Raises:
    ------
    FileNotFoundError
        If any of the specified file paths do not exist.
    ValueError
        If the file format is not compatible with pandas.read_parquet or pandas.read_csv.
    ImportError
        If the 'pyarrow' library is not installed for reading Parquet files.

    Example:
    -------
    >>> train_df, test_df, sample_df = load_data(
    ...     "data/train.parquet",
    ...     "data/test.parquet",
    ...     "data/sample_submission.csv"
    ... )
    >>> print(train_df.shape)
    (100000, 200)

    Notes:
    -----
    - Ensure that the `pyarrow` library is installed, or `pandas.read_parquet` will raise an ImportError.
      You can install it with `pip install pyarrow`.
    - The function assumes that the files exist and are well-formed. Use `try/except` to catch issues in production.
    - This function does not perform any schema validation or preprocessing — just raw loading.
    """
    print("Loading data...")
    train_df = pd.read_parquet(train_path, engine="pyarrow")
    test_df = pd.read_parquet(test_path, engine="pyarrow")
    sample_df = pd.read_csv(sample_path)
    return train_df, test_df, sample_df
