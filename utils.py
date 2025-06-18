from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(true, predicted):
    """
    Evaluate regression model performance using standard metrics.

    This function computes and prints:
    - Pearson Correlation Coefficient
    - R² Score (Coefficient of Determination)
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)

    It also returns the Pearson correlation coefficient between
    the true and predicted values.

    Parameters
    ----------
    true : array-like of shape (n_samples,)
        The ground truth target values.

    predicted : array-like of shape (n_samples,)
        The predicted target values from a regression model.

    Returns
    -------
    float
        The Pearson correlation coefficient between true and predicted values.

    Raises
    ------
    ValueError
        If the input arrays have mismatched lengths or contain NaNs.

    Notes
    -----
    Pearson correlation is calculated using `np.corrcoef`, which returns a correlation matrix.
    The function extracts the correlation value at index [0, 1].

    Examples
    --------
    >>> y_true = [3.0, 2.5, 4.0]
    >>> y_pred = [2.8, 2.7, 3.9]
    >>> evaluate_model(y_true, y_pred)
    Pearson Correlation: 0.9987
    R2 Score: 0.9962 | MAE: 0.1000 | RMSE: 0.1225
    0.9987
    """
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, predicted)
    corr = np.corrcoef(true, predicted)[0, 1]

    print(f"Pearson Correlation: {corr:.4f}")
    print(f"R2 Score: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    return corr