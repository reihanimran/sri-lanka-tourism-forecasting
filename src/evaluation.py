"""
evaluation.py
Forecast accuracy metrics: MAE, RMSE, MAPE.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape(actual, predicted):
    """Mean Absolute Percentage Error, ignoring zero-actual months (COVID)."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = a != 0
    return np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100


def evaluate(actual, predicted, model_name, verbose=True):
    """Return a dict of MAE, RMSE, MAPE for one model."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    mp = mape(a, p)
    if verbose:
        print(f"{model_name:30s}  MAE={mae:9,.0f}  RMSE={rmse:9,.0f}  MAPE={mp:6.2f}%")
    return {'Model': model_name, 'MAE': round(mae, 2),
            'RMSE': round(rmse, 2), 'MAPE': round(mp, 2)}
