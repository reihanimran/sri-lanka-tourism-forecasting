"""
models.py
Feature engineering helpers for the machine learning and hybrid models.
"""
import pandas as pd
import numpy as np


def make_ml_features(series):
    """
    Build lag + rolling + calendar features from an arrivals series.
    Used by the standalone XGBoost model.
    """
    d = pd.DataFrame({'y': series})
    for lag in [1, 2, 3, 6, 12]:
        d[f'lag{lag}'] = d['y'].shift(lag)
    d['rm3'] = d['y'].shift(1).rolling(3).mean()
    d['rm6'] = d['y'].shift(1).rolling(6).mean()
    d['month'] = d.index.month
    d['quarter'] = d.index.quarter
    return d


ML_FEATURES = ['lag1', 'lag2', 'lag3', 'lag6', 'lag12', 'rm3', 'rm6', 'month', 'quarter']


def make_residual_features(series):
    """
    Build features from SARIMA residuals for the hybrid model.
    Kept deliberately simple to avoid overfitting on a small residual set.
    """
    d = pd.DataFrame({'r': series})
    d['lag1'] = d['r'].shift(1)
    d['lag12'] = d['r'].shift(12)
    d['month'] = d.index.month
    return d


RESID_FEATURES = ['lag1', 'lag12', 'month']
