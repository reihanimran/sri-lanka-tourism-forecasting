"""
run_full_matrix.py
------------------
Reproduces EVERY quantity the thesis reports, for BOTH data versions, so the
thesis tables can be filled with one internally-consistent set of numbers.

For each version (zero, synthetic) it produces:
  - Seasonal Naive, SARIMA, Auto-SARIMA, SARIMAX
  - XGBoost  DEFAULT  (n=200, depth=4, lr=0.05)            [notebook 04 config]
  - XGBoost  TUNED    (best of 54-combo TimeSeriesSplit grid) [notebook 09/12]
  - Hybrid   DEFAULT  (n=50, depth=2; resid feats lag1,lag12,month)  [notebook 05]
  - Hybrid   TUNED    (n=25, depth=2; resid feats lag1,lag2,lag12,month) [nb 11/12]

Metrics: MAE, RMSE, MAPE, WAPE.  Test = real 2025-01..2026-03 (same for both).
Writes results/full_matrix.csv and prints a reconciliation-ready table.
"""
import sys, os, itertools, warnings
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
import pmdarima as pm
import xgboost as xgb

from data_loader import get_monthly_total, get_monthly_total_synthetic, train_test_split
from models import make_ml_features, ML_FEATURES

DATA = os.path.join(os.path.dirname(__file__), 'data', 'Tourism_MOM_Dataset.csv')
SPLIT = '2025-01-01'


def metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    mae = np.mean(np.abs(a - p))
    rmse = np.sqrt(np.mean((a - p) ** 2))
    m = a != 0
    mape = np.mean(np.abs((a[m] - p[m]) / a[m])) * 100
    wape = np.sum(np.abs(a - p)) / np.sum(np.abs(a)) * 100
    return dict(MAE=round(mae), RMSE=round(rmse), MAPE=round(mape, 2), WAPE=round(wape, 2))


def fit_sarima(train):
    return SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)


def xgb_predict(y, params):
    fd = make_ml_features(y).dropna()
    tr, te = fd[fd.index < SPLIT], fd[fd.index >= SPLIT]
    m = xgb.XGBRegressor(random_state=42, verbosity=0, **params)
    m.fit(tr[ML_FEATURES], tr['y'])
    return pd.Series(m.predict(te[ML_FEATURES]), index=te.index)


def tune_xgb(y):
    """54-combo TimeSeriesSplit(5) grid, RMSE scoring — returns best params."""
    fd = make_ml_features(y).dropna()
    tr = fd[fd.index < SPLIT]
    X, Y = tr[ML_FEATURES], tr['y']
    grid = list(itertools.product([100, 200, 300], [3, 4, 5], [0.03, 0.05, 0.1], [0.8, 1.0]))
    tscv = TimeSeriesSplit(n_splits=5)
    best, best_rmse = None, 1e18
    for n, d, lr, sub in grid:
        errs = []
        for ti, vi in tscv.split(X):
            m = xgb.XGBRegressor(n_estimators=n, max_depth=d, learning_rate=lr,
                                 subsample=sub, random_state=42, verbosity=0)
            m.fit(X.iloc[ti], Y.iloc[ti])
            errs.append(np.sqrt(np.mean((Y.iloc[vi].values - m.predict(X.iloc[vi])) ** 2)))
        cv = np.mean(errs)
        if cv < best_rmse:
            best_rmse, best = cv, dict(n_estimators=n, max_depth=d, learning_rate=lr, subsample=sub)
    return best


def hybrid_predict(train, sarima, sarima_fc, test_idx, params, resid_feats):
    resid = train - sarima.fittedvalues
    rd = pd.DataFrame({'r': resid})
    for lag in [1, 2, 12]:
        rd[f'lag{lag}'] = rd['r'].shift(lag)
    rd['month'] = rd.index.month
    rd = rd.dropna()
    xr = xgb.XGBRegressor(random_state=42, verbosity=0, **params)
    xr.fit(rd[resid_feats], rd['r'])
    allr, out = resid.copy(), []
    for i, d in enumerate(test_idx):
        feat = {'lag1': allr.iloc[-1], 'lag2': allr.iloc[-2], 'lag12': allr.iloc[-12], 'month': d.month}
        rp = xr.predict(pd.DataFrame([feat])[resid_feats])[0]
        out.append(sarima_fc.iloc[i] + rp)
        allr = pd.concat([allr, pd.Series([rp], index=[d])])
    return pd.Series(np.clip(out, 0, None), index=test_idx)


def run(y, test_actual, label):
    train, _ = train_test_split(y, SPLIT)
    n = len(test_actual)
    ti = test_actual.index
    rows = {}

    sn = pd.Series([y.iloc[y.index.get_loc(d) - 12] for d in ti], index=ti)
    rows['Seasonal Naive'] = metrics(test_actual, sn)

    sarima = fit_sarima(train)
    sfc = sarima.forecast(n)
    rows['SARIMA'] = metrics(test_actual, sfc)

    auto = pm.auto_arima(train, seasonal=True, m=12, stepwise=True,
                         suppress_warnings=True, error_action='ignore')
    rows[f'Auto-SARIMA {auto.order}{auto.seasonal_order}'] = metrics(
        test_actual, pd.Series(auto.predict(n), index=ti))

    ex = pd.DataFrame({'month': y.index.month, 'quarter': y.index.quarter,
                       't': np.arange(len(y))}, index=y.index)
    smx = SARIMAX(train, exog=ex[ex.index < SPLIT], order=(1, 1, 1),
                  seasonal_order=(1, 1, 1, 12), enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)
    rows['SARIMAX'] = metrics(test_actual, smx.forecast(n, exog=ex[ex.index >= SPLIT]))

    rows['XGBoost (default 200/4/0.05)'] = metrics(
        test_actual, xgb_predict(y, dict(n_estimators=200, max_depth=4, learning_rate=0.05)))

    best = tune_xgb(y)
    rows[f'XGBoost (tuned {best["n_estimators"]}/{best["max_depth"]}/{best["learning_rate"]}/{best["subsample"]})'] = \
        metrics(test_actual, xgb_predict(y, best))

    rows['Hybrid (default 50/2)'] = metrics(test_actual, hybrid_predict(
        train, sarima, sfc, ti, dict(n_estimators=50, max_depth=2, learning_rate=0.05),
        ['lag1', 'lag12', 'month']))

    rows['Hybrid (tuned 25/2)'] = metrics(test_actual, hybrid_predict(
        train, sarima, sfc, ti, dict(n_estimators=25, max_depth=2, learning_rate=0.05),
        ['lag1', 'lag2', 'lag12', 'month']))

    df = pd.DataFrame(rows).T
    df.insert(0, 'Version', label)
    return df


def main():
    out = os.path.join(os.path.dirname(__file__), 'results')
    y0, ys = get_monthly_total(DATA), get_monthly_total_synthetic(DATA)
    test_actual = y0[y0.index >= SPLIT]

    z = run(y0, test_actual, 'zero')
    s = run(ys, test_actual, 'synthetic')
    full = pd.concat([z, s])
    full.to_csv(os.path.join(out, 'full_matrix.csv'))

    pd.set_option('display.width', 170)
    print('\n===== ZERO (COVID = 0) =====')
    print(z.drop(columns='Version').to_string())
    print('\n===== SYNTHETIC (COVID imputed) =====')
    print(s.drop(columns='Version').to_string())
    print('\nSaved results/full_matrix.csv')


if __name__ == '__main__':
    main()
