"""
run_comparison.py
-----------------
Clean A/B comparison of all six forecasting models on TWO data versions:

  1. ZERO      -> COVID closure months (Apr-Nov 2020) kept as 0
  2. SYNTHETIC -> COVID months imputed with 2018-2019 seasonal average

The ONLY thing that differs between the two runs is the COVID handling.
Everything else is held identical:
  - same train/test split (train < 2025-01, test = 2025-01 .. 2026-03)
  - same six models with the same (final/tuned) hyper-parameters
  - same three metrics (MAE, RMSE, MAPE) via src/evaluation.evaluate

Outputs (does NOT overwrite the existing result files):
  results/comparison_zero.csv
  results/comparison_synthetic.csv
  results/comparison_combined.csv
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import xgboost as xgb

from data_loader import get_monthly_total, get_monthly_total_synthetic, train_test_split
from evaluation import evaluate
from models import make_ml_features, ML_FEATURES

DATA = os.path.join(os.path.dirname(__file__), 'data', 'Tourism_MOM_Dataset.csv')
SPLIT = '2025-01-01'

# Final/tuned hyper-parameters (identical for both data versions) ------------
XGB_PARAMS    = dict(n_estimators=100, max_depth=3, learning_rate=0.03,
                     subsample=0.8, random_state=42, verbosity=0)
HYB_PARAMS    = dict(n_estimators=25, max_depth=2, learning_rate=0.05,
                     random_state=42, verbosity=0)
RESID_FEATS   = ['lag1', 'lag2', 'lag12', 'month']   # tuned hybrid residual feats


def run_all_models(y, test_actual):
    """Run the six models on series `y`; evaluate against the real test_actual."""
    train, _ = train_test_split(y, SPLIT)
    n_test = len(test_actual)
    rows = []

    # 1. Seasonal Naive: same month one year earlier ------------------------
    sn = pd.Series(
        [y.iloc[y.index.get_loc(d) - 12] for d in test_actual.index],
        index=test_actual.index)
    rows.append(evaluate(test_actual, sn, 'Seasonal Naive', verbose=False))

    # 2. SARIMA(1,1,1)(1,1,1,12) -------------------------------------------
    sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_fc = sarima.forecast(n_test)
    rows.append(evaluate(test_actual, sarima_fc, 'SARIMA', verbose=False))

    # 3. Auto-SARIMA (pmdarima order search) -------------------------------
    auto = pm.auto_arima(train, seasonal=True, m=12, stepwise=True,
                         suppress_warnings=True, error_action='ignore')
    auto_fc = pd.Series(auto.predict(n_test), index=test_actual.index)
    rows.append(evaluate(test_actual, auto_fc, 'Auto-SARIMA', verbose=False))

    # 4. SARIMAX (month/quarter/time-index exogenous) ----------------------
    ex = pd.DataFrame({'month': y.index.month, 'quarter': y.index.quarter,
                       't': np.arange(len(y))}, index=y.index)
    ex_tr = ex[ex.index < SPLIT]
    ex_te = ex[ex.index >= SPLIT]
    sarimax = SARIMAX(train, exog=ex_tr, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarimax_fc = sarimax.forecast(n_test, exog=ex_te)
    rows.append(evaluate(test_actual, sarimax_fc, 'SARIMAX', verbose=False))

    # 5. XGBoost (tuned) — one-step features from true lags ----------------
    fd = make_ml_features(y).dropna()
    tr = fd[fd.index < SPLIT]
    te = fd[fd.index >= SPLIT]
    xm = xgb.XGBRegressor(**XGB_PARAMS)
    xm.fit(tr[ML_FEATURES], tr['y'])
    xgb_fc = pd.Series(xm.predict(te[ML_FEATURES]), index=te.index)
    rows.append(evaluate(test_actual, xgb_fc.reindex(test_actual.index), 'XGBoost (tuned)', verbose=False))

    # 6. SARIMA-XGBoost Hybrid (tuned) — XGBoost corrects SARIMA residuals -
    resid = train - sarima.fittedvalues
    rd = pd.DataFrame({'r': resid})
    rd['lag1'] = rd['r'].shift(1)
    rd['lag2'] = rd['r'].shift(2)
    rd['lag12'] = rd['r'].shift(12)
    rd['month'] = rd.index.month
    rd = rd.dropna()
    xr = xgb.XGBRegressor(**HYB_PARAMS)
    xr.fit(rd[RESID_FEATS], rd['r'])

    all_resid = resid.copy()
    hyb = []
    for i, d in enumerate(test_actual.index):
        feat = {'lag1': all_resid.iloc[-1], 'lag2': all_resid.iloc[-2],
                'lag12': all_resid.iloc[-12], 'month': d.month}
        rp = xr.predict(pd.DataFrame([feat])[RESID_FEATS])[0]
        hyb.append(sarima_fc.iloc[i] + rp)
        all_resid = pd.concat([all_resid, pd.Series([rp], index=[d])])
    hyb = pd.Series(np.clip(hyb, 0, None), index=test_actual.index)
    rows.append(evaluate(test_actual, hyb, 'SARIMA-XGBoost Hybrid (tuned)', verbose=False))

    return pd.DataFrame(rows)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'results')

    y_zero = get_monthly_total(DATA)
    y_syn  = get_monthly_total_synthetic(DATA)

    # Real (true) test values — identical for both, COVID is in the train span
    test_actual = y_zero[y_zero.index >= SPLIT]

    print('Test period:', test_actual.index.min().date(), '->',
          test_actual.index.max().date(), f'({len(test_actual)} months)\n')

    res_zero = run_all_models(y_zero, test_actual).sort_values('MAPE').reset_index(drop=True)
    res_syn  = run_all_models(y_syn,  test_actual).sort_values('MAPE').reset_index(drop=True)

    res_zero.to_csv(os.path.join(out_dir, 'comparison_zero.csv'), index=False)
    res_syn.to_csv(os.path.join(out_dir, 'comparison_synthetic.csv'), index=False)

    # Side-by-side combined table -----------------------------------------
    z = res_zero.set_index('Model').add_suffix('_zero')
    s = res_syn.set_index('Model').add_suffix('_syn')
    combined = z.join(s)
    combined['MAPE_improvement_pp'] = (combined['MAPE_zero'] - combined['MAPE_syn']).round(2)
    combined = combined.sort_values('MAPE_syn')
    combined.to_csv(os.path.join(out_dir, 'comparison_combined.csv'))

    pd.set_option('display.width', 160, 'display.max_columns', 20)
    print('=' * 70)
    print('VERSION 1 — COVID AS ZERO')
    print('=' * 70)
    print(res_zero.to_string(index=False))
    print('\n' + '=' * 70)
    print('VERSION 2 — SYNTHETIC COVID IMPUTATION')
    print('=' * 70)
    print(res_syn.to_string(index=False))
    print('\n' + '=' * 70)
    print('SIDE-BY-SIDE (MAPE, lower = better)')
    print('=' * 70)
    show = combined[['MAPE_zero', 'MAPE_syn', 'MAPE_improvement_pp']]
    print(show.to_string())
    print('\nBest (zero):     ', res_zero.iloc[0]['Model'], f"{res_zero.iloc[0]['MAPE']}%")
    print('Best (synthetic):', res_syn.iloc[0]['Model'], f"{res_syn.iloc[0]['MAPE']}%")
    print('\nSaved: comparison_zero.csv, comparison_synthetic.csv, comparison_combined.csv')


if __name__ == '__main__':
    main()
