# Sri Lanka Tourism Demand Forecasting — Final Year Project

**Student:** Reihan Imran | **ID:** CB016070 | **Supervisor:** Mr. Krishnamoorthy Caucidheesan
**Institution:** Asia Pacific Institute of Information Technology (APIIT) — Staffordshire University

Comparative evaluation of statistical, machine learning, and hybrid forecasting models on SLTDA monthly tourist arrival data (January 2018 – March 2026).

---

## PROJECT STRUCTURE

```
FYP_Tourism_Forecasting/
├── data/
│   ├── Tourism_MOM_Dataset.csv          <- raw SLTDA dataset (217 countries)
│   └── monthly_total_arrivals.csv       <- aggregated monthly totals
├── notebooks/                           <- run in order (00 → 12)
│   ├── 00_data_preprocessing.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_sarima_models.ipynb
│   ├── 04_xgboost_model.ipynb
│   ├── 05_hybrid_sarima_xgboost.ipynb
│   ├── 06_walkforward_validation.ipynb
│   ├── 07_residual_analysis.ipynb
│   ├── 08_final_summary.ipynb
│   ├── 09_hyperparameter_tuning.ipynb
│   ├── 10_diagnostics_and_improvements.ipynb
│   ├── 11_hybrid_tuning.ipynb
│   └── 12_covid_handling_and_final_tuning.ipynb
├── src/                                 <- shared helper modules
│   ├── data_loader.py
│   ├── models.py
│   └── evaluation.py
├── app/
│   ├── streamlit_app.py                 <- interactive dashboard
│   ├── Tourism_MOM_Dataset.csv
│   ├── model_results.csv
│   ├── future_forecast.csv
│   ├── walkforward_results.csv
│   └── walkforward_summary.csv
├── results/                             <- charts and CSVs (generated on run)
├── regenerate_results.py                <- rebuild all CSVs in one command
└── requirements.txt
```

---

## PART A — INSTALLATION (do once)

### Step 1: Install Anaconda
Download from https://www.anaconda.com/download and install it.

### Step 2: Open Terminal (Mac) or Anaconda Prompt (Windows)

### Step 3: Navigate to the project folder
```
cd /path/to/FYP_Tourism_Forecasting
```

### Step 4: Install dependencies
```
pip install -r requirements.txt
```
If `pmdarima` fails:
```
pip install pmdarima --no-build-isolation
```

---

## PART B — GENERATE RESULTS

### Option A: Run all notebooks in order (full pipeline, ~1–2 hours)
```
jupyter notebook
```
Open each notebook and run all cells (Cell → Run All) in order: 00 → 12.

### Option B: Rebuild results CSVs only (fast, ~5 minutes)
```
python regenerate_results.py
```
Regenerates all CSVs and the master summary chart used by the dashboard and thesis.

---

## PART C — LAUNCH THE DASHBOARD

**Important:** run from inside the `app/` directory so relative file paths resolve correctly.
```
cd app
streamlit run streamlit_app.py
```
Opens at http://localhost:8501. Press Ctrl+C to stop.

---

## RESULTS SUMMARY

Single-holdout evaluation on January 2025 – March 2026 (15 months):

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| **Auto-SARIMA** | **27,109** | **38,139** | **12.36% (BEST)** |
| Seasonal Naive | 27,869 | 29,655 | 14.19% |
| XGBoost | 31,507 | 38,651 | 16.63% |
| SARIMA | 42,083 | 48,964 | 22.89% |
| SARIMAX | 42,083 | 48,964 | 22.89% |
| SARIMA-XGBoost Hybrid | 44,935 | 51,957 | 24.43% |

**Best model: Auto-SARIMA** with MAPE of 12.36% on the holdout test set.

Walk-forward validation (rolling 3-month windows, min 48-month training window):

| Model | Mean MAPE | Median MAPE | Windows Won |
|-------|-----------|-------------|-------------|
| SARIMA-XGBoost Hybrid | 24.05% | 13.44% | 6 |
| SARIMA | 25.09% | 14.02% | 6 |
| XGBoost | 27.47% | 18.51% | 4 |

Note: A Diebold-Mariano test (DM = −0.890, p = 0.389) confirms XGBoost and SARIMA are statistically equivalent in accuracy on the holdout set; differences are not significant at the 5% level.

---

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: Tourism_MOM_Dataset.csv` | Run `streamlit run` from inside the `app/` directory, not the project root |
| `pip not recognized` | Reinstall Python, tick "Add to PATH" |
| `module not found` | Run `pip install -r requirements.txt` again |
| `pmdarima` won't install | `pip install pmdarima --no-build-isolation` |
| Notebook errors | Make sure you run them in order starting from 00 |
| Streamlit blank screen | Navigate to http://localhost:8501 manually |
