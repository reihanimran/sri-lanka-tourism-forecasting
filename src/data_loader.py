"""
data_loader.py
Helper functions to load and prepare the SLTDA tourism dataset.
"""
import pandas as pd
import numpy as np


def load_raw(path='../data/Tourism_MOM_Dataset.csv'):
    """Load the raw country-level dataset."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['Arrivals'] = pd.to_numeric(
        df['Arrivals'].astype(str).str.replace(',', '', regex=False),
        errors='coerce')  # strip thousand-separator commas before parsing
    df['Date'] = pd.to_datetime(df['Date'])
    df['Country'] = df['Country'].str.strip()
    return df


def get_monthly_total(path='../data/Tourism_MOM_Dataset.csv'):
    """
    Aggregate all countries into a single monthly total series.
    Returns a pandas Series indexed by month-end date.
    COVID border-closure months (Apr-Nov 2020) are kept as 0.
    """
    df = load_raw(path)
    monthly = df.groupby('Date')['Arrivals'].sum()
    monthly = monthly[monthly.index <= '2026-03-31']  # keep only real data
    monthly = monthly.asfreq('ME')
    monthly = monthly.fillna(0)  # COVID closure = 0 arrivals
    return monthly


def get_monthly_total_synthetic(path='../data/Tourism_MOM_Dataset.csv'):
    """
    Aggregate all countries into a single monthly total series, with the
    COVID-19 border closure months (Apr-Nov 2020) imputed using the seasonal
    average of the corresponding 2018 and 2019 months.

    Justification: the closure was an external policy intervention, not a
    reflection of underlying tourism demand. Retaining the closure months as
    zero biases the seasonal parameter estimates and destabilises model
    training. Imputing them reconstructs the demand level that would have
    occurred under normal conditions.
    """
    df = load_raw(path)
    monthly = df.groupby('Date')['Arrivals'].sum()
    monthly = monthly[monthly.index <= '2026-03-31']
    monthly = monthly.asfreq('ME')
    monthly = monthly.fillna(0)

    # Impute COVID closure months with 2018-2019 seasonal average
    covid_months = pd.date_range('2020-04-30', '2020-11-30', freq='ME')
    for d in covid_months:
        m = d.month
        v2018 = monthly[(monthly.index.year == 2018) & (monthly.index.month == m)].values[0]
        v2019 = monthly[(monthly.index.year == 2019) & (monthly.index.month == m)].values[0]
        monthly.loc[d] = (v2018 + v2019) / 2.0
    return monthly


def train_test_split(y, split_date='2025-01-01'):
    """Split a time series into train and test by date."""
    train = y[y.index < split_date]
    test = y[y.index >= split_date]
    return train, test
