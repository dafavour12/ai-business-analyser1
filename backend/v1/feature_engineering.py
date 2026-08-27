"""
STEP 3: Feature engineering.

We build three families of features:
  1. Calendar features   - month, quarter (captures seasonality)
  2. Lag features         - sales 1 and 2 months ago, for THIS SAME
                             category/region (captures momentum)
  3. Rolling features     - 3-month rolling average/std (captures trend
                             and volatility, smooths out one-off spikes)

Lags/rolling stats are computed per (Category, Region) group so we never
leak information across unrelated series (e.g. Furniture-Europe's history
should never inform Technology-Asia's lag features).
"""
import pandas as pd
import numpy as np

df = pd.read_csv('../data/processed/monthly_agg.csv', parse_dates=['month'])
df = df.sort_values(['Product_Category', 'Region', 'month']).reset_index(drop=True)

group_cols = ['Product_Category', 'Region']

# --- Calendar features ---
df['month_num'] = df['month'].dt.month
df['quarter'] = df['month'].dt.quarter
df['year'] = df['month'].dt.year
# months since start, gives the model a sense of overall time progression
df['time_idx'] = (df['year'] - df['year'].min()) * 12 + df['month_num']

# --- Lag features (per category/region series) ---
g = df.groupby(group_cols)['total_sales']
df['lag_1'] = g.shift(1)
df['lag_2'] = g.shift(2)

# --- Rolling features (per category/region series, using prior months only) ---
# shift(1) first so the rolling window never includes the current month
# (that would be leaking the answer into the features)
shifted = df.groupby(group_cols)['total_sales'].shift(1)
df['roll_mean_3'] = shifted.groupby([df['Product_Category'], df['Region']]).transform(lambda s: s.rolling(3, min_periods=1).mean())
df['roll_std_3'] = shifted.groupby([df['Product_Category'], df['Region']]).transform(lambda s: s.rolling(3, min_periods=1).std())

# --- Categorical encoding ---
df = pd.get_dummies(df, columns=['Product_Category', 'Region'], prefix=['cat', 'reg'])

# Drop rows where we don't have enough lag history yet (first 2 months of
# each series) - can't forecast without at least some history
before = len(df)
df = df.dropna(subset=['lag_1', 'lag_2']).reset_index(drop=True)
print(f"Dropped {before - len(df)} rows with insufficient lag history (first 2 months per series)")
print(f"Remaining rows: {len(df)}")

# fill any remaining NaN (e.g. roll_std_3 when only 1 prior point) with 0
df['roll_std_3'] = df['roll_std_3'].fillna(0)
df['avg_unit_price'] = df['avg_unit_price'].fillna(df['avg_unit_price'].median())
df['avg_discount'] = df['avg_discount'].fillna(0)

df.to_csv('../data/processed/features.csv', index=False)
print("\nSaved -> features.csv")
print(f"Feature columns: {[c for c in df.columns if c not in ['month','total_sales','total_profit']]}")