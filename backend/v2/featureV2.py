"""
STEP 7: Feature Engineering V2

V2 keeps all features from V1 and adds stronger historical features
to help the forecasting model recognize:

1. Short-term momentum
   - lag_1
   - lag_2
   - lag_3

2. Medium-term behavior
   - lag_6
   - rolling 6-month statistics

3. Year-over-year behavior
   - lag_12
   - rolling 12-month statistics

4. Seasonality
   - month
   - quarter
   - year
   - time index

All lag and rolling features are calculated separately for each
Product_Category × Region series.

IMPORTANT:
All lag and rolling features use ONLY previous months.
The current month's sales are never included in the features.
"""

import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD MONTHLY AGGREGATED DATA
# ============================================================

df = pd.read_csv(
    '../data/processed/monthly_agg.csv',
    parse_dates=['month']
)

df = df.sort_values(
    ['Product_Category', 'Region', 'month']
).reset_index(drop=True)

group_cols = ['Product_Category', 'Region']


# ============================================================
# 2. CALENDAR FEATURES
# ============================================================

df['month_num'] = df['month'].dt.month
df['quarter'] = df['month'].dt.quarter
df['year'] = df['month'].dt.year

# Number of months since the beginning of the dataset
df['time_idx'] = (
    (df['year'] - df['year'].min()) * 12
    + df['month_num']
    - 1
)


# ============================================================
# 3. LAG FEATURES
# ============================================================

g = df.groupby(group_cols)['total_sales']

# Short-term history
df['lag_1'] = g.shift(1)
df['lag_2'] = g.shift(2)
df['lag_3'] = g.shift(3)

# Medium-term history
df['lag_6'] = g.shift(6)

# Same month from previous year
df['lag_12'] = g.shift(12)


# ============================================================
# 4. ROLLING FEATURES
# ============================================================

# Shift first so the current month's sales are NEVER included.
shifted = g.shift(1)

shifted_group = shifted.groupby(
    [df['Product_Category'], df['Region']]
)

# 3-month rolling statistics
df['roll_mean_3'] = shifted_group.transform(
    lambda s: s.rolling(
        window=3,
        min_periods=1
    ).mean()
)

df['roll_std_3'] = shifted_group.transform(
    lambda s: s.rolling(
        window=3,
        min_periods=1
    ).std()
)

# 6-month rolling statistics
df['roll_mean_6'] = shifted_group.transform(
    lambda s: s.rolling(
        window=6,
        min_periods=1
    ).mean()
)

df['roll_std_6'] = shifted_group.transform(
    lambda s: s.rolling(
        window=6,
        min_periods=1
    ).std()
)

# 12-month rolling statistics
df['roll_mean_12'] = shifted_group.transform(
    lambda s: s.rolling(
        window=12,
        min_periods=1
    ).mean()
)

df['roll_std_12'] = shifted_group.transform(
    lambda s: s.rolling(
        window=12,
        min_periods=1
    ).std()
)

# Recent maximum sales
df['roll_max_3'] = shifted_group.transform(
    lambda s: s.rolling(
        window=3,
        min_periods=1
    ).max()
)

df['roll_max_6'] = shifted_group.transform(
    lambda s: s.rolling(
        window=6,
        min_periods=1
    ).max()
)


# ============================================================
# 5. CATEGORICAL ENCODING
# ============================================================

df = pd.get_dummies(
    df,
    columns=['Product_Category', 'Region'],
    prefix=['cat', 'reg']
)


# ============================================================
# 6. REMOVE ROWS WITHOUT ENOUGH HISTORY
# ============================================================

before = len(df)

# lag_12 requires 12 previous months.
df = df.dropna(
    subset=[
        'lag_1',
        'lag_2',
        'lag_3',
        'lag_6',
        'lag_12'
    ]
).reset_index(drop=True)

print(
    f"Dropped {before - len(df)} rows "
    f"without enough historical data"
)

print(f"Remaining rows: {len(df)}")


# ============================================================
# 7. HANDLE REMAINING MISSING VALUES
# ============================================================

rolling_std_cols = [
    'roll_std_3',
    'roll_std_6',
    'roll_std_12'
]

for col in rolling_std_cols:
    df[col] = df[col].fillna(0)


# These describe the current month, so we keep them available
# for analysis but they will be excluded from model training.
df['avg_unit_price'] = df['avg_unit_price'].fillna(
    df['avg_unit_price'].median()
)

df['avg_discount'] = df['avg_discount'].fillna(0)


# ============================================================
# 8. SAVE V2 FEATURES
# ============================================================

output_path = '../data/processed/features_v2.csv'

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 9. REPORT
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING V2 COMPLETE")
print("=" * 60)

print(f"Saved -> {output_path}")

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nNew V2 features:")

new_features = [
    'lag_3',
    'lag_6',
    'lag_12',
    'roll_mean_6',
    'roll_std_6',
    'roll_mean_12',
    'roll_std_12',
    'roll_max_3',
    'roll_max_6'
]

for feature in new_features:
    print(f"  ✓ {feature}")

print("\nFeature columns:")

excluded = [
    'month',
    'total_sales',
    'total_profit'
]

feature_columns = [
    col for col in df.columns
    if col not in excluded
]

print(feature_columns)