"""
STEP 2 V3: Feature Engineering

V3 improvements:
1. Uses the corrected monthly aggregation.
2. Does NOT create artificial zero-sales months.
3. Uses calendar-aware lag features.
4. Only creates a lag when the required month actually exists.
5. Rolling features use previous observed months only.
"""

import pandas as pd
import numpy as np


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "../data/processed/monthly_agg_v3.csv",
    parse_dates=["month"]
)

df = df.sort_values(
    ["Product_Category", "Region", "month"]
).reset_index(drop=True)

group_cols = ["Product_Category", "Region"]


# ============================================================
# CALENDAR FEATURES
# ============================================================

df["month_num"] = df["month"].dt.month
df["quarter"] = df["month"].dt.quarter
df["year"] = df["month"].dt.year

df["time_idx"] = (
    (df["year"] - df["year"].min()) * 12
    + df["month_num"]
)


# ============================================================
# CALENDAR-AWARE LAGS
# ============================================================

# Create a lookup table for each category/region/month.
lookup = df.set_index(
    ["Product_Category", "Region", "month"]
)["total_sales"]


def get_lag(row, months):
    """
    Return sales from exactly N calendar months earlier.

    If that month does not exist in the dataset, return NaN.
    """

    target_month = row["month"] - pd.DateOffset(months=months)

    key = (
        row["Product_Category"],
        row["Region"],
        target_month
    )

    try:
        return lookup.loc[key]
    except KeyError:
        return np.nan


df["lag_1"] = df.apply(
    lambda row: get_lag(row, 1),
    axis=1
)

df["lag_2"] = df.apply(
    lambda row: get_lag(row, 2),
    axis=1
)

df["lag_3"] = df.apply(
    lambda row: get_lag(row, 3),
    axis=1
)

df["lag_6"] = df.apply(
    lambda row: get_lag(row, 6),
    axis=1
)

df["lag_12"] = df.apply(
    lambda row: get_lag(row, 12),
    axis=1
)


# ============================================================
# ROLLING FEATURES
# ============================================================

# IMPORTANT:
# shift(1) ensures the current month's sales are never included.

df["roll_mean_3"] = (
    df.groupby(group_cols)["total_sales"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)

df["roll_std_3"] = (
    df.groupby(group_cols)["total_sales"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=3,
            min_periods=1
        ).std()
    )
)

df["roll_mean_6"] = (
    df.groupby(group_cols)["total_sales"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=6,
            min_periods=1
        ).mean()
    )
)

df["roll_std_6"] = (
    df.groupby(group_cols)["total_sales"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=6,
            min_periods=1
        ).std()
    )
)

df["roll_mean_12"] = (
    df.groupby(group_cols)["total_sales"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=12,
            min_periods=1
        ).mean()
    )
)

df["roll_std_12"] = (
    df.groupby(group_cols)["total_sales"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=12,
            min_periods=1
        ).std()
    )
)


# ============================================================
# ROLLING MAX
# ============================================================

df["roll_max_3"] = (
    df.groupby(group_cols)["total_sales"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=3,
            min_periods=1
        ).max()
    )
)

df["roll_max_6"] = (
    df.groupby(group_cols)["total_sales"]
    .transform(
        lambda s: s.shift(1).rolling(
            window=6,
            min_periods=1
        ).max()
    )
)


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

df = pd.get_dummies(
    df,
    columns=["Product_Category", "Region"],
    prefix=["cat", "reg"]
)


# ============================================================
# REMOVE ROWS WITHOUT REQUIRED HISTORY
# ============================================================

before = len(df)

# We need lag_1, lag_2 and lag_3.
df = df.dropna(
    subset=["lag_1", "lag_2", "lag_3"]
).reset_index(drop=True)

print(
    f"Dropped {before - len(df)} rows "
    f"without enough historical data"
)

print(f"Remaining rows: {len(df)}")


# ============================================================
# HANDLE REMAINING NaN VALUES
# ============================================================

rolling_cols = [
    "roll_std_3",
    "roll_std_6",
    "roll_std_12"
]

for col in rolling_cols:
    df[col] = df[col].fillna(0)


# ============================================================
# REMOVE LEAKAGE-PRONE CURRENT-MONTH FEATURES
# ============================================================

# These describe the same month we're trying to predict.
# They will NOT be used during training.

leaky_cols = [
    "order_count",
    "avg_unit_price",
    "avg_discount",
    "total_quantity"
]


# ============================================================
# SAVE
# ============================================================

output_path = "../data/processed/features_v3.csv"

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING V3 COMPLETE")
print("=" * 60)

print(f"Saved -> {output_path}")

print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nNew V3 features:")
print("  ✓ Calendar-aware lag_1")
print("  ✓ Calendar-aware lag_2")
print("  ✓ Calendar-aware lag_3")
print("  ✓ Calendar-aware lag_6")
print("  ✓ Calendar-aware lag_12")
print("  ✓ roll_mean_3")
print("  ✓ roll_std_3")
print("  ✓ roll_mean_6")
print("  ✓ roll_std_6")
print("  ✓ roll_mean_12")
print("  ✓ roll_std_12")
print("  ✓ roll_max_3")
print("  ✓ roll_max_6")

print("\nFeature engineering finished.")