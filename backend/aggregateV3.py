"""
STEP 1-2 V3: Aggregate raw orders into a monthly sales time series.

V3 CHANGE:
We do NOT create a complete category × region × month grid
and fill missing combinations with zero.

Why?
A missing combination does not necessarily mean zero demand.
It may simply mean that no transaction was recorded for that
category/region during that month.

We therefore preserve only months that actually appear in the
dataset.
"""

import pandas as pd
import numpy as np


# ============================================================
# STEP 1: LOAD RAW DATA
# ============================================================

df = pd.read_csv("../data/raw/global_ecommerce_sales.csv")

df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Convert each order date to the first day of its month
df["month"] = (
    df["Order_Date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)


# ============================================================
# STEP 2: MONTHLY AGGREGATION
# ============================================================

agg = (
    df.groupby(
        ["month", "Product_Category", "Region"]
    )
    .agg(
        total_sales=("Total_Sales", "sum"),
        total_profit=("Profit", "sum"),
        order_count=("Order_ID", "count"),
        avg_unit_price=("Unit_Price", "mean"),
        avg_discount=("Discount_Percent", "mean"),
        total_quantity=("Quantity", "sum"),
    )
    .reset_index()
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("=" * 60)
print("V3 MONTHLY AGGREGATION")
print("=" * 60)

print(f"\nRaw orders: {len(df)}")

print(
    f"Aggregated rows: {len(agg)}"
)

print(
    f"Date range: "
    f"{agg['month'].min()} to "
    f"{agg['month'].max()}"
)

print(
    f"\nCategories: "
    f"{df['Product_Category'].unique().tolist()}"
)

print(
    f"\nRegions: "
    f"{df['Region'].unique().tolist()}"
)


# ============================================================
# CHECK FOR REAL ZERO-SALES MONTHS
# ============================================================

zero_sales = agg[agg["total_sales"] == 0]

print("\n" + "=" * 60)
print("ZERO-SALES CHECK")
print("=" * 60)

print(
    f"Actual zero-sales observations: {len(zero_sales)}"
)

if len(zero_sales) > 0:
    print("\nZero-sales observations:")
    print(zero_sales.to_string(index=False))
else:
    print("No actual zero-sales observations found.")


# ============================================================
# CHECK CATEGORY × REGION COVERAGE
# ============================================================

coverage = (
    agg.groupby(
        ["Product_Category", "Region"]
    )
    .agg(
        months_observed=("month", "count"),
        total_sales=("total_sales", "sum"),
        total_orders=("order_count", "sum"),
    )
    .reset_index()
    .sort_values("months_observed")
)

print("\n" + "=" * 60)
print("CATEGORY × REGION COVERAGE")
print("=" * 60)

print(
    coverage.to_string(index=False)
)


# ============================================================
# CHECK FOR MISSING MONTHS
# ============================================================

print("\n" + "=" * 60)
print("MISSING MONTH CHECK")
print("=" * 60)

for (category, region), group in agg.groupby(
    ["Product_Category", "Region"]
):

    months = pd.date_range(
        group["month"].min(),
        group["month"].max(),
        freq="MS"
    )

    observed = set(group["month"])

    missing = [
        month
        for month in months
        if month not in observed
    ]

    print(
        f"{category} | {region}: "
        f"{len(missing)} missing month(s)"
    )


# ============================================================
# SAVE
# ============================================================

output_path = "../data/processed/monthly_agg_v3.csv"

agg.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("V3 AGGREGATION COMPLETE")
print("=" * 60)

print(f"Saved -> {output_path}")
print(f"Rows: {len(agg)}")
print(f"Columns: {len(agg.columns)}")

print("\nFirst 10 rows:")
print(agg.head(10).to_string(index=False))
