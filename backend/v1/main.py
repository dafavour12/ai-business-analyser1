"""
STEP 1-2: Load raw orders and aggregate into a monthly demand time series
per (Product_Category x Region). This is the grain our forecast will run at.

Why aggregate instead of forecasting individual orders?
Individual orders are noisy and sparse (a handful per day). Aggregating to
monthly totals per category/region gives us a stable signal to forecast,
and it matches how a business actually plans (e.g. "how much Furniture
demand in Europe next month?").
"""
import pandas as pd
import numpy as np

df = pd.read_csv('../data/raw/global_ecommerce_sales.csv')
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['month'] = df['Order_Date'].dt.to_period('M').dt.to_timestamp()

# Aggregate to monthly Category x Region grain
agg = (
    df.groupby(['month', 'Product_Category', 'Region'])
    .agg(
        total_sales=('Total_Sales', 'sum'),
        total_profit=('Profit', 'sum'),
        order_count=('Order_ID', 'count'),
        avg_unit_price=('Unit_Price', 'mean'),
        avg_discount=('Discount_Percent', 'mean'),
        total_quantity=('Quantity', 'sum'),
    )
    .reset_index()
)

print(f"Raw orders: {len(df)}")
print(f"Aggregated rows (month x category x region combos): {len(agg)}")
print(f"Date range: {agg['month'].min()} to {agg['month'].max()}")
print(f"\nCategories: {df['Product_Category'].unique().tolist()}")
print(f"Regions: {df['Region'].unique().tolist()}")

# IMPORTANT: not every category x region combo has an order in every month
# (2000 orders spread over 36 months x 4 categories x 5 regions = 720 cells,
# ~2.8 orders/cell on average -> many months will have zero orders for a
# given combo). We need to fill those gaps with 0 rather than silently
# dropping them, or the model will never learn what "no demand" looks like.
all_months = pd.date_range(agg['month'].min(), agg['month'].max(), freq='MS')
all_cats = df['Product_Category'].unique()
all_regions = df['Region'].unique()

full_index = pd.MultiIndex.from_product(
    [all_months, all_cats, all_regions],
    names=['month', 'Product_Category', 'Region']
)
full = pd.DataFrame(index=full_index).reset_index()

merged = full.merge(agg, on=['month', 'Product_Category', 'Region'], how='left')
fill_cols = ['total_sales', 'total_profit', 'order_count', 'total_quantity']
merged[fill_cols] = merged[fill_cols].fillna(0)
# avg_unit_price / avg_discount: leave NaN where no orders occurred, we'll
# handle via forward-fill within group during feature engineering
merged = merged.sort_values(['Product_Category', 'Region', 'month']).reset_index(drop=True)

print(f"\nFull grid rows (should be {len(all_months)}mo x {len(all_cats)}cat x {len(all_regions)}reg = {len(all_months)*len(all_cats)*len(all_regions)}): {len(merged)}")
print(f"Rows with zero orders in that month/combo: {(merged['order_count']==0).sum()} ({(merged['order_count']==0).mean():.1%})")

merged.to_csv('../data/processed/monthly_agg.csv', index=False)
print("\nSaved -> monthly_agg.csv")
print(merged.head(10))

print("Raw orders:", len(df))
print("Aggregated rows:", len(agg))
print("Final rows:", len(merged))

print("\nDate range:")
print(merged["month"].min(), "to", merged["month"].max())

print("\nCategories:")
print(merged["Product_Category"].unique())

print("\nRegions:")
print(merged["Region"].unique())

print("\nZero-order rows:")
print((merged["order_count"] == 0).sum())