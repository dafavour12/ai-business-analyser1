import pandas as pd

# Load dataset
df = pd.read_csv("../data/raw/global_ecommerce_sales.csv")

df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["month"] = df["Order_Date"].dt.to_period("M")

print("=" * 60)
print("DATASET DIAGNOSIS")
print("=" * 60)

print(f"\nTotal orders: {len(df)}")
print(f"Date range: {df['Order_Date'].min()} to {df['Order_Date'].max()}")

# ---------------------------------------------------------
# 1. ORDERS PER CATEGORY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ORDERS PER CATEGORY")
print("=" * 60)

category_orders = (
    df.groupby("Product_Category")
    .agg(
        orders=("Order_ID", "count"),
        total_sales=("Total_Sales", "sum"),
        avg_sales=("Total_Sales", "mean"),
    )
    .sort_values("orders", ascending=False)
)

print(category_orders)


# ---------------------------------------------------------
# 2. ORDERS PER REGION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ORDERS PER REGION")
print("=" * 60)

region_orders = (
    df.groupby("Region")
    .agg(
        orders=("Order_ID", "count"),
        total_sales=("Total_Sales", "sum"),
        avg_sales=("Total_Sales", "mean"),
    )
    .sort_values("orders", ascending=False)
)

print(region_orders)


# ---------------------------------------------------------
# 3. ORDERS PER CATEGORY × REGION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ORDERS PER CATEGORY × REGION")
print("=" * 60)

combo_orders = (
    df.groupby(["Product_Category", "Region"])
    .agg(
        orders=("Order_ID", "count"),
        total_sales=("Total_Sales", "sum"),
        avg_sales=("Total_Sales", "mean"),
        max_sale=("Total_Sales", "max"),
    )
    .sort_values("orders", ascending=True)
)

print(combo_orders)


# ---------------------------------------------------------
# 4. MONTHLY CATEGORY × REGION SPIKES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("LARGEST MONTHLY SALES SPIKES")
print("=" * 60)

monthly = (
    df.groupby(["month", "Product_Category", "Region"])
    .agg(
        orders=("Order_ID", "count"),
        total_sales=("Total_Sales", "sum"),
    )
    .reset_index()
)

largest_spikes = monthly.sort_values(
    "total_sales",
    ascending=False
).head(20)

print(largest_spikes.to_string(index=False))


# ---------------------------------------------------------
# 5. ZERO-SALES MONTHS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ZERO-SALES MONTH ANALYSIS")
print("=" * 60)

zero_sales = monthly[monthly["total_sales"] == 0]

print(f"Zero-sales category-region months: {len(zero_sales)}")

print("\nZero-sales by category:")
print(
    zero_sales["Product_Category"]
    .value_counts()
)

print("\nZero-sales by region:")
print(
    zero_sales["Region"]
    .value_counts()
)

print("\nZero-sales combinations:")
print(
    zero_sales[
        ["month", "Product_Category", "Region", "orders", "total_sales"]
    ].to_string(index=False)
)


# ---------------------------------------------------------
# 6. HOW MANY MONTHS HAVE ORDERS?
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MONTH COVERAGE PER CATEGORY × REGION")
print("=" * 60)

coverage = (
    monthly.groupby(["Product_Category", "Region"])
    .agg(
        months=("month", "count"),
        months_with_orders=("orders", lambda x: (x > 0).sum()),
        zero_months=("orders", lambda x: (x == 0).sum()),
        total_sales=("total_sales", "sum"),
    )
    .reset_index()
)

coverage["coverage_percent"] = (
    coverage["months_with_orders"] / coverage["months"] * 100
)

print(
    coverage
    .sort_values("coverage_percent")
    .to_string(index=False)
)


print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)