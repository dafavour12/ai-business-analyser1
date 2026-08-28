import pandas as pd


def aggregate_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw ecommerce orders into a monthly
    category × region sales time series.

    Missing category/region/month combinations are NOT
    filled with zero.
    """

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    required_columns = [
        "Order_Date",
        "Product_Category",
        "Region",
        "Total_Sales",
        "Profit",
        "Order_ID",
        "Unit_Price",
        "Discount_Percent",
        "Quantity",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Work on a copy so we don't modify the original
    # uploaded DataFrame.
    df = df.copy()

    # Convert order date
    df["Order_Date"] = pd.to_datetime(
        df["Order_Date"],
        errors="coerce"
    )

    # Check for invalid dates
    if df["Order_Date"].isna().any():
        raise ValueError(
            "Some Order_Date values are invalid."
        )

    # Convert each order date to the first day
    # of its month.
    df["month"] = (
        df["Order_Date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # Monthly aggregation
    agg = (
        df.groupby(
            [
                "month",
                "Product_Category",
                "Region"
            ]
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

    return agg