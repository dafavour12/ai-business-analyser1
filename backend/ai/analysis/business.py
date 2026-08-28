from pathlib import Path
import pandas as pd
import json


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = DATA_DIR / "monthly_agg_v3.csv"
OUTPUT_FILE = DATA_DIR / "business_analysis.json"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["month"] = pd.to_datetime(df["month"])

    return df


# ============================================================
# BASIC BUSINESS SUMMARY
# ============================================================

def get_basic_summary(df):

    total_sales = df["total_sales"].sum()

    total_orders = df["order_count"].sum()

    months = df["month"].nunique()

    average_monthly_sales = (
        total_sales / months
        if months > 0
        else 0
    )

    return {
        "total_sales": round(float(total_sales), 2),
        "total_orders": int(total_orders),
        "months_analyzed": int(months),
        "average_monthly_sales": round(
            float(average_monthly_sales), 2
        )
    }


# ============================================================
# CATEGORY ANALYSIS
# ============================================================

def analyze_categories(df):

    category = (
        df.groupby("Product_Category")
        .agg(
            total_sales=("total_sales", "sum"),
            total_orders=("order_count", "sum")
        )
        .reset_index()
    )

    category["average_order_value"] = (
        category["total_sales"] /
        category["total_orders"]
    )

    category = category.sort_values(
        "total_sales",
        ascending=False
    )

    result = category.to_dict(orient="records")

    for row in result:
        row["total_sales"] = round(
            float(row["total_sales"]), 2
        )

        row["total_orders"] = int(
            row["total_orders"]
        )

        row["average_order_value"] = round(
            float(row["average_order_value"]), 2
        )

    return {
        "ranking": result,
        "best_category": result[0] if result else None,
        "worst_category": result[-1] if result else None
    }


# ============================================================
# REGION ANALYSIS
# ============================================================

def analyze_regions(df):

    region = (
        df.groupby("Region")
        .agg(
            total_sales=("total_sales", "sum"),
            total_orders=("order_count", "sum")
        )
        .reset_index()
    )

    region["average_order_value"] = (
        region["total_sales"] /
        region["total_orders"]
    )

    region = region.sort_values(
        "total_sales",
        ascending=False
    )

    result = region.to_dict(orient="records")

    for row in result:
        row["total_sales"] = round(
            float(row["total_sales"]), 2
        )

        row["total_orders"] = int(
            row["total_orders"]
        )

        row["average_order_value"] = round(
            float(row["average_order_value"]), 2
        )

    return {
        "ranking": result,
        "best_region": result[0] if result else None,
        "worst_region": result[-1] if result else None
    }


# ============================================================
# MONTHLY TREND
# ============================================================

def analyze_monthly_trend(df):

    monthly = (
        df.groupby("month")
        .agg(
            total_sales=("total_sales", "sum"),
            total_orders=("order_count", "sum")
        )
        .reset_index()
        .sort_values("month")
    )

    monthly["month"] = (
        monthly["month"]
        .dt.strftime("%Y-%m")
    )

    result = monthly.to_dict(orient="records")

    for row in result:
        row["total_sales"] = round(
            float(row["total_sales"]), 2
        )

        row["total_orders"] = int(
            row["total_orders"]
        )

    return result


# ============================================================
# TOP SALES SPIKES
# ============================================================

def analyze_spikes(df, limit=10):

    spikes = (
        df[
            [
                "month",
                "Product_Category",
                "Region",
                "total_sales",
                "order_count"
            ]
        ]
        .sort_values(
            "total_sales",
            ascending=False
        )
        .head(limit)
        .copy()
    )

    spikes["month"] = (
        spikes["month"]
        .dt.strftime("%Y-%m")
    )

    result = spikes.to_dict(orient="records")

    for row in result:
        row["total_sales"] = round(
            float(row["total_sales"]), 2
        )

        row["order_count"] = int(
            row["order_count"]
        )

    return result


# ============================================================
# CATEGORY × REGION ANALYSIS
# ============================================================

def analyze_category_region(df):

    result = (
        df.groupby(
            [
                "Product_Category",
                "Region"
            ]
        )
        .agg(
            total_sales=("total_sales", "sum"),
            total_orders=("order_count", "sum")
        )
        .reset_index()
        .sort_values(
            "total_sales",
            ascending=False
        )
    )

    records = result.to_dict(
        orient="records"
    )

    for row in records:

        row["total_sales"] = round(
            float(row["total_sales"]), 2
        )

        row["total_orders"] = int(
            row["total_orders"]
        )

    return records


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def generate_analysis(df):

    if df.empty:
        raise ValueError(
            "Monthly aggregated data is empty."
        )

    df = df.copy()

    df["month"] = pd.to_datetime(df["month"])

    analysis = {

        "basic_summary":
            get_basic_summary(df),

        "category_analysis":
            analyze_categories(df),

        "region_analysis":
            analyze_regions(df),

        "monthly_trend":
            analyze_monthly_trend(df),

        "sales_spikes":
            analyze_spikes(df),

        "category_region_analysis":
            analyze_category_region(df)
    }

    return analysis


# ============================================================
# SAVE ANALYSIS
# ============================================================

def save_analysis(analysis):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            analysis,
            file,
            indent=4
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BUSINESS ANALYSIS")
    print("=" * 60)

    df = load_data()

    analysis = generate_analysis(df)

    save_analysis(analysis)

    print()
    print("BUSINESS SUMMARY")
    print("=" * 60)

    summary = analysis["basic_summary"]

    print(
        f"Total sales: "
        f"{summary['total_sales']}"
    )

    print(
        f"Total orders: "
        f"{summary['total_orders']}"
    )

    print(
        f"Months analyzed: "
        f"{summary['months_analyzed']}"
    )

    print(
        f"Average monthly sales: "
        f"{summary['average_monthly_sales']}"
    )

    print()
    print("BEST CATEGORY")
    print("=" * 60)

    print(
        analysis["category_analysis"]
        ["best_category"]
    )

    print()
    print("WORST CATEGORY")
    print("=" * 60)

    print(
        analysis["category_analysis"]
        ["worst_category"]
    )

    print()
    print("BEST REGION")
    print("=" * 60)

    print(
        analysis["region_analysis"]
        ["best_region"]
    )

    print()
    print("WORST REGION")
    print("=" * 60)

    print(
        analysis["region_analysis"]
        ["worst_region"]
    )

    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    print(
        f"Saved → {OUTPUT_FILE}"
    )