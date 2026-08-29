from pathlib import Path
import pandas as pd
import json
from ai.analysis.business import (
    get_basic_summary,
    analyze_categories,
    analyze_regions,
    analyze_monthly_trend,
    analyze_spikes,
    analyze_category_region,
)

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "processed"


def load_business_context():
    """
    Load the existing processed business context.

    Used for development/testing with the project's
    preprocessed dataset.
    """

    business_file = DATA_DIR / "business_analysis.json"

    with open(business_file, "r", encoding="utf-8") as f:
        business = json.load(f)

    monthly = pd.read_csv(
        DATA_DIR / "monthly_agg_v3.csv"
    )

    predictions = pd.read_csv(
        DATA_DIR / "predictions_v3.csv"
    )

    category_eval = pd.read_csv(
        DATA_DIR / "category_evaluation_v3.csv"
    )

    region_eval = pd.read_csv(
        DATA_DIR / "region_evaluation_v3.csv"
    )

    largest_errors = pd.read_csv(
        DATA_DIR / "largest_errors_v3.csv"
    )

    return {
        "business_summary": business,
        "monthly_sales": monthly.to_dict(orient="records"),
        "predictions": predictions.to_dict(orient="records"),
        "category_evaluation": category_eval.to_dict(orient="records"),
        "region_evaluation": region_eval.to_dict(orient="records"),
        "largest_errors": largest_errors.to_dict(orient="records"),
    }


def build_business_context(
    df: pd.DataFrame,
    monthly: pd.DataFrame,
    features: pd.DataFrame,
    predictions
):
    """
    Build business context from the current uploaded dataset.
    """

    # --------------------------------------------------
    # BUSINESS ANALYSIS
    # --------------------------------------------------

    business_summary = {
        "basic_summary": get_basic_summary(monthly),
        "category_analysis": analyze_categories(monthly),
        "region_analysis": analyze_regions(monthly),
        "monthly_trend": analyze_monthly_trend(monthly),
        "sales_spikes": analyze_spikes(monthly),
        "category_region_analysis": analyze_category_region(monthly),
    }

    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()

    return {
        "business_summary": business_summary,

        "monthly_sales": monthly.to_dict(
            orient="records"
        ),

        "predictions": predictions,

        "category_evaluation": [],

        "region_evaluation": [],

        "largest_errors": []
    }


if __name__ == "__main__":
    context = load_business_context()

    print("=" * 60)
    print("AGENT CONTEXT TEST")
    print("=" * 60)

    print("\nAvailable context:")
    print(context.keys())

    print("\nBusiness summary:")
    print(context["business_summary"])

    print("\nMonthly records:")
    print(len(context["monthly_sales"]))

    print("\nPredictions:")
    print(len(context["predictions"]))

    print("\nCategory evaluation:")
    print(len(context["category_evaluation"]))

    print("\nRegion evaluation:")
    print(len(context["region_evaluation"]))

    print("\nLargest errors:")
    print(len(context["largest_errors"]))