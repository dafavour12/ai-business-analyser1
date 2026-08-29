from typing import Any, Dict, Optional


def compact_business_summary(
    summary: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Keep only the important business summary information.

    This prevents unnecessary data from being passed
    to the next analysis stage.
    """

    category_analysis = summary.get(
        "category_analysis",
        {}
    )

    region_analysis = summary.get(
        "region_analysis",
        {}
    )

    return {
        "basic_summary": summary.get(
            "basic_summary",
            {}
        ),

        "best_category": category_analysis.get(
            "best_category"
        ),

        "worst_category": category_analysis.get(
            "worst_category"
        ),

        "best_region": region_analysis.get(
            "best_region"
        ),

        "worst_region": region_analysis.get(
            "worst_region"
        ),
    }


def select_context(
    context: Dict[str, Any],
    category: Optional[str] = None,
    region: Optional[str] = None,
    max_monthly_records: int = 12
) -> Dict[str, Any]:
    """
    Select the most relevant business context for analysis.

    category:
        Optional product category filter.

    region:
        Optional region filter.

    max_monthly_records:
        Maximum number of recent monthly records to keep.
    """

    selected = {}

    # ============================================================
    # BUSINESS SUMMARY
    # ============================================================

    selected["business_summary"] = compact_business_summary(
        context.get("business_summary", {})
    )

    # ============================================================
    # MONTHLY SALES
    # ============================================================

    monthly = list(
        context.get("monthly_sales", [])
    )

    if category:
        monthly = [
            row
            for row in monthly
            if row.get("Product_Category") == category
        ]

    if region:
        monthly = [
            row
            for row in monthly
            if row.get("Region") == region
        ]

    # Keep the most recent records
    monthly = monthly[-max_monthly_records:]

    selected["monthly_sales"] = monthly

    # ============================================================
    # PREDICTIONS
    # ============================================================

    predictions = list(
        context.get("predictions", [])
    )

    if category:
        predictions = [
            row
            for row in predictions
            if row.get("Product_Category") == category
        ]

    if region:
        predictions = [
            row
            for row in predictions
            if row.get("Region") == region
        ]

    # Keep recent predictions
    selected["predictions"] = predictions[-12:]

    # ============================================================
    # CATEGORY MODEL EVALUATION
    # ============================================================

    category_eval = list(
        context.get("category_evaluation", [])
    )

    if category:
        category_eval = [
            row
            for row in category_eval
            if row.get("Product_Category") == category
        ]

    selected["category_evaluation"] = category_eval

    # ============================================================
    # REGION MODEL EVALUATION
    # ============================================================

    region_eval = list(
        context.get("region_evaluation", [])
    )

    if region:
        region_eval = [
            row
            for row in region_eval
            if row.get("Region") == region
        ]

    selected["region_evaluation"] = region_eval

    # ============================================================
    # LARGEST ERRORS
    # ============================================================

    errors = list(
        context.get("largest_errors", [])
    )

    if category:
        errors = [
            row
            for row in errors
            if row.get("Product_Category") == category
        ]

    if region:
        errors = [
            row
            for row in errors
            if row.get("Region") == region
        ]

    selected["largest_errors"] = errors[:5]

    return selected