def select_context(
    context,
    category=None,
    region=None,
    max_monthly_records=12
):
    selected = {}

    # --------------------------------------------------
    # BUSINESS SUMMARY
    # --------------------------------------------------

    selected["business_summary"] = compact_business_summary(
    context["business_summary"]
)

    # --------------------------------------------------
    # MONTHLY SALES
    # --------------------------------------------------

    monthly = context.get("monthly_sales", [])

    if category:
        monthly = [
            row for row in monthly
            if row.get("Product_Category") == category
        ]

    if region:
        monthly = [
            row for row in monthly
            if row.get("Region") == region
        ]

    # Only keep the most recent records
    monthly = monthly[-max_monthly_records:]

    selected["monthly_sales"] = monthly

    # --------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------

    predictions = context.get("predictions", [])

    if category:
        predictions = [
            row for row in predictions
            if row.get("Product_Category") == category
        ]

    if region:
        predictions = [
            row for row in predictions
            if row.get("Region") == region
        ]

    selected["predictions"] = predictions[-12:]

    # --------------------------------------------------
    # CATEGORY EVALUATION
    # --------------------------------------------------

    category_eval = context.get("category_evaluation", [])

    if category:
        category_eval = [
            row for row in category_eval
            if row.get("Product_Category") == category
        ]

    selected["category_evaluation"] = category_eval

    # --------------------------------------------------
    # REGION EVALUATION
    # --------------------------------------------------

    region_eval = context.get("region_evaluation", [])

    if region:
        region_eval = [
            row for row in region_eval
            if row.get("Region") == region
        ]

    selected["region_evaluation"] = region_eval

    # --------------------------------------------------
    # LARGEST ERRORS
    # --------------------------------------------------

    errors = context.get("largest_errors", [])

    if category:
        errors = [
            row for row in errors
            if row.get("Product_Category") == category
        ]

    if region:
        errors = [
            row for row in errors
            if row.get("Region") == region
        ]

    selected["largest_errors"] = errors[:5]

    return selected

def compact_business_summary(summary):
    return {
        "basic_summary": summary.get("basic_summary", {}),
        "best_category": summary.get(
            "category_analysis", {}
        ).get("best_category"),

        "worst_category": summary.get(
            "category_analysis", {}
        ).get("worst_category"),

        "best_region": summary.get(
            "region_analysis", {}
        ).get("best_region"),

        "worst_region": summary.get(
            "region_analysis", {}
        ).get("worst_region"),
    }
