# ai/analysis/facts.py

from typing import Any, Dict


def calculate_facts(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate verified business facts from the selected context.

    The LLM should explain these facts, not invent or recalculate them.
    """

    facts = {}

    # ============================================================
    # BUSINESS SUMMARY
    # ============================================================

    summary = context.get("business_summary", {})
    basic = summary.get("basic_summary", {})

    total_sales = basic.get("total_sales", 0)
    total_orders = basic.get("total_orders", 0)
    months_analyzed = basic.get("months_analyzed", 0)
    average_monthly_sales = basic.get("average_monthly_sales", 0)

    facts["business"] = {
        "total_sales": round(total_sales, 2),
        "total_orders": total_orders,
        "months_analyzed": months_analyzed,
        "average_monthly_sales": round(average_monthly_sales, 2),
    }

    # ============================================================
    # CATEGORY FACTS
    # ============================================================

    best_category = summary.get("best_category")
    worst_category = summary.get("worst_category")

    if best_category:
        category_sales = best_category.get("total_sales", 0)
        category_orders = best_category.get("total_orders", 0)

        facts["best_category"] = {
            "category": best_category.get("Product_Category"),
            "total_sales": round(category_sales, 2),
            "sales_share_percent": round(
                (category_sales / total_sales) * 100, 2
            ) if total_sales else 0,
            "total_orders": category_orders,
            "order_share_percent": round(
                (category_orders / total_orders) * 100, 2
            ) if total_orders else 0,
            "average_order_value": round(
                best_category.get("average_order_value", 0), 2
            ),
        }

    if worst_category:
        category_sales = worst_category.get("total_sales", 0)
        category_orders = worst_category.get("total_orders", 0)

        facts["worst_category"] = {
            "category": worst_category.get("Product_Category"),
            "total_sales": round(category_sales, 2),
            "sales_share_percent": round(
                (category_sales / total_sales) * 100, 2
            ) if total_sales else 0,
            "total_orders": category_orders,
            "order_share_percent": round(
                (category_orders / total_orders) * 100, 2
            ) if total_orders else 0,
            "average_order_value": round(
                worst_category.get("average_order_value", 0), 2
            ),
        }

    # ============================================================
    # REGION FACTS
    # ============================================================

    best_region = summary.get("best_region")
    worst_region = summary.get("worst_region")

    if best_region:
        region_sales = best_region.get("total_sales", 0)
        region_orders = best_region.get("total_orders", 0)

        facts["best_region"] = {
            "region": best_region.get("Region"),
            "total_sales": round(region_sales, 2),
            "sales_share_percent": round(
                (region_sales / total_sales) * 100, 2
            ) if total_sales else 0,
            "total_orders": region_orders,
            "order_share_percent": round(
                (region_orders / total_orders) * 100, 2
            ) if total_orders else 0,
            "average_order_value": round(
                best_region.get("average_order_value", 0), 2
            ),
        }

    if worst_region:
        region_sales = worst_region.get("total_sales", 0)
        region_orders = worst_region.get("total_orders", 0)

        facts["worst_region"] = {
            "region": worst_region.get("Region"),
            "total_sales": round(region_sales, 2),
            "sales_share_percent": round(
                (region_sales / total_sales) * 100, 2
            ) if total_sales else 0,
            "total_orders": region_orders,
            "order_share_percent": round(
                (region_orders / total_orders) * 100, 2
            ) if total_orders else 0,
            "average_order_value": round(
                worst_region.get("average_order_value", 0), 2
            ),
        }

    # ============================================================
    # PREDICTION FACTS
    # ============================================================

    predictions = context.get("predictions", [])

    if predictions:
        errors = [
            abs(float(row.get("xgb_error", 0)))
            for row in predictions
            if row.get("xgb_error") is not None
        ]

        if errors:
            facts["forecast"] = {
                "prediction_records": len(predictions),
                "largest_absolute_error": round(max(errors), 2),
                "average_absolute_error": round(
                    sum(errors) / len(errors), 2
                ),
            }

    # ============================================================
    # EVALUATION FACTS
    # ============================================================

    category_evaluation = context.get("category_evaluation", [])

    if category_evaluation:
        row = category_evaluation[0]

        facts["category_model_evaluation"] = {
            "category": row.get("Product_Category"),
            "baseline_mae": round(
                float(row.get("Baseline_MAE", 0)), 2
            ),
            "xgboost_mae": round(
                float(row.get("XGBoost_MAE", 0)), 2
            ),
            "improvement_percent": round(
                float(row.get("Improvement_%", 0)), 2
            ),
        }

    region_evaluation = context.get("region_evaluation", [])

    if region_evaluation:
        row = region_evaluation[0]

        facts["region_model_evaluation"] = {
            "region": row.get("Region"),
            "baseline_mae": round(
                float(row.get("Baseline_MAE", 0)), 2
            ),
            "xgboost_mae": round(
                float(row.get("XGBoost_MAE", 0)), 2
            ),
            "improvement_percent": round(
                float(row.get("Improvement_%", 0)), 2
            ),
        }

    return facts