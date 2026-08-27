"""
STEP 4: V3 MODEL EVALUATION

Evaluates the V3 forecasting model against the naive lag-1 baseline.

Analysis:
1. Overall performance
2. Performance by category
3. Performance by region
4. Zero-sales observations
5. Largest XGBoost errors
6. Category × Region performance
7. Saves evaluation CSV files
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# LOAD PREDICTIONS
# ============================================================

predictions = pd.read_csv(
    "../data/processed/predictions_v3.csv",
    parse_dates=["month"]
)

print("=" * 60)
print("V3 MODEL EVALUATION")
print("=" * 60)


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

actual = predictions["actual"]
baseline = predictions["baseline_pred"]
xgb = predictions["xgb_pred"]


baseline_mae = mean_absolute_error(actual, baseline)
xgb_mae = mean_absolute_error(actual, xgb)

baseline_rmse = np.sqrt(
    mean_squared_error(actual, baseline)
)

xgb_rmse = np.sqrt(
    mean_squared_error(actual, xgb)
)

improvement = (
    (baseline_mae - xgb_mae)
    / baseline_mae
) * 100


print("\n" + "=" * 60)
print("OVERALL MODEL PERFORMANCE")
print("=" * 60)

print(f"Baseline MAE : {baseline_mae:.2f}")
print(f"XGBoost V3 MAE  : {xgb_mae:.2f}")

print(f"\nBaseline RMSE: {baseline_rmse:.2f}")
print(f"XGBoost V3 RMSE : {xgb_rmse:.2f}")

print(
    f"\nXGBoost V3 MAE improvement: "
    f"{improvement:.1f}%"
)


# ============================================================
# PERFORMANCE BY CATEGORY
# ============================================================

print("\n" + "=" * 60)
print("PERFORMANCE BY CATEGORY")
print("=" * 60)


category_results = []

for category, group in predictions.groupby(
    "Product_Category"
):

    baseline_mae_cat = mean_absolute_error(
        group["actual"],
        group["baseline_pred"]
    )

    xgb_mae_cat = mean_absolute_error(
        group["actual"],
        group["xgb_pred"]
    )

    improvement_cat = (
        (baseline_mae_cat - xgb_mae_cat)
        / baseline_mae_cat
    ) * 100

    category_results.append({
        "Product_Category": category,
        "Baseline_MAE": baseline_mae_cat,
        "XGBoost_MAE": xgb_mae_cat,
        "Improvement_%": improvement_cat
    })


category_df = pd.DataFrame(category_results)

category_df = category_df.sort_values(
    "XGBoost_MAE"
)

print(category_df.to_string(index=False))


# ============================================================
# PERFORMANCE BY REGION
# ============================================================

print("\n" + "=" * 60)
print("PERFORMANCE BY REGION")
print("=" * 60)


region_results = []

for region, group in predictions.groupby(
    "Region"
):

    baseline_mae_region = mean_absolute_error(
        group["actual"],
        group["baseline_pred"]
    )

    xgb_mae_region = mean_absolute_error(
        group["actual"],
        group["xgb_pred"]
    )

    improvement_region = (
        (baseline_mae_region - xgb_mae_region)
        / baseline_mae_region
    ) * 100

    region_results.append({
        "Region": region,
        "Baseline_MAE": baseline_mae_region,
        "XGBoost_MAE": xgb_mae_region,
        "Improvement_%": improvement_region
    })


region_df = pd.DataFrame(region_results)

region_df = region_df.sort_values(
    "XGBoost_MAE"
)

print(region_df.to_string(index=False))


# ============================================================
# ZERO-SALES ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("ZERO-SALES ANALYSIS")
print("=" * 60)


zero_sales = predictions[
    predictions["actual"] == 0
]

zero_count = len(zero_sales)

zero_percentage = (
    zero_count / len(predictions)
) * 100


print(
    f"Zero-sales observations: "
    f"{zero_count}"
)

print(
    f"Percentage of test observations: "
    f"{zero_percentage:.1f}%"
)


if zero_count > 0:

    zero_baseline_mae = mean_absolute_error(
        zero_sales["actual"],
        zero_sales["baseline_pred"]
    )

    zero_xgb_mae = mean_absolute_error(
        zero_sales["actual"],
        zero_sales["xgb_pred"]
    )

    print(
        f"\nZero-sales Baseline MAE: "
        f"{zero_baseline_mae:.2f}"
    )

    print(
        f"Zero-sales XGBoost V3 MAE: "
        f"{zero_xgb_mae:.2f}"
    )

else:

    print("\nNo zero-sales observations in test set.")


# ============================================================
# LARGEST ERRORS
# ============================================================

print("\n" + "=" * 60)
print("LARGEST XGBOOST V3 ERRORS")
print("=" * 60)


error_df = predictions.copy()

error_df["xgb_error"] = (
    error_df["actual"]
    - error_df["xgb_pred"]
)

error_df["absolute_error"] = (
    error_df["xgb_error"]
    .abs()
)


largest_errors = (
    error_df
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(10)
)


display_errors = largest_errors[
    [
        "month",
        "Product_Category",
        "Region",
        "actual",
        "xgb_pred",
        "xgb_error"
    ]
]


print(
    display_errors.to_string(
        index=False
    )
)


# ============================================================
# CATEGORY × REGION PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("CATEGORY × REGION PERFORMANCE")
print("=" * 60)


combination_results = []


for (
    category,
    region
), group in predictions.groupby(
    [
        "Product_Category",
        "Region"
    ]
):

    baseline_mae_combo = mean_absolute_error(
        group["actual"],
        group["baseline_pred"]
    )

    xgb_mae_combo = mean_absolute_error(
        group["actual"],
        group["xgb_pred"]
    )

    improvement_combo = (
        (baseline_mae_combo - xgb_mae_combo)
        / baseline_mae_combo
    ) * 100

    combination_results.append({
        "Product_Category": category,
        "Region": region,
        "Baseline_MAE": baseline_mae_combo,
        "XGBoost_MAE": xgb_mae_combo,
        "Improvement_%": improvement_combo
    })


combination_df = pd.DataFrame(
    combination_results
)


# ============================================================
# BEST COMBINATIONS
# ============================================================

print("\nBest performing combinations:")

best = (
    combination_df
    .sort_values(
        "Improvement_%",
        ascending=False
    )
    .head(10)
)

print(
    best.to_string(
        index=False
    )
)


# ============================================================
# WORST COMBINATIONS
# ============================================================

print("\nWorst performing combinations:")

worst = (
    combination_df
    .sort_values(
        "Improvement_%",
        ascending=True
    )
    .head(10)
)

print(
    worst.to_string(
        index=False
    )
)


# ============================================================
# SAVE EVALUATION FILES
# ============================================================

category_df.to_csv(
    "../data/processed/category_evaluation_v3.csv",
    index=False
)

region_df.to_csv(
    "../data/processed/region_evaluation_v3.csv",
    index=False
)

combination_df.to_csv(
    "../data/processed/category_region_evaluation_v3.csv",
    index=False
)

largest_errors.to_csv(
    "../data/processed/largest_errors_v3.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("V3 EVALUATION FILES SAVED")
print("=" * 60)

print("category_evaluation_v3.csv")
print("region_evaluation_v3.csv")
print("category_region_evaluation_v3.csv")
print("largest_errors_v3.csv")

print("\nV3 evaluation complete.")