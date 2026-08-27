"""
STEP 9: Evaluate XGBoost V2

Analyzes:
    1. Overall performance
    2. Performance by category
    3. Performance by region
    4. Zero-sales performance
    5. Largest prediction errors
    6. Category × Region performance

The results can be compared directly with evaluate.py from V1.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD V2 PREDICTIONS
# ============================================================

df = pd.read_csv(
    '../data/processed/predictions_v2.csv',
    parse_dates=['month']
)

print("=" * 60)
print("V2 MODEL EVALUATION")
print("=" * 60)


# ============================================================
# 2. CREATE ERROR COLUMNS
# ============================================================

df['baseline_error'] = (
    df['actual'] - df['baseline_pred']
)

df['xgb_error'] = (
    df['actual'] - df['xgb_pred']
)

df['baseline_abs_error'] = (
    df['baseline_error'].abs()
)

df['xgb_abs_error'] = (
    df['xgb_error'].abs()
)


# ============================================================
# 3. OVERALL PERFORMANCE
# ============================================================

baseline_mae = mean_absolute_error(
    df['actual'],
    df['baseline_pred']
)

xgb_mae = mean_absolute_error(
    df['actual'],
    df['xgb_pred']
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        df['actual'],
        df['baseline_pred']
    )
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        df['actual'],
        df['xgb_pred']
    )
)

improvement = (
    (baseline_mae - xgb_mae)
    / baseline_mae
    * 100
)


print("\n" + "=" * 60)
print("OVERALL MODEL PERFORMANCE")
print("=" * 60)

print(f"Baseline MAE : {baseline_mae:.2f}")
print(f"XGBoost V2 MAE  : {xgb_mae:.2f}")

print(f"\nBaseline RMSE: {baseline_rmse:.2f}")
print(f"XGBoost V2 RMSE : {xgb_rmse:.2f}")

print(
    f"\nXGBoost V2 MAE improvement: "
    f"{improvement:.1f}%"
)


# ============================================================
# 4. PERFORMANCE BY CATEGORY
# ============================================================

category_eval = (
    df.groupby('Product_Category')
    .agg(
        Baseline_MAE=(
            'baseline_abs_error',
            'mean'
        ),
        XGBoost_MAE=(
            'xgb_abs_error',
            'mean'
        )
    )
    .reset_index()
)

category_eval['Improvement_%'] = (
    (
        category_eval['Baseline_MAE']
        - category_eval['XGBoost_MAE']
    )
    / category_eval['Baseline_MAE']
    * 100
)

print("\n" + "=" * 60)
print("PERFORMANCE BY CATEGORY")
print("=" * 60)

print(
    category_eval.to_string(
        index=False
    )
)


# ============================================================
# 5. PERFORMANCE BY REGION
# ============================================================

region_eval = (
    df.groupby('Region')
    .agg(
        Baseline_MAE=(
            'baseline_abs_error',
            'mean'
        ),
        XGBoost_MAE=(
            'xgb_abs_error',
            'mean'
        )
    )
    .reset_index()
)

region_eval['Improvement_%'] = (
    (
        region_eval['Baseline_MAE']
        - region_eval['XGBoost_MAE']
    )
    / region_eval['Baseline_MAE']
    * 100
)

print("\n" + "=" * 60)
print("PERFORMANCE BY REGION")
print("=" * 60)

print(
    region_eval.to_string(
        index=False
    )
)


# ============================================================
# 6. ZERO-SALES ANALYSIS
# ============================================================

zero_sales = df[
    df['actual'] == 0
]

zero_count = len(zero_sales)
zero_percentage = (
    zero_count / len(df) * 100
)

print("\n" + "=" * 60)
print("ZERO-SALES ANALYSIS")
print("=" * 60)

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
        zero_sales['actual'],
        zero_sales['baseline_pred']
    )

    zero_xgb_mae = mean_absolute_error(
        zero_sales['actual'],
        zero_sales['xgb_pred']
    )

    print(
        f"\nZero-sales Baseline MAE: "
        f"{zero_baseline_mae:.2f}"
    )

    print(
        f"Zero-sales XGBoost V2 MAE: "
        f"{zero_xgb_mae:.2f}"
    )


# ============================================================
# 7. LARGEST XGBOOST ERRORS
# ============================================================

largest_errors = (
    df.sort_values(
        'xgb_abs_error',
        ascending=False
    )
    [
        [
            'month',
            'Product_Category',
            'Region',
            'actual',
            'xgb_pred',
            'xgb_abs_error'
        ]
    ]
    .head(10)
)

largest_errors = largest_errors.rename(
    columns={
        'xgb_abs_error': 'xgb_error'
    }
)

print("\n" + "=" * 60)
print("LARGEST XGBOOST V2 ERRORS")
print("=" * 60)

print(
    largest_errors.to_string(
        index=False
    )
)


# ============================================================
# 8. CATEGORY × REGION PERFORMANCE
# ============================================================

combo_eval = (
    df.groupby(
        [
            'Product_Category',
            'Region'
        ]
    )
    .agg(
        Baseline_MAE=(
            'baseline_abs_error',
            'mean'
        ),
        XGBoost_MAE=(
            'xgb_abs_error',
            'mean'
        )
    )
    .reset_index()
)

combo_eval['Improvement_%'] = (
    (
        combo_eval['Baseline_MAE']
        - combo_eval['XGBoost_MAE']
    )
    / combo_eval['Baseline_MAE']
    * 100
)

# Best combinations
best_combinations = (
    combo_eval
    .sort_values(
        'Improvement_%',
        ascending=False
    )
    .head(10)
)

# Worst combinations
worst_combinations = (
    combo_eval
    .sort_values(
        'Improvement_%',
        ascending=True
    )
    .head(10)
)


print("\n" + "=" * 60)
print("CATEGORY × REGION PERFORMANCE")
print("=" * 60)

print("\nBest performing combinations:")

print(
    best_combinations.to_string(
        index=False
    )
)

print("\nWorst performing combinations:")

print(
    worst_combinations.to_string(
        index=False
    )
)


# ============================================================
# 9. SAVE EVALUATION FILES
# ============================================================

category_eval.to_csv(
    '../data/processed/category_evaluation_v2.csv',
    index=False
)

region_eval.to_csv(
    '../data/processed/region_evaluation_v2.csv',
    index=False
)

combo_eval.to_csv(
    '../data/processed/category_region_evaluation_v2.csv',
    index=False
)

largest_errors.to_csv(
    '../data/processed/largest_errors_v2.csv',
    index=False
)


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION FILES SAVED")
print("=" * 60)

print("category_evaluation_v2.csv")
print("region_evaluation_v2.csv")
print("category_region_evaluation_v2.csv")
print("largest_errors_v2.csv")

print("\nV2 evaluation complete.")