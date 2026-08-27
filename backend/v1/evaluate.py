import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load predictions produced by train.py
df = pd.read_csv(
    '../data/processed/predictions.csv',
    parse_dates=['month']
)

# ---------------------------------------------------------
# 1. Overall performance
# ---------------------------------------------------------

baseline_mae = mean_absolute_error(
    df['actual'],
    df['baseline_pred']
)

xgb_mae = mean_absolute_error(
    df['actual'],
    df['xgb_pred']
)

baseline_rmse = np.sqrt(
    mean_squared_error(df['actual'], df['baseline_pred'])
)

xgb_rmse = np.sqrt(
    mean_squared_error(df['actual'], df['xgb_pred'])
)

print("=" * 60)
print("OVERALL MODEL PERFORMANCE")
print("=" * 60)

print(f"Baseline MAE : {baseline_mae:.2f}")
print(f"XGBoost MAE  : {xgb_mae:.2f}")

print(f"\nBaseline RMSE: {baseline_rmse:.2f}")
print(f"XGBoost RMSE : {xgb_rmse:.2f}")

improvement = (
    (baseline_mae - xgb_mae)
    / baseline_mae
    * 100
)

print(f"\nXGBoost MAE improvement: {improvement:.1f}%")


# ---------------------------------------------------------
# 2. Performance by category
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PERFORMANCE BY CATEGORY")
print("=" * 60)

category_results = []

for category, group in df.groupby('Product_Category'):

    baseline = mean_absolute_error(
        group['actual'],
        group['baseline_pred']
    )

    xgb = mean_absolute_error(
        group['actual'],
        group['xgb_pred']
    )

    improvement = (
        (baseline - xgb)
        / baseline
        * 100
        if baseline != 0 else 0
    )

    category_results.append({
        'Product_Category': category,
        'Baseline_MAE': baseline,
        'XGBoost_MAE': xgb,
        'Improvement_%': improvement
    })

category_results = pd.DataFrame(category_results)

print(
    category_results
    .sort_values('XGBoost_MAE')
    .to_string(index=False)
)


# ---------------------------------------------------------
# 3. Performance by region
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PERFORMANCE BY REGION")
print("=" * 60)

region_results = []

for region, group in df.groupby('Region'):

    baseline = mean_absolute_error(
        group['actual'],
        group['baseline_pred']
    )

    xgb = mean_absolute_error(
        group['actual'],
        group['xgb_pred']
    )

    improvement = (
        (baseline - xgb)
        / baseline
        * 100
        if baseline != 0 else 0
    )

    region_results.append({
        'Region': region,
        'Baseline_MAE': baseline,
        'XGBoost_MAE': xgb,
        'Improvement_%': improvement
    })

region_results = pd.DataFrame(region_results)

print(
    region_results
    .sort_values('XGBoost_MAE')
    .to_string(index=False)
)


# ---------------------------------------------------------
# 4. Zero-sales analysis
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ZERO-SALES ANALYSIS")
print("=" * 60)

zero_sales = df[df['actual'] == 0]

non_zero_sales = df[df['actual'] > 0]

print(f"Zero-sales observations: {len(zero_sales)}")
print(
    f"Percentage of test observations: "
    f"{len(zero_sales) / len(df) * 100:.1f}%"
)

if len(zero_sales) > 0:

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
        f"Zero-sales XGBoost MAE: "
        f"{zero_xgb_mae:.2f}"
    )


# ---------------------------------------------------------
# 5. Largest prediction errors
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("LARGEST XGBOOST ERRORS")
print("=" * 60)

df['xgb_error'] = (
    df['actual'] - df['xgb_pred']
).abs()

largest_errors = (
    df[
        [
            'month',
            'Product_Category',
            'Region',
            'actual',
            'xgb_pred',
            'xgb_error'
        ]
    ]
    .sort_values('xgb_error', ascending=False)
    .head(10)
)

print(largest_errors.to_string(index=False))


# ---------------------------------------------------------
# 6. Best and worst category-region combinations
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("CATEGORY × REGION PERFORMANCE")
print("=" * 60)

combination_results = []

for (category, region), group in df.groupby(
    ['Product_Category', 'Region']
):

    baseline = mean_absolute_error(
        group['actual'],
        group['baseline_pred']
    )

    xgb = mean_absolute_error(
        group['actual'],
        group['xgb_pred']
    )

    improvement = (
        (baseline - xgb)
        / baseline
        * 100
        if baseline != 0 else 0
    )

    combination_results.append({
        'Product_Category': category,
        'Region': region,
        'Baseline_MAE': baseline,
        'XGBoost_MAE': xgb,
        'Improvement_%': improvement
    })

combination_results = pd.DataFrame(
    combination_results
)

print("\nBest performing combinations:")

print(
    combination_results
    .sort_values('XGBoost_MAE')
    .head(10)
    .to_string(index=False)
)

print("\nWorst performing combinations:")

print(
    combination_results
    .sort_values('XGBoost_MAE', ascending=False)
    .head(10)
    .to_string(index=False)
)


# ---------------------------------------------------------
# 7. Save evaluation results
# ---------------------------------------------------------

category_results.to_csv(
    '../data/processed/category_evaluation.csv',
    index=False
)

region_results.to_csv(
    '../data/processed/region_evaluation.csv',
    index=False
)

combination_results.to_csv(
    '../data/processed/category_region_evaluation.csv',
    index=False
)

print("\n" + "=" * 60)
print("Evaluation files saved.")
print("=" * 60)

print("category_evaluation.csv")
print("region_evaluation.csv")
print("category_region_evaluation.csv")