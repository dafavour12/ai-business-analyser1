"""
STEP 8: Train and evaluate XGBoost V2.

V2 uses the expanded feature set from features_v2.csv:

- lag_1, lag_2, lag_3
- lag_6
- lag_12
- 3, 6 and 12-month rolling statistics
- calendar features
- category and region

We exclude features that would not be known before the
forecasted month:
    order_count
    avg_unit_price
    avg_discount
    total_quantity

The test period remains 2025 so we can fairly compare
V2 against our V1 model.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. LOAD V2 FEATURES
# ============================================================

df = pd.read_csv(
    '../data/processed/features_v2.csv',
    parse_dates=['month']
)


# ============================================================
# 2. DEFINE FEATURES
# ============================================================

target = 'total_sales'

# These values come from the same month as the target.
# Therefore, they would cause data leakage in a real forecast.
leaky_cols = [
    'order_count',
    'avg_unit_price',
    'avg_discount',
    'total_quantity'
]

drop_cols = [
    'month',
    'total_sales',
    'total_profit'
] + leaky_cols

feature_cols = [
    col for col in df.columns
    if col not in drop_cols
]

print("=" * 60)
print("V2 MODEL TRAINING")
print("=" * 60)

print(f"\nUsing {len(feature_cols)} leakage-safe features:")
print(feature_cols)


# ============================================================
# 3. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

split_date = pd.Timestamp('2025-01-01')

train = df[df['month'] < split_date]
test = df[df['month'] >= split_date]

print("\nTrain:")
print(
    f"{len(train)} rows "
    f"({train['month'].min().date()} "
    f"to {train['month'].max().date()})"
)

print("\nTest:")
print(
    f"{len(test)} rows "
    f"({test['month'].min().date()} "
    f"to {test['month'].max().date()})"
)


# ============================================================
# 4. PREPARE X AND Y
# ============================================================

X_train = train[feature_cols]
y_train = train[target]

X_test = test[feature_cols]
y_test = test[target]


# ============================================================
# 5. BASELINE
# ============================================================

# Predict this month using the previous month's actual sales.
baseline_pred = X_test['lag_1']

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_pred
    )
)


# ============================================================
# 6. XGBOOST V2
# ============================================================

model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ============================================================
# 7. PREDICTIONS
# ============================================================

xgb_pred = model.predict(X_test)

# Sales cannot be negative.
xgb_pred = np.clip(
    xgb_pred,
    0,
    None
)


# ============================================================
# 8. EVALUATION
# ============================================================

xgb_mae = mean_absolute_error(
    y_test,
    xgb_pred
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_pred
    )
)

improvement = (
    (baseline_mae - xgb_mae)
    / baseline_mae
    * 100
)


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("V2 RESULTS")
print("=" * 60)

print(
    f"{'Model':<25}"
    f"{'MAE':>12}"
    f"{'RMSE':>12}"
)

print(
    f"{'Baseline (lag-1)':<25}"
    f"{baseline_mae:>12.2f}"
    f"{baseline_rmse:>12.2f}"
)

print(
    f"{'XGBoost V2':<25}"
    f"{xgb_mae:>12.2f}"
    f"{xgb_rmse:>12.2f}"
)

print(
    f"\nV2 improvement over baseline: "
    f"{improvement:.1f}%"
)


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importances = (
    pd.Series(
        model.feature_importances_,
        index=feature_cols
    )
    .sort_values(
        ascending=False
    )
)

print("\nTop 10 feature importances:")
print(importances.head(10))


# ============================================================
# 11. SAVE PREDICTIONS
# ============================================================

test_out = test[
    ['month']
].copy()

# Recover category from one-hot encoded columns
cat_cols = [
    c for c in df.columns
    if c.startswith('cat_')
]

reg_cols = [
    c for c in df.columns
    if c.startswith('reg_')
]

test_out['Product_Category'] = (
    df.loc[test.index, cat_cols]
    .idxmax(axis=1)
    .str.replace('cat_', '', regex=False)
)

test_out['Region'] = (
    df.loc[test.index, reg_cols]
    .idxmax(axis=1)
    .str.replace('reg_', '', regex=False)
)

test_out['actual'] = y_test.values
test_out['baseline_pred'] = baseline_pred.values
test_out['xgb_pred'] = xgb_pred

test_out.to_csv(
    '../data/processed/predictions_v2.csv',
    index=False
)


# ============================================================
# 12. SAVE FEATURE IMPORTANCE
# ============================================================

importances.to_csv(
    '../data/processed/feature_importance_v2.csv'
)


# ============================================================
# DONE
# ============================================================

print("\nSaved:")
print("  → predictions_v2.csv")
print("  → feature_importance_v2.csv")