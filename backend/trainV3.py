"""
STEP 3: V3 MODEL TRAINING

Train and compare:
1. Naive baseline using lag_1
2. XGBoost V3 using leakage-safe historical features

V3 uses calendar-aware lag and rolling features, so missing months
do not incorrectly become the previous month's lag.

We train chronologically:
    Train -> before 2025
    Test  -> 2025

This prevents future information from leaking into training.
"""

import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "../data/processed/features_v3.csv",
    parse_dates=["month"]
)

df = df.sort_values("month").reset_index(drop=True)


# ============================================================
# TARGET AND LEAKAGE PROTECTION
# ============================================================

target = "total_sales"

# These values come from the same month as total_sales.
# They would NOT be known when making a real future forecast.
leaky_cols = [
    "order_count",
    "avg_unit_price",
    "avg_discount",
    "total_quantity"
]

drop_cols = [
    "month",
    "total_sales",
    "total_profit"
] + leaky_cols


feature_cols = [
    col for col in df.columns
    if col not in drop_cols
]


print("=" * 60)
print("V3 MODEL TRAINING")
print("=" * 60)

print(f"\nUsing {len(feature_cols)} leakage-safe features:")

for feature in feature_cols:
    print(f"  - {feature}")


# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

split_date = pd.Timestamp("2025-01-01")

train = df[df["month"] < split_date].copy()
test = df[df["month"] >= split_date].copy()


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(
    f"\nTrain: {len(train)} rows "
    f"({train['month'].min().date()} to "
    f"{train['month'].max().date()})"
)

print(
    f"Test:  {len(test)} rows "
    f"({test['month'].min().date()} to "
    f"{test['month'].max().date()})"
)


# ============================================================
# PREPARE X AND Y
# ============================================================

X_train = train[feature_cols]
y_train = train[target]

X_test = test[feature_cols]
y_test = test[target]


# ============================================================
# BASELINE MODEL
# ============================================================

print("\n" + "=" * 60)
print("BASELINE MODEL")
print("=" * 60)

# Predict this month's sales using the previous calendar month's sales.
baseline_pred = X_test["lag_1"]

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

print(f"\nBaseline MAE : {baseline_mae:.2f}")
print(f"Baseline RMSE: {baseline_rmse:.2f}")


# ============================================================
# XGBOOST V3
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST V3")
print("=" * 60)


model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

xgb_pred = model.predict(X_test)

# Sales cannot be negative.
xgb_pred = np.clip(
    xgb_pred,
    0,
    None
)


# ============================================================
# EVALUATION
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
) * 100


print("\n" + "=" * 60)
print("V3 RESULTS")
print("=" * 60)

print(
    f"{'Model':<30}"
    f"{'MAE':>12}"
    f"{'RMSE':>12}"
)

print(
    f"{'Baseline (lag-1)':<30}"
    f"{baseline_mae:>12.2f}"
    f"{baseline_rmse:>12.2f}"
)

print(
    f"{'XGBoost V3':<30}"
    f"{xgb_mae:>12.2f}"
    f"{xgb_rmse:>12.2f}"
)

print(
    f"\nV3 improvement over baseline: "
    f"{improvement:.1f}%"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 FEATURE IMPORTANCES")
print("=" * 60)


importances = (
    pd.Series(
        model.feature_importances_,
        index=feature_cols
    )
    .sort_values(ascending=False)
)


print(importances.head(10))


# ============================================================
# RECONSTRUCT CATEGORY AND REGION
# ============================================================

category_cols = [
    col for col in df.columns
    if col.startswith("cat_")
]

region_cols = [
    col for col in df.columns
    if col.startswith("reg_")
]

predictions = test[["month"]].copy()

predictions["Product_Category"] = (
    test[category_cols]
    .idxmax(axis=1)
    .str.replace("cat_", "", regex=False)
)

predictions["Region"] = (
    test[region_cols]
    .idxmax(axis=1)
    .str.replace("reg_", "", regex=False)
)

predictions["actual"] = y_test.values
predictions["baseline_pred"] = baseline_pred.values
predictions["xgb_pred"] = xgb_pred


predictions["xgb_error"] = (
    predictions["actual"]
    - predictions["xgb_pred"]
)

predictions["absolute_error"] = (
    predictions["xgb_error"]
    .abs()
)


predictions.to_csv(
    "../data/processed/predictions_v3.csv",
    index=False
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importances.to_csv(
    "../data/processed/feature_importance_v3.csv"
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("V3 TRAINING COMPLETE")
print("=" * 60)

print("\nSaved:")
print("  → predictions_v3.csv")
print("  → feature_importance_v3.csv")