import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ai.models.xgboost_model import train_model, predict


# ============================================================
# LOAD FEATURES
# ============================================================

df = pd.read_csv(
    "../data/processed/features_v3.csv",
    parse_dates=["month"]
)

df = df.sort_values("month").reset_index(drop=True)


# ============================================================
# TIME SPLIT
# ============================================================

split_date = pd.Timestamp("2025-01-01")

train = df[df["month"] < split_date].copy()
test = df[df["month"] >= split_date].copy()


print("=" * 60)
print("XGBOOST MODEL EVALUATION")
print("=" * 60)

print(
    f"\nTraining rows: {len(train)}"
)

print(
    f"Test rows: {len(test)}"
)

print(
    f"Train period: "
    f"{train['month'].min().date()} → "
    f"{train['month'].max().date()}"
)

print(
    f"Test period: "
    f"{test['month'].min().date()} → "
    f"{test['month'].max().date()}"
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining XGBoost...")

model = train_model(train)

print("Model trained successfully.")


# ============================================================
# XGBOOST PREDICTIONS
# ============================================================

xgb_pred = predict(
    model,
    test
)


# ============================================================
# BASELINE
# ============================================================

y_test = test["total_sales"]

baseline_pred = test["lag_1"]

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
# XGBOOST EVALUATION
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


# ============================================================
# IMPROVEMENT
# ============================================================

improvement = (
    (baseline_mae - xgb_mae)
    / baseline_mae
) * 100


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(
    f"\nBaseline MAE : {baseline_mae:.2f}"
)

print(
    f"Baseline RMSE: {baseline_rmse:.2f}"
)

print(
    f"\nXGBoost MAE  : {xgb_mae:.2f}"
)

print(
    f"XGBoost RMSE : {xgb_rmse:.2f}"
)

print(
    f"\nImprovement  : {improvement:.2f}%"
)


# ============================================================
# PREDICTION SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

results = test[
    [
        "month",
        "total_sales",
        "lag_1"
    ]
].copy()

results["xgb_prediction"] = xgb_pred

results["absolute_error"] = (
    results["total_sales"]
    - results["xgb_prediction"]
).abs()

print(
    results.head(10).to_string(
        index=False
    )
)


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)