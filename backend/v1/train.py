"""
STEP 4-6: Train/test split, baseline model, and the real forecasting model.

IMPORTANT DATA LEAKAGE NOTE:
`order_count`, `avg_unit_price`, `avg_discount`, `total_quantity` in
features.csv are computed from the SAME month as the target (total_sales).
In a real forecast you would not know this month's order count before the
month happens - so we must NOT use them as predictors. We only use lagged/
rolling/calendar features, which are legitimately known in advance.

STEP 4: Time-based split (not random!)
Shuffling months randomly would let the model "see the future" via nearby
months in the training set. We split chronologically: train on everything
before 2025, test on 2025 (the final 12 months).

STEP 5: Baseline
A naive baseline predicts each month = last month's value (lag_1). Any
real model needs to beat this to be worth using.

STEP 6: XGBoost model
Gradient-boosted trees handle the mix of categorical (category/region)
and numeric (lags, rolling stats, calendar) features well without needing
scaling, and cope fine with the modest data size here.
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('../data/processed/features.csv', parse_dates=['month'])

# Features we're ALLOWED to use (known before the month happens)
leaky_cols = ['order_count', 'avg_unit_price', 'avg_discount', 'total_quantity']
target = 'total_sales'
drop_cols = ['month', 'total_sales', 'total_profit'] + leaky_cols
feature_cols = [c for c in df.columns if c not in drop_cols]

print(f"Using {len(feature_cols)} features (leakage-safe): {feature_cols}\n")

# --- Time-based split ---
split_date = pd.Timestamp('2025-01-01')
train = df[df['month'] < split_date]
test = df[df['month'] >= split_date]
print(f"Train: {len(train)} rows ({train['month'].min().date()} to {train['month'].max().date()})")
print(f"Test:  {len(test)} rows ({test['month'].min().date()} to {test['month'].max().date()})\n")

X_train, y_train = train[feature_cols], train[target]
X_test, y_test = test[feature_cols], test[target]

# --- Baseline: predict = lag_1 (last month's actual value) ---
baseline_pred = X_test['lag_1']
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

# --- XGBoost model ---
model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train)
xgb_pred = model.predict(X_test)
xgb_pred = np.clip(xgb_pred, 0, None)  # sales can't be negative

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

print("=" * 50)
print("RESULTS (test = 2025, 12 held-out months)")
print("=" * 50)
print(f"{'Model':<20}{'MAE':>12}{'RMSE':>12}")
print(f"{'Baseline (lag-1)':<20}{baseline_mae:>12.2f}{baseline_rmse:>12.2f}")
print(f"{'XGBoost':<20}{xgb_mae:>12.2f}{xgb_rmse:>12.2f}")
improvement = (baseline_mae - xgb_mae) / baseline_mae * 100
print(f"\nXGBoost improves MAE over baseline by {improvement:.1f}%")

# --- Feature importance ---
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 8 feature importances:")
print(importances.head(8))

# Save predictions for charting
test_out = test[['month']].copy()
test_out['Product_Category'] = df.loc[test.index, [c for c in df.columns if c.startswith('cat_')]].idxmax(axis=1).str.replace('cat_', '')
test_out['Region'] = df.loc[test.index, [c for c in df.columns if c.startswith('reg_')]].idxmax(axis=1).str.replace('reg_', '')
test_out['actual'] = y_test.values
test_out['baseline_pred'] = baseline_pred.values
test_out['xgb_pred'] = xgb_pred
test_out.to_csv('../data/processed/predictions.csv', index=False)
print("\nSaved -> predictions.csv")

importances.to_csv('../data/processed/feature_importance.csv')