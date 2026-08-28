import pandas as pd
import numpy as np


GROUP_COLS = ["Product_Category", "Region"]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create leakage-safe features from monthly aggregated sales data.

    Input:
        Monthly aggregated DataFrame from aggregate_sales()

    Output:
        Feature-engineered DataFrame ready for model training/prediction.
    """

    if df.empty:
        raise ValueError("Monthly aggregated data is empty.")

    df = df.copy()

    # ============================================================
    # PREPARE DATA
    # ============================================================

    df["month"] = pd.to_datetime(df["month"])

    df = df.sort_values(
        GROUP_COLS + ["month"]
    ).reset_index(drop=True)

    # ============================================================
    # CALENDAR FEATURES
    # ============================================================

    df["month_num"] = df["month"].dt.month
    df["quarter"] = df["month"].dt.quarter
    df["year"] = df["month"].dt.year

    df["time_idx"] = (
        (df["year"] - df["year"].min()) * 12
        + df["month_num"]
    )

    # ============================================================
    # CALENDAR-AWARE LAGS
    # ============================================================

    lookup = df.set_index(
        ["Product_Category", "Region", "month"]
    )["total_sales"]

    def get_lag(row, months):
        target_month = (
            row["month"] -
            pd.DateOffset(months=months)
        )

        key = (
            row["Product_Category"],
            row["Region"],
            target_month
        )

        try:
            return lookup.loc[key]
        except KeyError:
            return np.nan

    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df.apply(
            lambda row: get_lag(row, lag),
            axis=1
        )

    # ============================================================
    # ROLLING FEATURES
    # ============================================================

    grouped_sales = df.groupby(GROUP_COLS)["total_sales"]

    df["roll_mean_3"] = grouped_sales.transform(
        lambda s: s.shift(1).rolling(
            3,
            min_periods=1
        ).mean()
    )

    df["roll_std_3"] = grouped_sales.transform(
        lambda s: s.shift(1).rolling(
            3,
            min_periods=1
        ).std()
    )

    df["roll_mean_6"] = grouped_sales.transform(
        lambda s: s.shift(1).rolling(
            6,
            min_periods=1
        ).mean()
    )

    df["roll_std_6"] = grouped_sales.transform(
        lambda s: s.shift(1).rolling(
            6,
            min_periods=1
        ).std()
    )

    df["roll_mean_12"] = grouped_sales.transform(
        lambda s: s.shift(1).rolling(
            12,
            min_periods=1
        ).mean()
    )

    df["roll_std_12"] = grouped_sales.transform(
        lambda s: s.shift(1).rolling(
            12,
            min_periods=1
        ).std()
    )

    # ============================================================
    # ROLLING MAX
    # ============================================================

    df["roll_max_3"] = grouped_sales.transform(
        lambda s: s.shift(1).rolling(
            3,
            min_periods=1
        ).max()
    )

    df["roll_max_6"] = grouped_sales.transform(
        lambda s: s.shift(1).rolling(
            6,
            min_periods=1
        ).max()
    )

    # ============================================================
    # CATEGORICAL ENCODING
    # ============================================================

    df = pd.get_dummies(
        df,
        columns=[
            "Product_Category",
            "Region"
        ],
        prefix=[
            "cat",
            "reg"
        ]
    )

    # ============================================================
    # REMOVE ROWS WITHOUT REQUIRED HISTORY
    # ============================================================

    before = len(df)

    df = df.dropna(
        subset=[
            "lag_1",
            "lag_2",
            "lag_3"
        ]
    ).reset_index(drop=True)

    print(
        f"Dropped {before - len(df)} rows "
        f"without enough historical data"
    )

    # ============================================================
    # HANDLE REMAINING NaN VALUES
    # ============================================================

    rolling_cols = [
        "roll_std_3",
        "roll_std_6",
        "roll_std_12"
    ]

    for col in rolling_cols:
        df[col] = df[col].fillna(0)

    return df