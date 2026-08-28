import numpy as np
# import pandas as pd
import pandas as pd
from xgboost import XGBRegressor


MODEL_FEATURES = [
    "month_num",
    "quarter",
    "year",
    "time_idx",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "roll_mean_3",
    "roll_std_3",
    "roll_mean_6",
    "roll_std_6",
    "roll_mean_12",
    "roll_std_12",
    "roll_max_3",
    "roll_max_6",
    "cat_Clothing & Accessories",
    "cat_Furniture",
    "cat_Office Supplies",
    "cat_Technology",
    "reg_Asia Pacific",
    "reg_Europe",
    "reg_Middle East & Africa",
    "reg_North America",
    "reg_South America",
]


def create_model():
    """
    Create the XGBoost V3 regression model.
    """

    return XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )


def train_model(features: pd.DataFrame):
    """
    Train XGBoost using the provided feature DataFrame.
    """

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    X = features[MODEL_FEATURES]
    y = features["total_sales"]

    model = create_model()

    model.fit(X, y)

    return model


# ============================================================
# PREDICT
# ============================================================

def predict(model, features: pd.DataFrame):
    """
    Generate sales predictions using a trained model.
    """

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    X = features[MODEL_FEATURES]

    predictions = model.predict(X)

    # Sales cannot be negative.
    predictions = np.clip(
        predictions,
        0,
        None
    )

    return predictions