from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_FILE = (
    BASE_DIR
    / "data"
    / "models"
    / "xgboost_v3.json"
)


FEATURE_COLUMNS = [
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
model = XGBRegressor()

model.load_model(MODEL_FILE)


def predict_sales(features: pd.DataFrame):

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X = features[FEATURE_COLUMNS]

    predictions = model.predict(X)

    predictions = np.clip(
        predictions,
        0,
        None
    )

    return predictions