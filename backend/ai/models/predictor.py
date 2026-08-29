import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ai.models.config import MODEL_FILE, FEATURE_COLUMNS


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

    # Features used by XGBoost
    X = features[FEATURE_COLUMNS]

    # Generate predictions
    predictions = model.predict(X)

    predictions = np.clip(
        predictions,
        0,
        None
    )

    # --------------------------------------------------
    # BUILD STRUCTURED PREDICTIONS
    # --------------------------------------------------

    results = features[
        [
            "month",
            "prediction_category",
            "prediction_region",
            "total_sales"
        ]
    ].copy()

    results["xgb_prediction"] = predictions

    results["xgb_error"] = (
        results["xgb_prediction"]
        - results["total_sales"]
    )

    # Rename identifiers
    results = results.rename(
        columns={
            "prediction_category":
                "Product_Category",

            "prediction_region":
                "Region",

            "total_sales":
                "actual_sales"
        }
    )

    return results.to_dict(
        orient="records"
    )