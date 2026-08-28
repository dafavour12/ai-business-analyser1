from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd

from ai.processing.aggregator import aggregate_sales
from ai.processing.features import create_features
from ai.models.predictor import predict_sales
from ai.analysis.context import build_business_context


app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported."
        )

    try:
        # 1. Read uploaded CSV
        df = pd.read_csv(file.file)

        # 2. Aggregate
        monthly = aggregate_sales(df)

        # 3. Features
        features = create_features(monthly)

        # 4. XGBoost prediction
        predictions = predict_sales(features)

        # 5. Build agent context
        business_context = build_business_context(
        df,
        monthly,
        features,
        predictions
        )

        return {
            "filename": file.filename,
            "raw_rows": len(df),
            "monthly_rows": len(monthly),
            "feature_rows": len(features),
            "prediction_count": len(predictions),
            "business_context": business_context
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )