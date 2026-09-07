from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
import pandas as pd

from database import (
    SessionLocal,
    init_db,
    AnalysisRun,
    SalesSegment,
    AnalysisRisk,
    RecommendedAction
)
from ai.processing.aggregator import aggregate_sales
from ai.processing.features import create_features
from ai.models.predictor import predict_sales
from ai.analysis.context import build_business_context
from ai.groq.agent import run_agent

app = FastAPI(title="AI Business Analyser API")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def startup_event():
    init_db()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Only CSV files are supported.")

    try:
        df = pd.read_csv(file.file)
        monthly = aggregate_sales(df)
        features = create_features(monthly)
        predictions = predict_sales(features)

        business_context = build_business_context(
            df, monthly, features, predictions)
        payload = run_agent(business_context)

        # 1. Create Analysis Run Record
        exec_summary = payload.get("executive_summary", {})
        biz_overview = exec_summary.get("business_overview", {})
        fc_overview = exec_summary.get("forecast_overview", {})

        analysis_run = AnalysisRun(
            filename=file.filename,
            total_sales=biz_overview.get("total_sales"),
            total_orders=biz_overview.get("total_orders"),
            analysis_period=biz_overview.get("analysis_period"),
            average_monthly_sales=biz_overview.get("average_monthly_sales"),
            average_order_value=biz_overview.get("average_order_value"),
            forecast_model=fc_overview.get("model"),
            forecast_category=fc_overview.get("category"),
            forecast_region=fc_overview.get("region"),
            prediction_records=fc_overview.get("prediction_records"),
            avg_absolute_error=fc_overview.get("average_absolute_error"),
            largest_absolute_error=fc_overview.get("largest_absolute_error"),
            accuracy_assessment=fc_overview.get("accuracy_assessment")
        )
        db.add(analysis_run)
        db.flush()  # Populates analysis_run.id

        # 2. Store Segments
        cat_perf = payload.get("category_performance", {})
        reg_perf = payload.get("regional_performance", {})

        segments = [
            SalesSegment(analysis_id=analysis_run.id, entity_type="category",
                         performance_rank="best", **cat_perf.get("best_category", {})),
            SalesSegment(analysis_id=analysis_run.id, entity_type="category",
                         performance_rank="worst", **cat_perf.get("worst_category", {})),
            SalesSegment(analysis_id=analysis_run.id, entity_type="region",
                         performance_rank="best", **reg_perf.get("best_region", {})),
            SalesSegment(analysis_id=analysis_run.id, entity_type="region",
                         performance_rank="worst", **reg_perf.get("worst_region", {}))
        ]
        db.add_all(segments)

        # 3. Store Risks
        for r in payload.get("risks", []):
            db.add(AnalysisRisk(
                analysis_id=analysis_run.id,
                risk_type=r.get("type"),
                severity=r.get("severity"),
                description=r.get("description")
            ))

        # 4. Store Recommended Actions
        for a in payload.get("recommended_actions", []):
            db.add(RecommendedAction(
                analysis_id=analysis_run.id,
                priority=a.get("priority"),
                area=a.get("area"),
                action=a.get("action"),
                reason=a.get("reason")
            ))

        db.commit()

        return {
            "status": "success",
            "analysis_id": analysis_run.id,
            "filename": file.filename,
            "data": payload
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
