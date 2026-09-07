import os
from sqlalchemy import create_engine, Column, Integer, Float, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:dafavour12@localhost:5432/analytics_db"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 1. Main Analysis Run Table


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Business Overview
    total_sales = Column(Float)
    total_orders = Column(Integer)
    analysis_period = Column(String(50))
    average_monthly_sales = Column(Float)
    average_order_value = Column(Float)

    # Forecast Performance
    forecast_model = Column(String(50))
    forecast_category = Column(String(100))
    forecast_region = Column(String(100))
    prediction_records = Column(Integer)
    avg_absolute_error = Column(Float)
    largest_absolute_error = Column(Float)
    accuracy_assessment = Column(String(100))

    # Relationships
    segments = relationship(
        "SalesSegment", back_populates="analysis_run", cascade="all, delete-orphan")
    risks = relationship(
        "AnalysisRisk", back_populates="analysis_run", cascade="all, delete-orphan")
    actions = relationship(
        "RecommendedAction", back_populates="analysis_run", cascade="all, delete-orphan")


# 2. Sales Segments Table
class SalesSegment(Base):
    """Represents sales performance data segmented by category or region."""
    __tablename__ = "sales_segments"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analysis_runs.id"))
    entity_type = Column(String(50), index=True)  # 'category' or 'region'
    performance_rank = Column(String(20))          # 'best' or 'worst'
    name = Column(String(100), index=True)
    total_sales = Column(Float)
    sales_share_percent = Column(Float)
    total_orders = Column(Integer)
    order_share_percent = Column(Float)
    average_order_value = Column(Float)

    analysis_run = relationship("AnalysisRun", back_populates="segments")


# 3. Risks Table
class AnalysisRisk(Base):
    """Represents identified risks from the analysis."""
    __tablename__ = "analysis_risks"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analysis_runs.id"))
    risk_type = Column(String(100))
    severity = Column(String(50))
    description = Column(Text)

    analysis_run = relationship("AnalysisRun", back_populates="risks")


# 4. Recommended Actions Table
class RecommendedAction(Base):
    __tablename__ = "recommended_actions"

    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("analysis_runs.id"))
    priority = Column(String(50))
    area = Column(String(100))
    action = Column(Text)
    reason = Column(Text)

    analysis_run = relationship("AnalysisRun", back_populates="actions")


def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)
