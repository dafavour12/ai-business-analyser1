from pathlib import Path
import pandas as pd

from ai.models.xgboost_model import train_model


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

FEATURES_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "features_v3.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "data"
    / "models"
)

MODEL_FILE = MODEL_DIR / "xgboost_v3.json"


# ============================================================
# LOAD FEATURES
# ============================================================

print("=" * 60)
print("TRAINING PRODUCTION XGBOOST MODEL")
print("=" * 60)

print("\nLoading features...")

df = pd.read_csv(
    FEATURES_FILE,
    parse_dates=["month"]
)

print(f"Feature rows: {len(df)}")


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining XGBoost...")

model = train_model(df)

print("Model trained successfully.")


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save_model(MODEL_FILE)

print("\nModel saved successfully.")

print("Location:")
print(MODEL_FILE)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("PRODUCTION MODEL READY")
print("=" * 60)