from fastapi.responses import JSONResponse
import joblib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent
_MODELS_DIR = _BASE_DIR / "models"

MODEL_FILE = _MODELS_DIR / "housing_model.pkl"
PIPELINE_FILE = _MODELS_DIR / "housing_pipeline.pkl"

# ── Load model & pipeline at startup ─────────────────────────────────────────
if not MODEL_FILE.exists() or not PIPELINE_FILE.exists():
    raise FileNotFoundError(
        "Model files not found. Run 'train model/train_model.py' first."
    )

model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

# ── FastAPI app 
app = FastAPI(
    title="California House Price Predictor",
    description="Predicts median house value based on California housing features.",
    version="1.0.0",
)

# Request / Response schemas - pydantic model to validate incoming data
class HouseFeatures(BaseModel):
    longitude: float = Field(..., example=-122.23)
    latitude: float = Field(..., example=37.88)
    housing_median_age: float = Field(..., example=41.0)
    total_rooms: float = Field(..., example=880.0)
    total_bedrooms: float = Field(..., example=129.0)
    population: float = Field(..., example=322.0)
    households: float = Field(..., example=126.0)
    median_income: float = Field(..., example=8.3252)
    ocean_proximity: str = Field(..., example="NEAR BAY")


class PredictionResponse(BaseModel):
    predicted_median_house_value: float


# ── Endpoints 
@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "California House Price Predictor is running."}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: HouseFeatures):
    """
    Predict the median house value for a given set of housing features.
    """
    import pandas as pd

    try:
        input_df = pd.DataFrame([data.model_dump()])
        transformed = pipeline.transform(input_df)
        prediction = model.predict(transformed)
        pred_value = round(float(prediction[0]), 2)
        return JSONResponse(
            status_code=200,
            content={"predicted_median_house_value": pred_value},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
