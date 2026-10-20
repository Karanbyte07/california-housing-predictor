from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from models.predict import predict_house_price
from schemas.user_input import HouseFeatures


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="California House Price Predictor",
    description="Predicts median house value based on California housing features.",
    version="1.0.0",
)

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Welcome to the California House Price Predictor API"}

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "California House Price Predictor is running.", "version": "1.0.0"}

@app.post("/predict")
def predict(data: HouseFeatures):
    """
    Predict the median house value for a given set of housing features.
    """
    try:
        pred_value = predict_house_price(data.model_dump())
        return JSONResponse(
            status_code=200,
            content={"predicted_median_house_value": pred_value},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

