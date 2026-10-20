import joblib
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODELS_DIR = Path(__file__).parent

_MODEL_FILE = _MODELS_DIR / "housing_model.pkl"
_PIPELINE_FILE = _MODELS_DIR / "housing_pipeline.pkl"

# ── Load model & pipeline once at import time ──────────────────────────────────
if not _MODEL_FILE.exists() or not _PIPELINE_FILE.exists():
    raise FileNotFoundError(
        "Model files not found. Run 'train model/train_model.py' first."
    )

_model = joblib.load(_MODEL_FILE)
_pipeline = joblib.load(_PIPELINE_FILE)


def predict_house_price(features: dict) -> float:
    """
    Run inference for a single house and return the predicted median house value.

    Parameters
    ----------
    features : dict
        Keys must match the HouseFeatures schema fields.

    Returns
    -------
    float
        Rounded predicted median house value.
    """
    input_df = pd.DataFrame([features])
    transformed = _pipeline.transform(input_df)
    prediction = _model.predict(transformed)
    return round(float(prediction[0]), 2)

