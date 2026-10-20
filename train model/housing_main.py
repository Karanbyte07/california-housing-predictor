import joblib
import pandas as pd
from pathlib import Path

_BASE_DIR = Path(__file__).parent.parent
_MODELS_DIR = _BASE_DIR / "models"
_PREDICTIONS_DIR = _BASE_DIR / "predictions"

MODEL_FILE = _MODELS_DIR / "housing_model.pkl"
PIPELINE_FILE = _MODELS_DIR / "housing_pipeline.pkl"
INPUT_FILE = _PREDICTIONS_DIR / "input_data.csv"
OUTPUT_FILE = _PREDICTIONS_DIR / "output_data.csv"


print("Model and pipeline already exists, loading them")
model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

input_data = pd.read_csv(INPUT_FILE)
transformed_input = pipeline.transform(input_data)
predictions = model.predict(transformed_input)
input_data["median_house_value"] = predictions

_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
input_data.to_csv(OUTPUT_FILE, index=False)
print(f"Predictions saved to {OUTPUT_FILE}")