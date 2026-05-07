from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_factory import ModelFactory
from src.config import MODELS_DIR, TARGET_COLUMN
from src.features import FeatureEngineer

app = FastAPI(title="Forecasting Service API")

# Load best model info
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.json")

class ForecastRequest(BaseModel):
    state: str
    days: int = 7

class ForecastResponse(BaseModel):
    model_used: str
    predictions: list

@app.get("/")
def read_root():
    return {"message": "Forecasting Service is running"}

@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest):
    if not os.path.exists(BEST_MODEL_PATH):
        raise HTTPException(status_code=404, detail="Best model not found. Please train models first.")
    
    with open(BEST_MODEL_PATH, "r") as f:
        best_model_info = json.load(f)
        best_model_name = best_model_info["best_model"]
    
    model = ModelFactory.get_model(best_model_name)
    model.load()
    
    # In a real scenario, we would fetch the latest data for the state from a DB
    # For this demo, we'll return mock predictions or placeholder
    # Ideally, we need the last 'N' days of data to make future predictions
    
    # Placeholder: Just returning some dummy values for the requested days
    # Since we don't have the "future" test_data here without a DB/file reload
    predictions = [1000000.0 * (1.05 ** i) for i in range(request.days)] 
    
    return {
        "model_used": best_model_name,
        "predictions": predictions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
