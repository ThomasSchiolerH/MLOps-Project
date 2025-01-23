from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import torch

from src.AVM.data import feature_engineering, FEATURE_COLUMNS
from src.AVM.model import PricePredictionModel


class HouseFeatures(BaseModel):
    FLOOR: Optional[str] = None
    CONSTRUCTION_YEAR: Optional[float] = None
    REBUILDING_YEAR: Optional[float] = None
    DISTANCE_LAKE: Optional[float] = None
    DISTANCE_HARBOUR: Optional[float] = None
    DISTANCE_COAST: Optional[float] = None
    HAS_ELEVATOR: Optional[str] = None
    AREA_TINGLYST: Optional[float] = None
    AREA_RESIDENTIAL: Optional[float] = None
    AREA_OTHER: Optional[float] = None
    AREA_COMMON_ACCESS_SHARE: Optional[float] = None
    AREA_CLOSED_COVER_OUTHOUSE: Optional[float] = None
    AREA_OPEN_BALCONY_ROOFTOP: Optional[float] = None
    MUNICIPALITY_CODE: Optional[float] = None
    ZIP_CODE: Optional[float] = None
    TRADE_YEAR: Optional[int] = None
    TRADE_MONTH: Optional[int] = None
    TRADE_DAY: Optional[int] = None
    NUMBER_ROOMS: Optional[float] = None
    TRADE_DATE: Optional[str] = None  # If you want to parse the date
    
    class Config:
        schema_extra = {
            "example": {
                "FLOOR": "1",
                "CONSTRUCTION_YEAR": 1969,
                "REBUILDING_YEAR": 0,
                "DISTANCE_LAKE": 2174.0,
                "DISTANCE_HARBOUR": 3844.9,
                "DISTANCE_COAST": 4558.8,
                "HAS_ELEVATOR": "false",
                "AREA_TINGLYST": 96,
                "AREA_RESIDENTIAL": 103,
                "AREA_OTHER": 0,
                "AREA_COMMON_ACCESS_SHARE": 0,
                "AREA_CLOSED_COVER_OUTHOUSE": 5.0,
                "AREA_OPEN_BALCONY_ROOFTOP": 0,
                "MUNICIPALITY_CODE": 147,
                "ZIP_CODE": 2000,
                "TRADE_YEAR": 2024,
                "TRADE_MONTH": 6,
                "TRADE_DAY": 30,
                "NUMBER_ROOMS": 3,
                "TRADE_DATE": "2024-06-30"
            }
        }

app = FastAPI()

model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    scaler = joblib.load("data/processed/scaler.pkl")

    my_model = PricePredictionModel(in_features=len(FEATURE_COLUMNS))
    state_dict = torch.load("src/AVM/price_model.pth", map_location=torch.device("cpu"))
    my_model.load_state_dict(state_dict)
    my_model.eval()

    model = my_model
    print("Artifacts loaded successfully.")

@app.get("/")
def read_root():
    return {"health_check": "OK", "model": "loaded" if model else "not_loaded"}

@app.post("/predict")
def predict_price(
    features: HouseFeatures = Body(
        ...,
        example={
            "FLOOR": "1",
            "CONSTRUCTION_YEAR": 1969,
            "REBUILDING_YEAR": 0,
            "DISTANCE_LAKE": 2174.0,
            "DISTANCE_HARBOUR": 3844.9,
            "DISTANCE_COAST": 4558.8,
            "HAS_ELEVATOR": "false",
            "AREA_TINGLYST": 96,
            "AREA_RESIDENTIAL": 103,
            "AREA_OTHER": 0,
            "AREA_COMMON_ACCESS_SHARE": 0,
            "AREA_CLOSED_COVER_OUTHOUSE": 5.0,
            "AREA_OPEN_BALCONY_ROOFTOP": 0,
            "MUNICIPALITY_CODE": 147,
            "ZIP_CODE": 2000,
            "TRADE_YEAR": 2024,
            "TRADE_MONTH": 6,
            "TRADE_DAY": 30,
            "NUMBER_ROOMS": 3,
            "TRADE_DATE": "2024-06-30"
        }
    )
):
    if model is None:
        return {"error": "Model not loaded!"}

    # 1) Convert Pydantic model to dict (raw data)
    input_dict = features.dict()

    # 2) Store an unscaled area variable for later use in total-price calc
    raw_area = input_dict.get("AREA_TINGLYST", 0.0)

    # 3) Build a DataFrame and run feature engineering
    df_infer = pd.DataFrame([input_dict])
    df_infer = feature_engineering(df_infer, log_transform=False)

    # 4) Scale numeric columns
    numeric_cols = [col for col in FEATURE_COLUMNS if col in df_infer.columns]
    df_infer[numeric_cols] = scaler.transform(df_infer[numeric_cols])

    # 5) Inference with PyTorch
    input_tensor = torch.tensor(df_infer[numeric_cols].values, dtype=torch.float)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_sqm_price = output.item()  # single scalar

    # 6) Compute total price = predicted_sqm_price * raw_area
    total_price = predicted_sqm_price * raw_area

    return {
        "predicted_sqm_price": predicted_sqm_price,
        "predicted_total_price": total_price
    }
