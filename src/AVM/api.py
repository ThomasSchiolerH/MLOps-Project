from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional

import torch
import joblib
import pandas as pd

# Gradio imports
import gradio as gr
from gradio.routes import mount_gradio_app

# Import your existing data processing and model code
from src.AVM.data import feature_engineering, FEATURE_COLUMNS
from src.AVM.model import PricePredictionModel


# -----------------------------------------------------------------------------
# Pydantic model for your FastAPI /predict endpoint (unchanged)
# -----------------------------------------------------------------------------
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
                "TRADE_DATE": "2024-06-30",
            }
        }


# -----------------------------------------------------------------------------
# Initialize FastAPI
# -----------------------------------------------------------------------------
app = FastAPI()

model = None
scaler = None


# -----------------------------------------------------------------------------
# Load artifacts on startup
# -----------------------------------------------------------------------------
@app.on_event("startup")
def load_artifacts():
    global model, scaler

    # Load scaler
    scaler = joblib.load("data/processed/scaler.pkl")

    # Load PyTorch model
    my_model = PricePredictionModel(in_features=len(FEATURE_COLUMNS))
    state_dict = torch.load("src/AVM/price_model.pth", map_location=torch.device("cpu"))
    my_model.load_state_dict(state_dict)
    my_model.eval()

    model = my_model
    print("Artifacts loaded successfully.")


# -----------------------------------------------------------------------------
# Basic health check
# -----------------------------------------------------------------------------
@app.get("/")
def read_root():
    """Basic health check endpoint."""
    return {"health_check": "OK", "model": "loaded" if model else "not_loaded"}


# -----------------------------------------------------------------------------
# /predict endpoint (unchanged)
# -----------------------------------------------------------------------------
@app.post("/predict")
def predict_price(features: HouseFeatures = Body(...)):
    """Predicts the sqm price and total price given HouseFeatures."""
    if model is None:
        return {"error": "Model not loaded!"}

    # 1) Convert Pydantic model to dict (raw data)
    input_dict = features.dict()
    raw_area = input_dict.get("AREA_TINGLYST", 0.0)

    # 2) Feature engineering
    df_infer = pd.DataFrame([input_dict])
    df_infer = feature_engineering(df_infer, log_transform=False)

    # 3) Scale numeric columns
    numeric_cols = [col for col in FEATURE_COLUMNS if col in df_infer.columns]
    df_infer[numeric_cols] = scaler.transform(df_infer[numeric_cols])

    # 4) Inference
    input_tensor = torch.tensor(df_infer[numeric_cols].values, dtype=torch.float)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_sqm_price = output.item()

    total_price = predicted_sqm_price * raw_area

    return {"predicted_sqm_price": predicted_sqm_price, "predicted_total_price": total_price}


# -----------------------------------------------------------------------------
# Gradio Inference Function:
# Return a Markdown string for "nice" display, rather than JSON
# -----------------------------------------------------------------------------
def gradio_inference_fn(
    FLOOR: str,
    CONSTRUCTION_YEAR: float,
    REBUILDING_YEAR: float,
    DISTANCE_LAKE: float,
    DISTANCE_HARBOUR: float,
    DISTANCE_COAST: float,
    HAS_ELEVATOR: str,
    AREA_TINGLYST: float,
    AREA_RESIDENTIAL: float,
    AREA_OTHER: float,
    AREA_COMMON_ACCESS_SHARE: float,
    AREA_CLOSED_COVER_OUTHOUSE: float,
    AREA_OPEN_BALCONY_ROOFTOP: float,
    MUNICIPALITY_CODE: float,
    ZIP_CODE: float,
    TRADE_YEAR: int,
    TRADE_MONTH: int,
    TRADE_DAY: int,
    NUMBER_ROOMS: float,
    TRADE_DATE: str,
):
    if model is None:
        return "**Error**: Model not loaded!"

    # Build the raw input dict
    input_dict = {
        "FLOOR": FLOOR,
        "CONSTRUCTION_YEAR": CONSTRUCTION_YEAR,
        "REBUILDING_YEAR": REBUILDING_YEAR,
        "DISTANCE_LAKE": DISTANCE_LAKE,
        "DISTANCE_HARBOUR": DISTANCE_HARBOUR,
        "DISTANCE_COAST": DISTANCE_COAST,
        "HAS_ELEVATOR": HAS_ELEVATOR,
        "AREA_TINGLYST": AREA_TINGLYST,
        "AREA_RESIDENTIAL": AREA_RESIDENTIAL,
        "AREA_OTHER": AREA_OTHER,
        "AREA_COMMON_ACCESS_SHARE": AREA_COMMON_ACCESS_SHARE,
        "AREA_CLOSED_COVER_OUTHOUSE": AREA_CLOSED_COVER_OUTHOUSE,
        "AREA_OPEN_BALCONY_ROOFTOP": AREA_OPEN_BALCONY_ROOFTOP,
        "MUNICIPALITY_CODE": MUNICIPALITY_CODE,
        "ZIP_CODE": ZIP_CODE,
        "TRADE_YEAR": TRADE_YEAR,
        "TRADE_MONTH": TRADE_MONTH,
        "TRADE_DAY": TRADE_DAY,
        "NUMBER_ROOMS": NUMBER_ROOMS,
        "TRADE_DATE": TRADE_DATE,  # not strictly used, but included
    }

    raw_area = input_dict.get("AREA_TINGLYST", 0.0)

    # Feature engineering
    df_infer = pd.DataFrame([input_dict])
    df_infer = feature_engineering(df_infer, log_transform=False)

    # Scale numeric columns
    numeric_cols = [col for col in FEATURE_COLUMNS if col in df_infer.columns]
    df_infer[numeric_cols] = scaler.transform(df_infer[numeric_cols])

    # Model inference
    input_tensor = torch.tensor(df_infer[numeric_cols].values, dtype=torch.float)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_sqm_price = output.item()

    total_price = predicted_sqm_price * raw_area

    # Build a formatted Markdown string
    result_md = f"""
**Predicted Apartment Valuation**
- **Price per m²:** {predicted_sqm_price:,.2f} DKK
- **Total Price:** {total_price:,.2f} DKK
"""
    return result_md


# -----------------------------------------------------------------------------
# Create the Gradio Interface
# ...with default values for each field and a Markdown output
# -----------------------------------------------------------------------------
gradio_demo = gr.Interface(
    fn=gradio_inference_fn,
    title="Denmark Apartment Price Valuation",
    description="Enter apartment features to predict the price per m² and total price.",
    allow_flagging="never",
    inputs=[
        gr.Textbox(label="FLOOR", value="1"),
        gr.Number(label="CONSTRUCTION_YEAR", value=1969),
        gr.Number(label="REBUILDING_YEAR", value=0),
        gr.Number(label="DISTANCE_LAKE", value=2174.0),
        gr.Number(label="DISTANCE_HARBOUR", value=3844.9),
        gr.Number(label="DISTANCE_COAST", value=4558.8),
        gr.Radio(label="HAS_ELEVATOR", choices=["true", "false"], value="false"),
        gr.Number(label="AREA_TINGLYST", value=96),
        gr.Number(label="AREA_RESIDENTIAL", value=103),
        gr.Number(label="AREA_OTHER", value=0),
        gr.Number(label="AREA_COMMON_ACCESS_SHARE", value=0),
        gr.Number(label="AREA_CLOSED_COVER_OUTHOUSE", value=5.0),
        gr.Number(label="AREA_OPEN_BALCONY_ROOFTOP", value=0),
        gr.Number(label="MUNICIPALITY_CODE", value=147),
        gr.Number(label="ZIP_CODE", value=2000),
        gr.Number(label="TRADE_YEAR", value=2024),
        gr.Number(label="TRADE_MONTH", value=6),
        gr.Number(label="TRADE_DAY", value=30),
        gr.Number(label="NUMBER_ROOMS", value=3),
        gr.Textbox(label="TRADE_DATE (optional)", value="2024-06-30"),
    ],
    # Instead of JSON, we now use a Markdown display for "nice" formatting
    outputs=gr.Markdown(),
)

# -----------------------------------------------------------------------------
# Mount the Gradio interface at /gradio
# -----------------------------------------------------------------------------
app = mount_gradio_app(app, gradio_demo, path="/gradio")
