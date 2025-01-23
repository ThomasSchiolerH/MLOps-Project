import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.AVM.api import app, load_artifacts

# Create a client for testing
client = TestClient(app)

# Test data for /predict endpoint
example_data = {
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


# Trigger startup manually for tests
@pytest.fixture(scope="module", autouse=True)
def setup_startup():
    """Ensure the FastAPI app's startup events are triggered."""
    with TestClient(app):
        app.router.startup()
        load_artifacts()


# Test the health check endpoint
def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"health_check": "OK", "model": "loaded"}


# Test /predict endpoint with valid data
def test_predict_success():
    response = client.post("/predict", json=example_data)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_sqm_price" in data
    assert "predicted_total_price" in data
    assert isinstance(data["predicted_sqm_price"], float)
    assert isinstance(data["predicted_total_price"], float)


# Test /predict endpoint with invalid data type
def test_predict_invalid_data():
    invalid_data = example_data.copy()
    invalid_data["AREA_TINGLYST"] = "invalid_value"  # Invalid type

    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Unprocessable Entity
