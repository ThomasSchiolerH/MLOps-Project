import torch
import pytest
from project_name.model import PricePredictionModel
from project_name.data import PriceDataset, FEATURE_COLUMNS
from project_name.evaluate import evaluate_model
# Test model loading
def test_model_loading():
    model = PricePredictionModel(in_features=len(FEATURE_COLUMNS))
    assert model is not None, "Model failed to initialize"
    
    